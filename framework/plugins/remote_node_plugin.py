# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
remote_node_plugin.py -- Location-transparent GPU test execution plugin.

Wires NodePool into pytest. Builds the pool at session start, registers topology
with xdist workers, and provides: target_executor, multi_gpu_fixture,
multi_node_fixture, node_pool.

CLI options: --remote-node, --gpu-acquire-timeout, --gpu-health-metrics,
             --monitor-gpu, --gpu-drain-secs, --gpu-drain-timeout.

Priority order (target_executor):
    1. --no-gpu          → DryRunExecutor
    2. --container-mode  → ContainerExecutor
    3. --remote-node     → SshExecutor
    4. (default)         → LocalExecutor
"""

from __future__ import annotations

import logging
import os

import pytest

from framework.common.helpers import executor_log_path, gpu_monitor_log_path
from framework.nodes.node_pool import MultiGpuSlots, NodePool, NodeSlot
from framework.nodes.node_spec import HostConfigLoader, NodeSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI option registration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register remote-node and scheduling CLI options."""
    group = parser.getgroup("rocm-nodes", "ROCm node fleet options")
    group.addoption(
        "--remote-node",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Path to a host.yaml file defining remote GPU nodes.  "
            "When set, GPU tests are dispatched to the listed nodes via SSH.  "
            "When absent, tests run locally (same behaviour as before)."
        ),
    )
    group.addoption(
        "--gpu-acquire-timeout",
        action="store",
        type=float,
        default=180.0,
        metavar="SECS",
        help=(
            "Seconds a test will wait in the GPU slot queue before being skipped "
            "(default: 180 — 3 minutes).  The clock starts when the test's fixture "
            "begins trying to acquire a GPU slot.  For large suites or slow tests, "
            "increase this to at least (test_count / num_gpus) x avg_test_duration_secs."
        ),
    )
    group.addoption(
        "--gpu-health-metrics",
        nargs="?",
        const="",
        default=None,
        metavar="METRICS",
        help=(
            "Enable pre/post GPU health snapshots per test (captured by amd-smi "
            "once immediately BEFORE and once immediately AFTER each test body).  "
            "Without a value, uses health_metrics from rocm-test.toml.  "
            "Override metric list: --gpu-health-metrics temp,vram,ecc  "
            "Valid: temp,vram,util,ecc,clock.  Has no effect when --no-gpu is active."
        ),
    )
    group.addoption(
        "--monitor-gpu",
        action="store_true",
        default=False,
        help=(
            "Enable continuous background GPU metric polling while the test runs.  "
            "Samples are written to output/artifacts/executor-logs/<test>_gpu_monitor.log "
            "at the interval set by monitor_interval_secs in rocm-test.toml.  "
            "The poller stops automatically when the test ends.  "
            "Has no effect when --no-gpu is active."
        ),
    )
    group.addoption(
        "--gpu-drain-secs",
        action="store",
        type=float,
        default=0.5,
        metavar="SECS",
        help=(
            "Seconds to wait after a test finishes before releasing its GPU slot "
            "(sequential / non-xdist mode only, default 0.5).  "
            "In parallel (xdist) mode, amd-smi is polled instead — see --gpu-drain-timeout."
        ),
    )
    group.addoption(
        "--gpu-drain-timeout",
        action="store",
        type=float,
        default=30.0,
        metavar="SECS",
        help=(
            "Maximum seconds to wait for GPU VRAM to drain below the idle threshold "
            "during parallel (xdist) execution before releasing the slot anyway "
            "(default 30.0).  Has no effect in sequential mode."
        ),
    )


# ---------------------------------------------------------------------------
# GPU slot visibility helpers
# ---------------------------------------------------------------------------


def _write_to_session_log(session_log: str, msg: str) -> None:
    """Append *msg* to the session log file (append-safe, best-effort).

    Args:
        session_log: Absolute path to the session-wide log.
        msg:         Text line to append (newline added automatically).
    """
    import pathlib as _pathlib

    try:
        with _pathlib.Path(session_log).open("a", encoding="utf-8") as _f:
            _f.write(msg + "\n")
    except OSError:
        pass


def _console_slot_acquired(
    pool: NodePool,
    slot: NodeSlot,
    test_id: str,
    session_log: str | None = None,
) -> None:
    """Print and log a GPU slot acquisition event to the console.

    Always-on: printed directly so users see GPU→test mapping without
    needing ``--log-cli-level=INFO``.

    Format::

        [GPU ACQUIRE gw0] test_hip_runtime       → localhost    | GPU-0  | Pool: 1/2 in use | PID 12345

    Args:
        pool:        Active ``NodePool`` for pool utilization query.
        slot:        The slot that was just acquired.
        test_id:     Test function name.
        session_log: Session-wide log path for appending (optional).
    """
    available, total = pool.pool_status()
    in_use = total - available
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    msg = (
        f"[GPU ACQUIRE {worker}] {test_id[:24]:<24} → {slot.node_spec.label[:14]:<14}"
        f"| {slot.gpu_label:<8}| Pool: {in_use}/{total} in use | PID {os.getpid()}"
    )
    print(msg, flush=True)
    logger.info(msg)
    if session_log:
        _write_to_session_log(session_log, msg)


def _console_slot_released(
    pool: NodePool,
    slot: NodeSlot,
    test_id: str,
    elapsed: float,
    session_log: str | None = None,
) -> None:
    """Print and log a GPU slot release event to the console.

    Format::

        [GPU RELEASE gw0] test_hip_runtime       ← localhost    | GPU-0  | 12.3s | Pool: 2/2 in use

    Args:
        pool:        Active ``NodePool`` for pool utilization query.
        slot:        The slot that was just released.
        test_id:     Test function name.
        elapsed:     Wall-clock seconds the slot was held.
        session_log: Session-wide log path for appending (optional).
    """
    available, total = pool.pool_status()
    in_use = total - available
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    msg = (
        f"[GPU RELEASE {worker}] {test_id[:24]:<24} ← {slot.node_spec.label[:14]:<14}"
        f"| {slot.gpu_label:<8}| {elapsed:.1f}s | Pool: {in_use}/{total} in use"
    )
    print(msg, flush=True)
    logger.info(msg)
    if session_log:
        _write_to_session_log(session_log, msg)


def _console_multi_acquired(
    pool: NodePool,
    multi: MultiGpuSlots,
    test_id: str,
    session_log: str | None = None,
) -> None:
    """Print and log a multi-GPU slot acquisition event.

    Args:
        pool:        Active ``NodePool`` for pool utilization query.
        multi:       The ``MultiGpuSlots`` group that was acquired.
        test_id:     Test function name.
        session_log: Session-wide log path for appending (optional).
    """
    available, total = pool.pool_status()
    in_use = total - available
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    gpu_label = f"GPU-{','.join(str(s.gpu_info.index) for s in multi.slots)}"
    msg = (
        f"[GPU ACQUIRE {worker}] {test_id[:24]:<24} → {multi.node_spec.label[:14]:<14}"
        f"| {gpu_label:<10}| Pool: {in_use}/{total} in use | PID {os.getpid()}"
    )
    print(msg, flush=True)
    logger.info(msg)
    if session_log:
        _write_to_session_log(session_log, msg)


def _console_multi_released(
    pool: NodePool,
    multi: MultiGpuSlots,
    test_id: str,
    elapsed: float,
    session_log: str | None = None,
) -> None:
    """Print and log a multi-GPU slot release event.

    Args:
        pool:        Active ``NodePool`` for pool utilization query.
        multi:       The ``MultiGpuSlots`` group that was released.
        test_id:     Test function name.
        elapsed:     Wall-clock seconds the slots were held.
        session_log: Session-wide log path for appending (optional).
    """
    available, total = pool.pool_status()
    in_use = total - available
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    gpu_label = f"GPU-{','.join(str(s.gpu_info.index) for s in multi.slots)}"
    msg = (
        f"[GPU RELEASE {worker}] {test_id[:24]:<24} ← {multi.node_spec.label[:14]:<14}"
        f"| {gpu_label:<10}| {elapsed:.1f}s | Pool: {in_use}/{total} in use"
    )
    print(msg, flush=True)
    logger.info(msg)
    if session_log:
        _write_to_session_log(session_log, msg)


# ---------------------------------------------------------------------------
# Private helpers shared by target_executor / multi_gpu_fixture / multi_node_fixture
# ---------------------------------------------------------------------------


def _resolve_session_log(config: pytest.Config, framework_config) -> str:
    """Return the session-wide log path as an absolute path, creating the directory if needed.

    Priority: ``config._session_log_path`` (set by ``reports_plugin.pytest_configure``)
    → ``framework_config.framework.session_log`` (from ``rocm-test.toml`` or code default).
    Always returns an absolute path so xdist workers writing from a different CWD land
    in the same file as the master process.
    """
    session_log = getattr(config, "_session_log_path", None)
    if session_log is None:
        import pathlib as _pathlib

        session_log = str(_pathlib.Path(framework_config.framework.session_log).resolve())
        _pathlib.Path(session_log).parent.mkdir(parents=True, exist_ok=True)
    return session_log


def _resolve_rock_dir(config: pytest.Config) -> str | None:
    """Resolve the TheRock/ROCm install path from CLI / env."""
    return (
        config.getoption("--rock-dir", default=None)
        or os.environ.get("ROCK_DIR")
        or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
    )


def _resolve_health_metrics(config: pytest.Config, framework_config) -> set[str] | None:
    """Return metric set for pre/post health snapshots, or None if disabled.

    Returns ``None`` when ``--gpu-health-metrics`` is not passed (feature off).
    Returns the toml default set when the flag is passed without a value.
    Returns a custom set when a comma-separated value is supplied.

    Args:
        config:           Active pytest config.
        framework_config: Loaded FrameworkConfig (provides gpu.health_metrics).

    Returns:
        Set of metric name strings, or None when the feature is disabled.
    """
    raw = config.getoption("--gpu-health-metrics", default=None)
    if raw is None:
        return None  # feature disabled
    if not raw:  # bare flag, no value → use toml defaults
        return set(framework_config.gpu.health_metrics)
    return {m.strip() for m in raw.split(",") if m.strip()}


def _resolve_monitor_config(framework_config) -> tuple[set[str], float, float]:
    """Return ``(metrics, interval_secs, duration_secs)`` for the background monitor.

    All values come from ``[gpu]`` section of rocm-test.toml (or env overrides).

    Args:
        framework_config: Loaded FrameworkConfig.

    Returns:
        Tuple of (metric set, poll interval, max duration in seconds).
        ``duration_secs == 0.0`` means the monitor runs until the test ends.
    """
    return (
        set(framework_config.gpu.monitor_metrics),
        framework_config.gpu.monitor_interval_secs,
        framework_config.gpu.monitor_duration_secs,
    )


def _monitoring_executor(slot_or_node_spec, rock_dir: str | None):
    """Return the right executor for amd-smi monitoring (no GPU env injection).

    Local node → ``CpuExecutor`` with optional ``rock_dir/bin`` prepended to PATH.
    Remote node → the slot's existing ``SshExecutor`` (``slot._ssh``), reused
    so no second SSH connection is opened.

    ``LocalExecutor`` is intentionally NOT used here: it injects
    ``ROCR_VISIBLE_DEVICES`` which restricts amd-smi's device view.

    Args:
        slot_or_node_spec: A ``NodeSlot`` (has ``.node_spec`` and ``._ssh``)
                           or a bare ``NodeSpec``.
        rock_dir:          Optional TheRock/ROCm install path.

    Returns:
        ``AbstractExecutor`` appropriate for running amd-smi on that node.
    """
    from framework.nodes.node_pool import NodeSlot

    spec = slot_or_node_spec.node_spec if isinstance(slot_or_node_spec, NodeSlot) else slot_or_node_spec
    if spec.is_local:
        from framework.executors.cpu_executor import CpuExecutor

        env: dict[str, str] = {}
        if rock_dir:
            env["PATH"] = f"{os.path.join(rock_dir, 'bin')}:{os.environ.get('PATH', '')}"
        return CpuExecutor(env_overrides=env, suppress_output_log=True)
    # Remote: reuse already-open SSH session — no new connection needed.
    return slot_or_node_spec._ssh


def _write_session_separator(session_log: str, test_name: str, phase: str, elapsed: float | None = None) -> None:
    """Append a test boundary separator to the session log.

    Writes a clearly delimited START or END block so the accumulated log is
    easy to navigate even when many tests run in sequence.

    Args:
        session_log: Absolute path to the session log file.
        test_name:   Test node name (``request.node.name``).
        phase:       ``"START"`` or ``"END"``.
        elapsed:     Wall-clock seconds the test held the GPU slot (END only).
    """
    import datetime as _dt
    import pathlib as _pathlib

    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    separator = "=" * 72
    if phase == "START":
        line = f"\n{separator}\n=== [TEST START] {test_name} | {ts} ===\n{separator}\n"
    else:
        dur = f" | {elapsed:.1f}s" if elapsed is not None else ""
        line = f"{separator}\n=== [TEST END]   {test_name}{dur} | {ts} ===\n{separator}\n"
    try:
        with _pathlib.Path(session_log).open("a", encoding="utf-8") as _f:
            _f.write(line)
    except OSError:
        pass


def _drain_gpu_slots(
    config: pytest.Config,
    rock_dir: str | None,
    gpu_indices: list[int],
) -> None:
    """Wait for VRAM to drain on *gpu_indices* before releasing slots.

    In parallel (xdist) mode, polls ``amd-smi`` until VRAM drops below the
    idle threshold.  In sequential mode, a short fixed sleep is sufficient.

    Args:
        config:      Active pytest config (reads --gpu-drain-* options).
        rock_dir:    TheRock/ROCm install path for ``amd-smi`` lookup.
        gpu_indices: Physical GPU ordinals to drain (one per slot held).
    """
    import time as _time

    xdist_active = getattr(config.option, "numprocesses", None) not in (None, 0, "no", "")
    if xdist_active:
        from framework.gpu.drain import GpuDrainChecker

        drain_timeout = config.getoption("--gpu-drain-timeout", default=30.0)
        checker = GpuDrainChecker(rock_dir=rock_dir)
        for idx in gpu_indices:
            checker.wait_for_drain(gpu_index=idx, timeout_secs=drain_timeout)
    else:
        drain_secs = config.getoption("--gpu-drain-secs", default=0.5)
        if drain_secs > 0:
            _time.sleep(drain_secs)


# ---------------------------------------------------------------------------
# Session-level hooks
# ---------------------------------------------------------------------------


class _XdistTopologyPlugin:
    """Provides xdist-specific hooks for GPU topology passing master → workers.

    Registered only when ``pytest-xdist`` is installed.  When xdist is absent,
    this class is not registered and the hook never fires.
    """

    def __init__(self, config: pytest.Config) -> None:
        self._config = config

    def pytest_configure_node(self, node) -> None:
        """Pass GPU topology from xdist master to workers.

        Serializes the master's ``NodePool`` topology to JSON and injects it into
        ``workerinput`` so workers can rebuild the pool without re-running GPU
        detection.  This prevents NxM redundant SSH calls when many workers start.
        """
        import json

        pool: NodePool | None = getattr(self._config, "_node_pool", None)
        if pool is None:
            node.workerinput["_node_pool_topology"] = ""
            return

        # Serialize pre-detected GPU list per node so workers don't re-run detection.
        gpu_topology: dict[str, list[dict]] = {}
        for spec in pool.node_specs:
            alloc = pool._allocators.get(spec.label)
            if alloc is not None:
                gpu_topology[spec.label] = [
                    {
                        "index": g.index,
                        "arch": g.arch,
                        "vram_mb": g.vram_mb,
                        "numa_node": g.numa_node,
                    }
                    for g in alloc._pool
                ]

        topology = {
            "nodes": [
                {
                    "hostname": spec.hostname,
                    "username": spec.username,
                    "password": spec.password,
                    "ssh_key": spec.ssh_key,
                    "gpu_arch": spec.gpu_arch,
                    "label": spec.label,
                }
                for spec in pool.node_specs
            ],
            "gpu_topology": gpu_topology,
            "rock_dir": pool._rock_dir,
            "headroom_gb": pool._headroom_gb,
        }
        node.workerinput["_node_pool_topology"] = json.dumps(topology)
        # Inject the absolute session log path so workers write to the same file
        # as the master regardless of their working directory.
        node.workerinput["_session_log_path"] = getattr(self._config, "_session_log_path", None) or ""


def pytest_configure(config: pytest.Config) -> None:
    """Build NodePool and attach it to the pytest config object.

    LOCAL mode  (no ``--remote-node``): single NodeSpec for localhost.
    REMOTE mode (``--remote-node=PATH``): NodeSpec list from host.yaml.

    In both cases, GPU detection runs at this point.  The result is stored on
    ``config._node_pool`` for use by the ``node_pool`` fixture and by
    ``pytest_collection_modifyitems`` for xdist grouping.

    Skips ``NodePool`` construction when ``--no-gpu`` is active (no hardware
    needed) or when the worker is a secondary xdist worker (it receives the
    topology JSON from the master instead of re-detecting).
    """
    # Skip during collection-only or no-gpu mode
    if config.getoption("--no-gpu", default=False):
        config._node_pool = None  # type: ignore[attr-defined]
        return

    # xdist secondary workers reconstruct the pool from serialized data
    # injected by the master via pytest_configure_node; skip full detection.
    if hasattr(config, "workerinput"):
        _configure_worker_pool(config)
        return

    # Remove stale pending-tracker files from any crashed previous session.
    # Must run on the master before workers are spawned (the workerinput check
    # above ensures workers never reach this point).  See pending_tracker.py
    # for why monotonic-clock resets after reboot make normal expiry unreliable.
    from framework.nodes.pending_tracker import cleanup_session_start

    cleanup_session_start()

    rock_dir = _resolve_rock_dir(config)

    remote_node_path = config.getoption("--remote-node", default=None)
    headroom_gb = config.getoption("--vram-headroom-gb", default=2.0)

    if remote_node_path:
        node_specs = HostConfigLoader.load(remote_node_path)
        logger.info("NodePool: REMOTE mode — %d node(s) from %s", len(node_specs), remote_node_path)
    else:
        node_specs = [NodeSpec(hostname="localhost", label="localhost")]
        logger.info("NodePool: LOCAL mode — detecting GPUs on localhost")

    from framework.config.loader import load_config as _load_cfg

    _cfg = _load_cfg(config_path=config.getoption("--rocm-config", default=None))

    try:
        pool = NodePool(
            node_specs=node_specs,
            rock_dir=rock_dir,
            headroom_gb=headroom_gb,
            detect_timeout=60.0,
            artifact_dir=_cfg.framework.artifact_dir,
        )
    except Exception as exc:
        logger.warning("NodePool construction failed: %s — tests requiring GPUs will skip", exc)
        config._node_pool = None  # type: ignore[attr-defined]
        return

    config._node_pool = pool  # type: ignore[attr-defined]

    total = pool.total_gpu_slots()
    if total == 0:
        topo = pool.topology_summary() if hasattr(pool, "topology_summary") else "unavailable"
        pytest.exit(
            "\n[rocm-test] ERROR: All GPU detection methods (lspci, KFD sysfs, amd-smi) "
            f"returned 0 devices on all nodes. 0 slots available. Topology: {topo}\n"
            "  Common causes in containers:\n"
            "    1. GPU not passed through — verify container --device /dev/kfd --device /dev/dri flags\n"
            "    2. ROCm amdgpu driver not loaded on the host — check: lsmod | grep amdgpu\n"
            "    3. /sys/class/kfd sysfs not exposed inside the container namespace\n"
            "    4. amd-smi not found at <rock_dir>/bin/amd-smi — verify --rock-dir path\n"
            "    5. --gpu-arch filter mismatch — lspci reports arch='unknown'; only KFD/amd-smi\n"
            "       return real arch strings; if --gpu-arch is set, lspci-only detections are excluded\n"
            "  See diagnostic output above for raw lspci and KFD sysfs details.\n"
            "  Use --no-gpu to run without hardware.",
            returncode=3,
        )

    topo = pool.topology_summary()
    print(
        f"\n[rocm-test] GPU topology: {topo}. "
        f"Total slots: {total}. "
        f"{'Add -n ' + str(total) + ' for parallel execution.' if total > 1 else ''}"
    )

    # Register xdist master→worker topology passing only when xdist is available
    try:
        import xdist  # noqa: F401

        config.pluginmanager.register(_XdistTopologyPlugin(config), name="remote_node_xdist_topology")
        logger.debug("remote_node_plugin: registered xdist topology integration")
    except ImportError:
        pass


def _configure_worker_pool(config: pytest.Config) -> None:
    """Reconstruct NodePool on an xdist worker from master-serialized topology.

    The master serializes the pool topology **including the already-detected
    GPU list** as JSON in ``workerinput`` (injected via ``pytest_configure_node``).
    Workers rebuild the pool using the pre-detected GPU list — no SSH connections
    or ``lspci``/``amd-smi`` calls are made.  This prevents flaky per-worker
    re-detection from silently setting ``_node_pool = None`` and skipping tests.
    """
    import json

    from framework.gpu.detector import GpuInfo

    topology_json = config.workerinput.get("_node_pool_topology")  # type: ignore[attr-defined]
    if not topology_json:
        logger.warning("xdist worker: empty topology from master — GPU tests will skip on this worker")
        config._node_pool = None  # type: ignore[attr-defined]
        return

    try:
        topology = json.loads(topology_json)
    except (ValueError, TypeError) as exc:
        logger.warning("xdist worker: failed to parse topology JSON: %s — GPU tests will skip", exc)
        config._node_pool = None  # type: ignore[attr-defined]
        return

    # Rebuild NodeSpec list from serialized data
    node_specs = [
        NodeSpec(
            hostname=entry["hostname"],
            username=entry.get("username"),
            password=entry.get("password"),
            ssh_key=entry.get("ssh_key"),
            gpu_arch=entry.get("gpu_arch"),
            label=entry["label"],
        )
        for entry in topology.get("nodes", [])
    ]

    if not node_specs:
        logger.warning("xdist worker: no node specs in topology — GPU tests will skip")
        config._node_pool = None  # type: ignore[attr-defined]
        return

    # Deserialize pre-detected GPU list so workers skip re-detection entirely.
    raw_gpu_topology: dict = topology.get("gpu_topology", {})
    prefilled_gpus: dict[str, list[GpuInfo]] = {}
    for label, gpu_list in raw_gpu_topology.items():
        prefilled_gpus[label] = [
            GpuInfo(
                index=g["index"],
                arch=g["arch"],
                vram_mb=g["vram_mb"],
                numa_node=g.get("numa_node", -1),
            )
            for g in gpu_list
        ]

    rock_dir = topology.get("rock_dir")
    headroom_gb = topology.get("headroom_gb", 0.0)

    try:
        pool = NodePool(
            node_specs=node_specs,
            rock_dir=rock_dir,
            headroom_gb=headroom_gb,
            prefilled_gpus=prefilled_gpus or None,
        )
        config._node_pool = pool  # type: ignore[attr-defined]
        logger.info(
            "xdist worker: NodePool ready — %d node(s), %d GPU slot(s)",
            len(node_specs),
            pool.total_gpu_slots(),
        )
    except Exception as exc:
        logger.warning("xdist worker: NodePool reconstruction failed: %s — GPU tests will skip", exc)
        config._node_pool = None  # type: ignore[attr-defined]


def _cleanup_gpu_lock_files(lock_dir: str = "output/.gpu-locks") -> None:
    """Remove all GPU lock and metadata files from *lock_dir*.

    Called at session end (master process only) to leave a clean state.
    Stale lock files from crashed workers are already harmless (the OS releases
    flock() on process exit), but removing them prevents confusion when
    inspecting the lock directory between sessions.

    Args:
        lock_dir: Directory containing ``.lock`` and ``.info`` files.
    """
    import glob as _glob

    removed = 0
    for pattern in ("*.lock", "*.info", "*_pending.json", "*_pending.lock"):
        for path in _glob.glob(os.path.join(lock_dir, pattern)):
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(
            "NodePool: removed %d GPU lock/metadata files from %s at session end",
            removed,
            lock_dir,
        )


def _write_session_summary(session: pytest.Session) -> None:
    """Append a SUMM line to session.log at the end of the master process.

    Reads the pytest terminal summary counters and writes one structured line
    so the session log is trivially grep-able for pass/fail totals.

    Args:
        session: The active pytest session (provides counts via terminalreporter).
    """
    try:
        from framework.logging.test_logger import _ts

        # _session_log_path is already resolved in pytest_configure — no load_config() needed.
        session_log = getattr(session.config, "_session_log_path", None)
        if not session_log:
            return

        tr = session.config.pluginmanager.getplugin("terminalreporter")
        if tr is None:
            return
        passed = len(tr.stats.get("passed", []))
        failed = len(tr.stats.get("failed", []))
        error = len(tr.stats.get("error", []))
        skipped = len(tr.stats.get("skipped", []))
        total = passed + failed + error + skipped
        line = f"SUMM  {_ts()} total={total} passed={passed} failed={failed}" f" error={error} skipped={skipped}\n"
        import pathlib as _pathlib

        with _pathlib.Path(session_log).open("a", encoding="utf-8") as _f:
            _f.write(line)
    except Exception:  # pylint: disable=broad-except
        pass  # session summary is best-effort — never crash the process


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Close SSH sessions, write session summary, and clean up GPU lock files."""
    pool: NodePool | None = getattr(session.config, "_node_pool", None)
    if pool is not None:
        pool.close_ssh_sessions()
        logger.info("NodePool: SSH sessions closed at session end")

    # Clean up lock files and write session summary on the master process only.
    if not hasattr(session.config, "workerinput"):
        _cleanup_gpu_lock_files()
        print("\n[rocm-test] GPU lock files cleaned up at session end.", flush=True)
        _write_session_summary(session)


# ---------------------------------------------------------------------------
# Private helpers — shared by target_executor, multi_gpu_fixture, multi_node_fixture
# ---------------------------------------------------------------------------


def _resolve_test_context(request, framework_config, config):
    """Extract all per-test configuration in one place.

    Returns a dict with all keys needed by the three fixture paths so that
    the marker/option reads are not duplicated across target_executor,
    multi_gpu_fixture, and multi_node_fixture.

    Returns:
        dict with keys: test_name, acquire_timeout, monitor_gpu,
        health_metrics, mon_metrics, mon_interval, mon_duration,
        log_path, session_log, rock_dir, gpu_count_marker,
        vram_required_gb, is_multi_gpu, is_multi_node.
    """
    test_name = request.node.name
    acquire_timeout = config.getoption("--gpu-acquire-timeout", default=30.0)
    monitor_gpu = config.getoption("--monitor-gpu", default=False)
    health_metrics = _resolve_health_metrics(config, framework_config)
    mon_metrics, mon_interval, mon_duration = _resolve_monitor_config(framework_config)
    log_path = executor_log_path(framework_config.framework.artifact_dir, test_name, request.node.nodeid)
    session_log = _resolve_session_log(config, framework_config)
    rock_dir = _resolve_rock_dir(config)
    gpu_count_marker = request.node.get_closest_marker("gpu_count")
    vram_marker = request.node.get_closest_marker("gpu_vram")
    vram_required_gb = float(vram_marker.args[0]) if vram_marker else 0.0
    return {
        "test_name": test_name,
        "acquire_timeout": acquire_timeout,
        "monitor_gpu": monitor_gpu,
        "health_metrics": health_metrics,
        "mon_metrics": mon_metrics,
        "mon_interval": mon_interval,
        "mon_duration": mon_duration,
        "log_path": log_path,
        "session_log": session_log,
        "rock_dir": rock_dir,
        "gpu_count_marker": gpu_count_marker,
        "vram_required_gb": vram_required_gb,
        "is_multi_gpu": request.node.get_closest_marker("hw.multi_gpu") is not None,
        "is_multi_node": request.node.get_closest_marker("e2e.multinode") is not None,
    }


def _acquire_single(node_pool, ctx):
    """Acquire a single GPU slot, translating errors to pytest outcomes.

    Args:
        node_pool: Active ``NodePool``.
        ctx:       Context dict from ``_resolve_test_context``.

    Returns:
        ``NodeSlot`` on success.

    Raises:
        pytest.skip: When slots are transiently unavailable.
        pytest.fail: When no GPUs are detected at all.
    """
    try:
        return node_pool.acquire_slot(
            vram_required_gb=ctx["vram_required_gb"],
            wait_timeout_secs=ctx["acquire_timeout"],
            test_id=ctx["test_name"],
        )
    except RuntimeError as exc:
        if node_pool.total_gpu_slots() == 0:
            pytest.fail(
                f"No GPUs detected on this platform — GPU driver or hardware missing. "
                f"Topology: {node_pool.topology_summary()}"
            )
        else:
            pytest.skip(f"TEST PLATFORM missing required GPUs (transient): {exc}")
    return None  # unreachable; satisfies type checker


def _acquire_multi_gpu(node_pool, ctx, count):
    """Acquire N GPUs from a single node, translating errors to pytest.skip.

    Args:
        node_pool: Active ``NodePool``.
        ctx:       Context dict from ``_resolve_test_context``.
        count:     Number of GPUs to acquire.

    Returns:
        ``MultiGpuSlots`` on success.

    Raises:
        pytest.skip: When fewer than *count* GPUs are available.
    """
    try:
        return node_pool.acquire_slots(
            count=count,
            vram_required_gb=ctx["vram_required_gb"],
            wait_timeout_secs=ctx["acquire_timeout"],
            test_id=ctx["test_name"],
        )
    except RuntimeError as exc:
        pytest.skip(f"TEST PLATFORM missing required GPUs: {exc}")
    return None  # unreachable


def _setup_monitoring(ctx, framework_config, multi_slots_list, is_multi_node_shape):
    """Set up health snapshots and background monitors before the test body.

    Returns a tuple of (health_monitors, pre_health_maps, bg_monitors)
    that must be passed to ``_teardown_monitoring`` after the test body.

    Args:
        ctx:                Context dict from ``_resolve_test_context``.
        framework_config:   Session-scoped ``FrameworkConfig``.
        multi_slots_list:   ``list[MultiGpuSlots]`` — each element is one node's group.
        is_multi_node_shape: When True, uses per-node labels for background monitor log files.

    Returns:
        Tuple ``(health_monitors, pre_health_maps, bg_monitors)`` for teardown.
    """
    health_metrics = ctx["health_metrics"]
    monitor_gpu = ctx["monitor_gpu"]
    mon_metrics, mon_interval, mon_duration = ctx["mon_metrics"], ctx["mon_interval"], ctx["mon_duration"]
    test_name = ctx["test_name"]
    rock_dir = ctx["rock_dir"]

    node_mon_execs = [_monitoring_executor(multi.slots[0], rock_dir) for multi in multi_slots_list]

    # Pre-test health snapshots
    pre_health_maps: list[dict] = [{} for _ in multi_slots_list]
    health_monitors: list = []
    if health_metrics is not None:
        from framework.gpu.monitor import GpuMonitor  # pylint: disable=import-outside-toplevel

        for i, multi in enumerate(multi_slots_list):
            hm = GpuMonitor(executor=node_mon_execs[i], metrics=health_metrics)
            health_monitors.append(hm)
            for slot in multi.slots:
                pre = hm.snapshot(slot.gpu_info.index)
                pre_health_maps[i][slot.gpu_info.index] = pre
                logger.info("[pre-health] %s", hm.summary_line(pre, "pre"))

    # Background continuous monitor
    bg_monitors: list = []
    if monitor_gpu:
        from framework.gpu.monitor import GpuBackgroundMonitor  # pylint: disable=import-outside-toplevel

        for i, multi in enumerate(multi_slots_list):
            bgm = GpuBackgroundMonitor(
                executor=node_mon_execs[i],
                metrics=mon_metrics,
                interval_secs=mon_interval,
                duration_secs=mon_duration,
            )
            if is_multi_node_shape:
                node_label = multi.node_spec.label.replace(".", "_")
                log_suffix = f"{test_name}_{node_label}"
            else:
                log_suffix = test_name
            bgm.start(
                gpu_indices=[s.gpu_info.index for s in multi.slots],
                log_path=gpu_monitor_log_path(framework_config.framework.artifact_dir, log_suffix),
            )
            bg_monitors.append(bgm)

    return health_monitors, pre_health_maps, bg_monitors


def _teardown_monitoring(ctx, multi_slots_list, health_monitors, pre_health_maps, bg_monitors, elapsed):
    """Tear down monitors and log health deltas after the test body.

    Args:
        ctx:              Context dict from ``_resolve_test_context``.
        multi_slots_list: ``list[MultiGpuSlots]`` matching the list passed to ``_setup_monitoring``.
        health_monitors:  List returned by ``_setup_monitoring``.
        pre_health_maps:  List returned by ``_setup_monitoring``.
        bg_monitors:      List returned by ``_setup_monitoring``.
        elapsed:          Wall-clock seconds from test start to finish.
    """
    health_metrics = ctx["health_metrics"]

    for bgm in bg_monitors:
        bgm.stop()

    if health_metrics is not None and health_monitors:
        for i, multi in enumerate(multi_slots_list):
            hm = health_monitors[i]
            for slot in multi.slots:
                pre = pre_health_maps[i].get(slot.gpu_info.index)
                post = hm.snapshot(slot.gpu_info.index)
                logger.info("[post-health] %s", hm.summary_line(post, "post"))
                if pre is not None:
                    logger.info("[health-delta] %s", hm.delta_line(pre, post))


def _acquire_and_yield(
    *,
    multi_list: list[MultiGpuSlots],
    ctx: dict,
    framework_config,
    config,
    node_pool: NodePool,
    is_multi_node: bool,
):
    """Shared generator: set up monitoring, yield NodeExecutorGroup, tear down.

    Normalises the single-slot and multi-slot shapes into a uniform list so the
    try/finally block is written exactly once. Callers supply the acquired
    ``MultiGpuSlots`` list and the resolved context dict.

    Args:
        multi_list: One or more acquired ``MultiGpuSlots`` (single-GPU callers pass a
            one-element list; multi-node callers pass one slot per node).
        ctx: Resolved test context from ``_resolve_test_context()``. Required keys:
            ``test_name`` (str), ``log_path`` (str), ``session_log`` (str),
            ``rock_dir`` (str), plus monitoring keys consumed by ``_setup_monitoring``.
        framework_config: Session-scoped ``FrameworkConfig``.
        config: ``pytest.Config`` (for drain/monitoring options).
        node_pool: Active ``NodePool`` used to release slots in the finally block.
        is_multi_node: ``True`` for ``e2e.multinode`` shapes (one executor per node);
            ``False`` for single-node multi-GPU shapes.
    """
    import time as _time  # pylint: disable=import-outside-toplevel

    from framework.executors.executor_group import NodeExecutorGroup  # pylint: disable=import-outside-toplevel

    test_name = ctx["test_name"]
    log_path = ctx["log_path"]
    session_log = ctx["session_log"]
    rock_dir = ctx["rock_dir"]

    for multi in multi_list:
        multi._log_path = log_path
        multi._session_log_path = session_log
        _console_multi_acquired(node_pool, multi, test_name, session_log)
    _write_session_separator(session_log, test_name, "START")

    health_monitors, pre_health_maps, bg_monitors = _setup_monitoring(
        ctx, framework_config, multi_list, is_multi_node_shape=is_multi_node
    )
    executors = [
        m.make_executor(test_id=test_name, log_path=log_path, session_log_path=session_log) for m in multi_list
    ]
    _t = _time.monotonic()
    try:
        yield NodeExecutorGroup(executors)
    finally:
        elapsed = _time.monotonic() - _t
        _teardown_monitoring(ctx, multi_list, health_monitors, pre_health_maps, bg_monitors, elapsed)
        for _exec in executors:
            if getattr(_exec, "test_logger", None) is not None:
                _exec.test_logger.close()
        _write_session_separator(session_log, test_name, "END", elapsed)
        all_indices = [s.gpu_info.index for m in multi_list for s in m.slots]
        _drain_gpu_slots(config, rock_dir, all_indices)
        for multi in multi_list:
            node_pool.release_multi(multi)
            _console_multi_released(node_pool, multi, test_name, elapsed, session_log)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def node_pool(request) -> NodePool | None:
    """Session-scoped ``NodePool`` — single source of truth for GPU slots.

    Returns the pool built by ``pytest_configure``.  ``None`` when ``--no-gpu``
    is active (no hardware session).

    Returns:
        ``NodePool`` or ``None``.
    """
    return getattr(request.config, "_node_pool", None)


@pytest.fixture
def target_executor(request, framework_config, node_pool):
    """Unified GPU test fixture: location- and topology-transparent executor.

    Reads the test's ``hw.*`` and ``e2e.*`` markers to dispatch automatically:

    +------------------------------+--------------------------------------------+
    | Marker combination           | Slots acquired / executor yielded          |
    +==============================+============================================+
    | ``hw.gpu`` (default)         | 1 GPU slot → NodeExecutorGroup(1 executor) |
    +------------------------------+--------------------------------------------+
    | ``hw.multi_gpu``             | N GPU slots (same node, gpu_count marker)  |
    |   + ``gpu_count(N)``         | → NodeExecutorGroup(1 multi-index executor)|
    +------------------------------+--------------------------------------------+
    | ``e2e.multinode``            | N GPUs x each node (gpu_count marker)      |
    |   + ``gpu_count(N)``         | → NodeExecutorGroup(1 executor per node)   |
    +------------------------------+--------------------------------------------+
    | ``--no-gpu``                 | No hardware → NodeExecutorGroup(DryRun)    |
    +------------------------------+--------------------------------------------+
    | ``--container-mode``         | No NodePool → NodeExecutorGroup(Container) |
    +------------------------------+--------------------------------------------+

    The yielded ``NodeExecutorGroup`` has a uniform API across all modes:
        - ``.run(cmd)``           — delegates to the first (or only) executor.
        - ``for e in group``      — iterates all executors (multi-node).
        - ``.count``              — number of executors (1 for single/multi-GPU).

    Markers read:
        ``@pytest.mark.hw.multi_gpu``  — acquire N GPUs from one node.
        ``@pytest.mark.e2e.multinode`` — acquire GPUs from every node in fleet.
        ``@pytest.mark.gpu_count(N)``  — number of GPUs (default 2 for multi-GPU,
                                         1 per node for multi-node).
        ``@pytest.mark.gpu_vram(N)``   — minimum VRAM in GB per GPU.

    Yields:
        ``NodeExecutorGroup`` ready for ``.run()`` calls.

    Example — single GPU::

        @pytest.mark.hw.gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.runtime
        @pytest.mark.runtime.fast
        def test_hip_device_count(target_executor):
            result = target_executor.run("rocm-smi --showid")
            assert result.ok

    Example — multi-GPU same node::

        @pytest.mark.hw.multi_gpu
        @pytest.mark.gpu_count(2)
        @pytest.mark.ci.nightly
        @pytest.mark.layer.math_lib
        @pytest.mark.runtime.medium
        def test_rccl_allreduce(target_executor):
            result = target_executor.run("python3 allreduce.py")
            assert result.ok

    Example — multi-node::

        @pytest.mark.e2e.multinode
        @pytest.mark.gpu_count(1)
        @pytest.mark.ci.nightly
        @pytest.mark.layer.math_lib
        @pytest.mark.runtime.medium
        def test_rccl_multinode(target_executor):
            for exec_ in target_executor:
                exec_.run("torchrun --nproc_per_node=1 allreduce.py")
    """
    import time as _time

    from framework.executors.executor_group import NodeExecutorGroup

    config = request.config

    # --no-gpu: synthetic executor, no hardware needed
    if config.getoption("--no-gpu", default=False):
        from framework.executors.dry_run_executor import DryRunExecutor

        yield NodeExecutorGroup([DryRunExecutor()])
        return

    # --container-mode: container-backed executor, no NodePool slot
    if config.getoption("--container-mode", default=False):
        from framework.executors.container_executor import ContainerExecutor

        image = config.getoption("--container-image", default="rocm/pytorch:latest")
        runtime = config.getoption("--container-runtime", default="docker")
        yield NodeExecutorGroup([ContainerExecutor(image=image, runtime=runtime)])
        return

    # GPU slot acquisition via NodePool
    if node_pool is None:
        pytest.fail(
            "target_executor: NodePool initialisation failed — GPU detection error or driver fault. Check session logs."
        )
        return

    ctx = _resolve_test_context(request, framework_config, config)
    test_name = ctx["test_name"]
    log_path = ctx["log_path"]
    session_log = ctx["session_log"]
    rock_dir = ctx["rock_dir"]
    gpu_count_marker = ctx["gpu_count_marker"]

    # -------------------------------------------------------------------------
    # Multi-node path: acquire GPUs from every node in the fleet
    # -------------------------------------------------------------------------
    if ctx["is_multi_node"]:
        if len(node_pool.node_specs) < 2:
            pytest.skip(
                "target_executor: e2e.multinode requires --remote-node with 2+ nodes "
                f"(currently {len(node_pool.node_specs)} node(s) in pool)"
            )
            return

        gpu_count_per_node = int(gpu_count_marker.args[0]) if gpu_count_marker and gpu_count_marker.args else 1

        try:
            multi_list: list[MultiGpuSlots] = node_pool.acquire_multi_node(
                gpu_count_per_node=gpu_count_per_node,
                vram_required_gb=ctx["vram_required_gb"],
                wait_timeout_secs=ctx["acquire_timeout"],
                test_id=test_name,
            )
        except RuntimeError as exc:
            pytest.skip(f"TEST PLATFORM missing required GPUs: {exc}")
            return

        yield from _acquire_and_yield(
            multi_list=multi_list,
            ctx=ctx,
            framework_config=framework_config,
            config=config,
            node_pool=node_pool,
            is_multi_node=True,
        )
        return

    # -------------------------------------------------------------------------
    # Multi-GPU same-node path: acquire N GPUs from one node
    # -------------------------------------------------------------------------
    if ctx["is_multi_gpu"]:
        count = int(gpu_count_marker.args[0]) if gpu_count_marker and gpu_count_marker.args else 2
        multi = _acquire_multi_gpu(node_pool, ctx, count)
        yield from _acquire_and_yield(
            multi_list=[multi],
            ctx=ctx,
            framework_config=framework_config,
            config=config,
            node_pool=node_pool,
            is_multi_node=False,
        )
        return

    # -------------------------------------------------------------------------
    # Single-GPU path (hw.gpu — default)
    # -------------------------------------------------------------------------
    slot = _acquire_single(node_pool, ctx)
    _console_slot_acquired(node_pool, slot, test_name, session_log)
    _write_session_separator(session_log, test_name, "START")

    # Wrap slot as a single-item MultiGpuSlots for unified monitoring API
    from framework.nodes.node_pool import MultiGpuSlots as _MultiGpuSlots  # pylint: disable=import-outside-toplevel

    slot_as_multi = _MultiGpuSlots(slots=[slot], node_spec=slot.node_spec)
    health_monitors, pre_health_maps, bg_monitors = _setup_monitoring(
        ctx, framework_config, [slot_as_multi], is_multi_node_shape=False
    )
    _single_exec = slot.make_executor(test_id=test_name, log_path=log_path, session_log_path=session_log)
    _t = _time.monotonic()
    try:
        yield NodeExecutorGroup([_single_exec])
    finally:
        elapsed = _time.monotonic() - _t
        _teardown_monitoring(ctx, [slot_as_multi], health_monitors, pre_health_maps, bg_monitors, elapsed)
        if getattr(_single_exec, "test_logger", None) is not None:
            _single_exec.test_logger.close()
        _write_session_separator(session_log, test_name, "END", elapsed)
        _drain_gpu_slots(config, rock_dir, [slot.gpu_info.index])
        node_pool.release([slot])
        _console_slot_released(node_pool, slot, test_name, elapsed, session_log)


@pytest.fixture
def multi_gpu_fixture(request, framework_config, node_pool):
    """N GPUs from ONE node — yields a ready executor (explicit alternative to target_executor).

    Prefer ``target_executor`` with ``@pytest.mark.hw.multi_gpu`` for new tests.
    Use ``multi_gpu_fixture`` when the test must declare multi-GPU acquisition
    explicitly (e.g., parametrized tests that switch between single and multi-GPU).

    Acquires ``@pytest.mark.gpu_count(N)`` GPUs (default 2) from a single node
    and yields a ``NodeExecutorGroup`` wrapping one executor with all GPU
    indices in ``ROCR_VISIBLE_DEVICES``.  Releases all slots on teardown.

    Markers read:
        ``@pytest.mark.gpu_count(N)`` — GPUs to allocate (default 2).

    Skips:
        When ``--no-gpu`` is active or fewer than N GPUs are available.

    Yields:
        ``NodeExecutorGroup`` — call ``.run(cmd)`` directly; no ``.make_executor()`` needed.

    Example::

        @pytest.mark.hw.multi_gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.math_lib
        @pytest.mark.e2e.stack
        @pytest.mark.runtime.medium
        @pytest.mark.gpu_count(4)
        def test_rccl_allreduce(multi_gpu_fixture):
            result = multi_gpu_fixture.run("python3 rccl_allreduce.py")
            assert result.ok
    """

    config = request.config

    if config.getoption("--no-gpu", default=False):
        pytest.skip("multi_gpu_fixture: skipped — --no-gpu is active")
        return

    if node_pool is None:
        pytest.fail(
            "multi_gpu_fixture: NodePool initialisation failed — GPU detection error"
            " or driver fault. Check session logs."
        )
        return

    ctx = _resolve_test_context(request, framework_config, config)
    gpu_count_marker = ctx["gpu_count_marker"]
    count = int(gpu_count_marker.args[0]) if gpu_count_marker and gpu_count_marker.args else 2

    multi = _acquire_multi_gpu(node_pool, ctx, count)
    yield from _acquire_and_yield(
        multi_list=[multi],
        ctx=ctx,
        framework_config=framework_config,
        config=config,
        node_pool=node_pool,
        is_multi_node=False,
    )


@pytest.fixture
def multi_node_fixture(request, framework_config, node_pool):
    """GPU slots from EACH node — yields a ready executor group (explicit alternative).

    Prefer ``target_executor`` with ``@pytest.mark.e2e.multinode`` for new tests.
    Use ``multi_node_fixture`` when the test needs to be explicit about multi-node
    acquisition.

    Acquires ``@pytest.mark.gpu_count(N)`` GPUs from every node in the fleet
    simultaneously and yields a ``NodeExecutorGroup`` (one executor per node).
    Releases all slots on teardown.

    Markers read:
        ``@pytest.mark.gpu_count(N)`` — GPUs per node (default 1).

    Skips:
        When ``--no-gpu`` is active or ``--remote-node`` is not specified
        (multi-node requires at least two nodes).

    Yields:
        ``NodeExecutorGroup`` — iterate with ``for exec_ in multi_node_fixture``
        to dispatch commands per node.  Call ``.count`` for the node count.

    Example::

        @pytest.mark.hw.multi_gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.math_lib
        @pytest.mark.e2e.multinode
        @pytest.mark.runtime.medium
        @pytest.mark.gpu_count(2)
        def test_rccl_multinode(multi_node_fixture):
            for exec_ in multi_node_fixture:
                result = exec_.run("torchrun --nproc_per_node=2 allreduce.py")
                assert result.ok
    """

    config = request.config

    if config.getoption("--no-gpu", default=False):
        pytest.skip("multi_node_fixture: skipped — --no-gpu is active")
        return

    if node_pool is None:
        pytest.fail(
            "multi_node_fixture: NodePool initialisation failed — GPU detection error"
            " or driver fault. Check session logs."
        )
        return

    if len(node_pool.node_specs) < 2:
        pytest.skip("multi_node_fixture: requires --remote-node with 2+ nodes " "(currently only one node in the pool)")
        return

    ctx = _resolve_test_context(request, framework_config, config)
    gpu_count_marker = ctx["gpu_count_marker"]
    gpu_count_per_node = int(gpu_count_marker.args[0]) if gpu_count_marker and gpu_count_marker.args else 1

    try:
        multi_list: list[MultiGpuSlots] = node_pool.acquire_multi_node(
            gpu_count_per_node=gpu_count_per_node,
            wait_timeout_secs=ctx["acquire_timeout"],
            test_id=ctx["test_name"],
        )
    except RuntimeError as exc:
        pytest.skip(f"multi_node_fixture: {exc}")
        return

    yield from _acquire_and_yield(
        multi_list=multi_list,
        ctx=ctx,
        framework_config=framework_config,
        config=config,
        node_pool=node_pool,
        is_multi_node=True,
    )
