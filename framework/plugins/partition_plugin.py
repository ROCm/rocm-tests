# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
partition_plugin.py -- partition markers, and an opt-in compute-mode switch.

Skip hook
---------
Compute markers (``partition.spx``, ``dpx``, ``tpx``, ``qpx``, ``cpx``) are compared
with ``current_compute_partition``. Memory markers (``partition.nps1``, ``nps2``,
``nps4``) are compared with ``current_memory_partition``. A test may carry one of
each. It runs only when every GPU on the assigned node reports the marked mode.
Other tests are unchanged.

A test that uses ``target_executor`` is checked on the node that fixture acquired.
The sysfs read goes through that executor, so ``--remote-node`` reads the GPU host
rather than the controller running pytest. A partition-marked test with no
``target_executor`` is checked on the controller.

The ``/sys`` probe runs only on Linux. On any other platform, including Windows,
the probe is not attempted and the test is not skipped. On Linux, a missing or
unreadable partition file skips the marked test: the partition state is
unavailable, so the marker cannot be enforced.

Compute switch
--------------
The skip hook does not change the partition. Suites that need more than one
compute mode in one pytest session request the ``compute_partition`` fixture and
call ``set("dpx")`` (or ``spx``, ``tpx``, ``qpx``, ``cpx``). The fixture records
the mode it found, and writes that mode back when the module ends, including
after a failure. Memory partition is not changed.

Call ``set`` from a class-scoped autouse fixture so it runs before this plugin's
function-scoped skip hook. Keep switching tests in their own module. Compute
mode is per node, not per GPU: ``NodePool.hold_node`` acquires every GPU on the
node, keeps those slots until the original mode is restored, and
``release_hold`` runs after restore. Tests in the module reuse the hold through
``target_executor`` with an unmasked executor (no ``ROCR_VISIBLE_DEVICES``).
The session-start GPU list is not rediscovered after ``set``: SPX card indices
stay locked, and the unmasked executor lets the test see CPX slices the driver
adds. The scheduler puts every test in that module in one ``xdist_group``.
On a one-node pool every ``compute_partition`` module shares that group, so a
second switching file cannot start ``hold_node`` until the first restores.
With two or more nodes, each module keeps its own group. ``--dist=load``
(worksteal) is rejected so two workers cannot each try to hold the node.
An empty ``NodePool`` SSH map is not the GPU host —
``NodePool.ssh_for`` opens SSH from the ``NodeSpec``.

``set`` needs root or passwordless sudo on the GPU host. It runs the
``amd-smi`` under ``--rock-dir`` when that path is set, otherwise
``${ROCM_PATH:-/opt/rocm}/bin/amd-smi``. It treats sysfs as the result, not
the ``amd-smi`` exit code. A DPX-to-CPX change can exit 2 after the cards have
already switched.

Example::

    class TestSomethingCpx:
        pytestmark = [pytest.mark.partition.cpx, pytest.mark.gpu_count("all")]

        @pytest.fixture(scope="class", autouse=True)
        def _use_cpx(self, compute_partition):
            compute_partition.set("cpx")

        def test_one_case(self, target_executor):
            ...
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import enum
import logging
import pathlib
import shlex
import time

import pytest

logger = logging.getLogger(__name__)

_COMPUTE_MODES = ("spx", "dpx", "tpx", "qpx", "cpx")
_MEMORY_MODES = ("nps1", "nps2", "nps4")
_COMPUTE_FILE = "current_compute_partition"
_MEMORY_FILE = "current_memory_partition"
_PROBE_TIMEOUT_SECS = 30.0
_SWITCH_TIMEOUT_SECS = 180.0
_SWITCH_AFTER_ERROR_SECS = 30.0
_POLL_SECS = 2.0


class _SysfsNotApplicable(enum.Enum):
    """Sentinel: this node has no Linux sysfs partition files.

    An empty read there is not a failed probe, so the marked test is allowed to run.
    This is an enum rather than ``object()``. ``set[str] | object`` collapses to
    ``object``, and mypy then rejects ``len``, ``sorted``, and ``iter`` on the result.
    """

    VALUE = "not_applicable"


_SYSFS_NOT_APPLICABLE = _SysfsNotApplicable.VALUE


# ---------------------------------------------------------------------------
# Sysfs reads used by the skip hook and by compute_partition.set
# ---------------------------------------------------------------------------


def _modes_from_text(text: str) -> set[str]:
    """Return distinct non-empty lines of partition text, lower-cased."""
    return {line.strip().lower() for line in text.splitlines() if line.strip()}


def _local_has_sysfs() -> bool:
    """True when this pytest process is Linux and ``/sys`` can be read."""
    from framework.os_adapter import os_adapter_factory  # pylint: disable=import-outside-toplevel

    return os_adapter_factory().get_platform_name() in ("linux", "wsl")


def _executor_has_sysfs(executor: object) -> bool | None:
    """True when ``executor``'s node is Linux.

    ``None`` means ``uname`` failed and the caller must not cache the result.
    ``False`` means another platform, such as Windows, so ``/sys`` is not read.
    """
    host = str(getattr(executor, "host", None) or type(executor).__name__)
    cache: dict[str, bool] = getattr(_executor_has_sysfs, "cache", {})
    if host in cache:
        return cache[host]
    result = executor.run("uname -s", timeout=_PROBE_TIMEOUT_SECS)  # type: ignore[attr-defined]
    if not result.ok:
        logger.info("partition probe could not run uname (exit %s); partition state unavailable", result.exit_code)
        return None
    cache[host] = result.stdout.strip().lower().startswith("linux")
    _executor_has_sysfs.cache = cache  # type: ignore[attr-defined]
    return cache[host]


def _read_local(filename: str) -> set[str] | _SysfsNotApplicable:
    """Return distinct values of one sysfs partition file on this machine."""
    if not _local_has_sysfs():
        logger.info("partition sysfs check does not apply on this platform; not skipping")
        return _SYSFS_NOT_APPLICABLE
    modes: set[str] = set()
    for path in pathlib.Path("/sys/bus/pci/devices").glob(f"*/{filename}"):
        try:
            modes.update(_modes_from_text(path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return modes


def _read_executor(executor: object, filename: str) -> set[str] | _SysfsNotApplicable | None:
    """Read ``filename`` through ``executor``.

    Returns ``None`` when the command failed and the result must not be cached.
    ``_SYSFS_NOT_APPLICABLE`` means the node is not Linux, so the caller does not
    skip. An empty set on Linux means the partition state is unavailable.
    """
    from framework.executors.dry_run_executor import DryRunExecutor  # pylint: disable=import-outside-toplevel

    if isinstance(executor, DryRunExecutor):
        return _SYSFS_NOT_APPLICABLE
    has_sysfs = _executor_has_sysfs(executor)
    if has_sysfs is None:
        return None
    if not has_sysfs:
        logger.info("partition sysfs check does not apply on the assigned node; not skipping")
        return _SYSFS_NOT_APPLICABLE
    command = f'for f in /sys/bus/pci/devices/*/{filename}; do [ -r "$f" ] && cat "$f"; done'
    result = executor.run(command, timeout=_PROBE_TIMEOUT_SECS)  # type: ignore[attr-defined]
    modes = _modes_from_text(result.stdout)
    if modes or result.ok or not result.stderr.strip():
        return modes
    logger.info("partition probe of %s failed (exit %s); partition state unavailable", filename, result.exit_code)
    return None


def _cached_modes(
    config: pytest.Config,
    key: tuple[str, str],
    read: Callable[[], set[str] | _SysfsNotApplicable | None],
) -> set[str] | _SysfsNotApplicable:
    """Return partition modes for ``key``, reading them once per session."""
    cache: dict[tuple[str, str], set[str] | _SysfsNotApplicable]
    stored = getattr(config, "_partition_mode_cache", None)
    if stored is None:
        cache = {}
        config._partition_mode_cache = cache  # type: ignore[attr-defined]
    else:
        cache = stored
    if key not in cache:
        modes = read()
        if modes is None:
            return set()
        cache[key] = modes
    found = cache[key]
    if found is _SYSFS_NOT_APPLICABLE:
        return found
    return set(found)


def _clear_partition_cache(config: pytest.Config) -> None:
    """Drop the session cache so the next skip check re-reads sysfs.

    After ``compute_partition.set``, a cached SPX value would skip every DPX test.
    """
    cache = getattr(config, "_partition_mode_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _partition_modes(request: pytest.FixtureRequest, filename: str) -> set[str] | _SysfsNotApplicable:
    """Read ``filename`` on the assigned node, or on the controller."""
    if "target_executor" not in request.fixturenames:
        return _cached_modes(request.config, ("controller", filename), lambda: _read_local(filename))

    group = request.getfixturevalue("target_executor")
    modes: set[str] = set()
    saw_node = False
    for executor in group:
        host = getattr(executor, "host", None) or type(executor).__name__

        def _read_this(bound=executor) -> set[str] | _SysfsNotApplicable | None:
            return _read_executor(bound, filename)

        found = _cached_modes(request.config, (str(host), filename), _read_this)
        if found is _SYSFS_NOT_APPLICABLE:
            continue
        saw_node = True
        modes.update(found)
    if not saw_node:
        return _SYSFS_NOT_APPLICABLE
    return modes


# ---------------------------------------------------------------------------
# Skip tests whose partition.* marker does not match sysfs
# ---------------------------------------------------------------------------


def _marked_modes(item: pytest.Item, allowed: tuple[str, ...]) -> list[str]:
    """Return partition marker values on ``item`` that are in ``allowed``."""
    found: list[str] = []
    for marker in item.iter_markers():
        name = marker.name
        if not name.startswith("partition."):
            continue
        mode = name.split(".", 1)[1]
        if mode in allowed:
            found.append(mode)
    return found


def _skip_unless(
    item: pytest.Item,
    kind: str,
    filename: str,
    wanted: list[str],
    set_flag: str,
    modes: set[str] | _SysfsNotApplicable,
) -> None:
    """Skip ``item`` when ``modes`` is not exactly one of ``wanted``."""
    marked = ",".join(wanted)
    if modes is _SYSFS_NOT_APPLICABLE:
        logger.info("partition.%s on %s: sysfs check does not apply; not skipping", marked, item.nodeid)
        return
    if not modes:
        pytest.skip(
            f"partition.{marked} test skipped: {kind} partition state unavailable "
            f"({filename} missing or unreadable on the assigned node)."
        )
    if len(modes) > 1:
        pytest.skip(
            f"partition test skipped: GPUs report mixed {kind} partitions {sorted(modes)}. "
            "Set one mode on every GPU before running."
        )
    node_mode = next(iter(modes))
    if node_mode not in wanted:
        pytest.skip(
            f"partition.{marked} test skipped: node {kind} partition is {node_mode.upper()}. "
            f"Set it with `sudo amd-smi set {set_flag} {marked.upper()}` to run this test."
        )


@pytest.fixture(autouse=True)
def _partition_on_execution_node(request: pytest.FixtureRequest) -> None:
    """Skip a test whose partition marker does not match the assigned node.

    Tests without a ``partition.*`` marker return immediately and do not acquire
    a GPU. When the test already requests ``target_executor``, the sysfs files
    are read through that executor after the node is acquired.
    """
    item = request.node
    compute = _marked_modes(item, _COMPUTE_MODES)
    memory = _marked_modes(item, _MEMORY_MODES)
    if compute:
        _skip_unless(item, "compute", _COMPUTE_FILE, compute, "-C", _partition_modes(request, _COMPUTE_FILE))
    if memory:
        _skip_unless(item, "memory", _MEMORY_FILE, memory, "-M", _partition_modes(request, _MEMORY_FILE))


# ---------------------------------------------------------------------------
# Opt-in compute partition switch: request compute_partition and call set()
# ---------------------------------------------------------------------------


def _target_label(executor: object) -> str:
    """Stable name for logs and for the per-node restore map."""
    return str(getattr(executor, "host", None) or getattr(executor, "label", None) or type(executor).__name__)


def _reject_worksteal_xdist(config: pytest.Config) -> None:
    """Fail when xdist worksteal would put this module on two workers."""
    dist = getattr(config.option, "dist", None)
    if dist in (None, "no", False, "loadscope", "loadfile", "loadgroup"):
        return
    if str(dist) not in ("load", "each"):
        return
    pytest.fail(
        "compute_partition cannot run under pytest-xdist --dist=load (or each): "
        "two workers would each try to hold every GPU on the node. "
        "Use --dist=loadgroup (the scheduler pins this module to one worker) or run without -n."
    )


def _amd_smi_bin(rock_dir: str | None) -> str:
    """Return a shell token for ``amd-smi`` on the GPU host.

    ``--rock-dir`` / TheRock installs are not on ``ROCM_PATH``. Over SSH that
    variable is usually unset, so a bare ``${ROCM_PATH:-/opt/rocm}`` fallback
    would pick the wrong binary.
    """
    if rock_dir:
        return shlex.quote(str(pathlib.Path(rock_dir) / "bin" / "amd-smi"))
    return '"${ROCM_PATH:-/opt/rocm}/bin/amd-smi"'


def _run_on_target(executor: object, command: str, timeout: float):
    """Run ``command`` on ``executor``.

    ``amd-smi set -C`` is node-wide. Drop the executor's GPU mask and test logger
    for this call so the tool sees every device on that host, then put them back.
    """
    saved_logger = getattr(executor, "test_logger", None)
    saved_indices = getattr(executor, "gpu_indices", None)
    if hasattr(executor, "test_logger"):
        executor.test_logger = None
    if saved_indices:
        executor.gpu_indices = []  # type: ignore[attr-defined]
    try:
        return executor.run(command, timeout=timeout)  # type: ignore[attr-defined]
    finally:
        if hasattr(executor, "test_logger"):
            executor.test_logger = saved_logger
        if saved_indices is not None:
            executor.gpu_indices = saved_indices  # type: ignore[attr-defined]


def _read_compute_modes_on(executor: object) -> set[str]:
    """Live sysfs compute modes. Not cached: a switch must see the new value."""
    found = _read_executor(executor, _COMPUTE_FILE)
    if found is None or found is _SYSFS_NOT_APPLICABLE:
        return set()
    return found


def _apply_compute_mode_on(executor: object, mode: str, rock_dir: str | None = None) -> None:
    """Set every GPU on ``executor`` to ``mode`` and wait until sysfs agrees."""
    if mode not in _COMPUTE_MODES:
        pytest.fail(f"refusing to set unknown compute partition {mode!r}")
    label = _target_label(executor)
    current = _read_compute_modes_on(executor)
    if current == {mode}:
        logger.info("compute partition on %s is already %s; not calling amd-smi", label, mode)
        return
    amd_smi = _amd_smi_bin(rock_dir)
    command = (
        f'if [ "$(id -u)" -eq 0 ]; then {amd_smi} set -C {mode.upper()}; '
        f"else sudo -n {amd_smi} set -C {mode.upper()}; fi"
    )
    result = _run_on_target(executor, command, timeout=_SWITCH_TIMEOUT_SECS)
    amd_smi_text = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
    if not result.ok and "sudo:" in amd_smi_text:
        pytest.fail(
            f"could not set compute partition to {mode.upper()} on {label}: amd-smi set needs "
            f"root or passwordless sudo.\n{amd_smi_text[-1500:]}"
        )
    # amd-smi exits non-zero when it cannot repartition a non-primary slice,
    # for example DPX to CPX reports NOT_SUPPORTED on each card's second half
    # although the card itself switched. sysfs is the verdict, not the exit code.
    wait_secs = _SWITCH_TIMEOUT_SECS if result.ok else _SWITCH_AFTER_ERROR_SECS
    deadline = time.monotonic() + wait_secs
    last: set[str] = set()
    while True:
        last = _read_compute_modes_on(executor)
        if last == {mode}:
            logger.info("compute partition on %s is %s (amd-smi exit %s)", label, mode, result.exit_code)
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_SECS)
    pytest.fail(
        f"compute partition on {label} did not become {mode.upper()} (amd-smi exit {result.exit_code}); "
        f"sysfs reported {sorted(last) or 'nothing'}.\n{amd_smi_text[-1500:]}"
    )


class ComputePartition:
    """Set the compute partition, then put the original mode back.

    One instance is created per test module that requests ``compute_partition``.
    :meth:`set` may be called more than once (SPX, then DPX, then CPX). Teardown
    restores the mode recorded at fixture setup, not the last mode ``set`` used.
    """

    def __init__(
        self,
        config: pytest.Config,
        targets: list[object],
        enabled: bool,
        saved: dict[str, str],
        rock_dir: str | None = None,
    ):
        self._config = config
        self._targets = targets
        self._enabled = enabled
        self._saved = saved
        self._rock_dir = rock_dir

    def set(self, mode: str) -> None:
        """Switch the held GPU node to ``mode``.

        ``mode`` is one of ``spx``, ``dpx``, ``tpx``, ``qpx``, ``cpx``. Unknown
        names fail. ``--no-gpu`` and a node with no sysfs file are no-ops.
        """
        if not self._enabled:
            logger.info("compute_partition.set(%s) skipped: switch is not enabled on this node", mode)
            return
        mode = mode.lower()
        logger.info("compute_partition: switching to %s", mode)
        for executor in self._targets:
            _apply_compute_mode_on(executor, mode, rock_dir=self._rock_dir)
        _clear_partition_cache(self._config)

    def _restore(self) -> None:
        """Write each node's recorded mode back. Always clear the skip-hook cache."""
        if not self._enabled or not self._saved:
            return
        try:
            for executor in self._targets:
                saved = self._saved.get(_target_label(executor))
                if saved is None:
                    continue
                logger.info("compute_partition: restoring %s on %s", saved, _target_label(executor))
                _apply_compute_mode_on(executor, saved, rock_dir=self._rock_dir)
        finally:
            _clear_partition_cache(self._config)


def _snapshot_compute_modes(targets: list[object]) -> dict[str, str]:
    """Record one compute mode per node. Fail when a node already has mixed modes."""
    saved: dict[str, str] = {}
    for executor in targets:
        label = _target_label(executor)
        modes = _read_compute_modes_on(executor)
        if not modes:
            logger.info("no compute-partition sysfs on %s; not switching", label)
            continue
        if len(modes) != 1:
            pytest.fail(
                f"GPUs on {label} report mixed compute partitions {sorted(modes)}. "
                "Set one mode on every GPU before running this suite."
            )
        saved[label] = next(iter(modes))
        logger.info("compute_partition: saving %s on %s for restore", saved[label], label)
    return saved


@pytest.fixture(scope="module")
def compute_partition(request: pytest.FixtureRequest, node_pool) -> Iterator[ComputePartition]:
    """Switch compute partition for this module, then restore the original mode.

    Opt-in: tests that only skip on a mismatch do not request this fixture.
    Setup calls ``NodePool.hold_node``, records the current mode, and leaves
    the slots held. Teardown restores that mode, then ``release_hold``, before
    pytest enters another folder. Tests reuse the hold through an unmasked
    ``target_executor`` so ``ROCR_VISIBLE_DEVICES`` does not hide CPX slices.
    """
    config = request.config
    if config.getoption("--no-gpu", default=False):
        yield ComputePartition(config, [], enabled=False, saved={})
        return
    if node_pool is None:
        pytest.fail(
            "compute_partition: NodePool is not available. The switch must run on the GPU "
            "host that will execute the test; a missing pool is not the local GPU host."
        )
    _reject_worksteal_xdist(config)
    handle: ComputePartition | None = None
    timeout = float(config.getoption("--gpu-acquire-timeout", default=180.0))
    module = getattr(getattr(request, "module", None), "__name__", "unknown")
    rock_dir = getattr(node_pool, "rock_dir", None) or config.getoption("--rock-dir", default=None)
    try:
        try:
            multi = node_pool.hold_node(test_id=f"compute_partition:{module}", wait_timeout_secs=timeout)
        except RuntimeError as exc:
            pytest.fail(f"compute_partition: could not acquire every GPU on the node before amd-smi set -C: {exc}")
        try:
            executor = node_pool.switch_executor(multi)
        except RuntimeError as exc:
            pytest.fail(str(exc))
        saved = _snapshot_compute_modes([executor])
        handle = ComputePartition(config, [executor], enabled=bool(saved), saved=saved, rock_dir=rock_dir)
        yield handle
    finally:
        try:
            if handle is not None:
                handle._restore()
        finally:
            node_pool.release_hold()
