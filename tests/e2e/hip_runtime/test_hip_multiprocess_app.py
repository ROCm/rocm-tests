# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hip_multiprocess_app.py -- HIP multi-process CLR integration stress tests.

**Test intent**

Validates that the Compute Language Runtime (CLR) remains stable and correct when
multiple independent processes compete concurrently for the same GPU(s).  Each
process exercises a distinct CLR path simultaneously, creating realistic resource
contention that single-process tests cannot expose:

    profiler      — event-based kernel timing; measures event-latency anomalies
                    under contention relative to an uncontended baseline
    compute       — continuous HIP kernel launch loop (HIP → ROCr → KFD → HW)
    memory_mover  — alloc / free / copy + cross-process IPC shared memory
                    (producer + consumer pair; holds the VRAM resident set when
                    VRAM pressure is enabled)
    library       — hipBLASLt GEMM sweep (enabled when hipBLASLt is present)
    compiler      — hipRTC compile / load / unload (enabled when hipRTC is present)
    monitor       — amd-smi GPU metrics queries (enabled when amd-smi is present)
    ipc_xfer      — cross-GPU peer-to-peer copy (multi-GPU scenarios only)

**Workflow**

Each test calls ``_run_suite()``, which orchestrates a two-phase workload:

- **Phase 1 — uncontended profiler baseline:** the profiler role is launched alone
  per GPU and the test waits for it to write the ``profiler_baseline_done`` signal
  file before proceeding. This gives anomaly detection a clean reference point.
- **Phase 2 — concurrent contention:** all remaining roles are launched
  simultaneously per GPU. The workload runs for a fixed ``--duration`` (seconds).
  Roles redirect their own stdout/stderr to node-local files
  (``<results_dir>/<label>_stdout.log`` / ``_stderr.log``) alongside the health,
  monitor, and profiler CSV outputs they emit.  A periodic live tail of each role's
  stdout log is written to the console every ``ROCM_TEST_ROCK_MPS_PROGRESS_SECS``
  seconds (default 30 s; set to ``0`` to silence) so long runs remain observable.
- **Pass / fail gate:** after the workload completes, ``check_results.py`` (vendored
  at ``tests/e2e/hip_runtime/src/mps/scripts/``) is staged onto the execution
  node and run against each GPU's results directory.  The test asserts a clean
  verdict from this analyzer; any anomaly or threshold violation reported by the
  analyzer is surfaced as a test failure with the full report attached.

VRAM pressure is expressed as a *percentage of total VRAM* (0-90 %), applied only
to Phase-2 contention roles — never to the profiler baseline — making the workload
architecture-agnostic (no fixed VRAM floor is declared via ``gpu_vram``).

**Coverage**

This file defines 7 marker-tagged test functions (18 parametrised cases) spanning:

- Single-GPU full-suite nightly + soak scenarios (``test_hip_multiprocess_app``,
  ``test_hip_multiprocess_app_soak``)
- Multi-GPU full-suite with cross-GPU peer copy (``test_hip_multiprocess_all_gpus``,
  ``test_hip_multiprocess_all_gpus_soak``)
- Targeted two-role interaction pairs for focused regression detection
  (``test_hip_multiprocess_role_pair``)
- PR-gate per-role smoke runs, 60 s each (``test_hip_multiprocess_single_role``)
- Cross-process IPC shared-memory producer + consumer isolation
  (``test_hip_multiprocess_ipc``)

``pytest -m`` selection alone picks the scenario and its parameters.

Binary compiled via CMake from ``tests/e2e/hip_runtime/src/mps``.
Analyzer vendored at ``tests/e2e/hip_runtime/src/mps/scripts/check_results.py``.
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import pathlib
import re
import time

import pytest

from framework.common.helpers import ExecutionResult
from framework.executors.background_process import AbstractBackgroundProcess

logger = logging.getLogger("rocm.test")

# Vendored analyzer that reproduces the original scripts' pass/fail gate.
_CHECKER_SRC = pathlib.Path(__file__).parent / "src" / "mps" / "scripts" / "check_results.py"
# Encoded once at import time so multi-GPU runs don't re-read and re-encode the same file N times.
_CHECKER_B64: str = base64.b64encode(_CHECKER_SRC.read_bytes()).decode()

_ENV_DURATION = "ROCM_TEST_ROCK_MPS_DURATION"
_ENV_VRAM_PRESSURE = "ROCM_TEST_ROCK_MPS_VRAM_PRESSURE"
# Console progress cadence (seconds) during the Phase-2 wait; 0 disables it.
_ENV_PROGRESS_SECS = "ROCM_TEST_ROCK_MPS_PROGRESS_SECS"

# Optional profiler/health threshold overrides -> binary flags (README "Tuning").
_THRESHOLD_ENV_TO_FLAG = {
    "ROCM_TEST_ROCK_MPS_ANOMALY_RATIO": "--anomaly-ratio",
    "ROCM_TEST_ROCK_MPS_SEVERE_RATIO": "--severe-ratio",
    "ROCM_TEST_ROCK_MPS_ANOMALY_PCT": "--anomaly-pct",
    "ROCM_TEST_ROCK_MPS_SEVERE_PCT": "--severe-pct",
    "ROCM_TEST_ROCK_MPS_RSS_WARN": "--rss-warn",
    "ROCM_TEST_ROCK_MPS_FD_WARN": "--fd-warn",
}
_CONTENTION_PROFILER_DEFAULTS = {
    "ROCM_TEST_ROCK_MPS_ANOMALY_PCT": "5.0",
    "ROCM_TEST_ROCK_MPS_SEVERE_PCT": "0.1",
}
_MEMORY_MOVER_RSS_WARN_MB = "1024"

# Phase-2 contention roles for the full suite (profiler is launched in Phase 1).
# memory_mover runs producer + consumer (cross-process IPC shared memory);
# compiler runs twice (concurrent compile/load/unload races).
_FULL_SUITE_ROLES: list[tuple[str, str, list[str]]] = [
    ("compute", "compute", []),
    ("memory_mover_producer", "memory_mover", ["--ipc-role", "producer"]),
    ("memory_mover_consumer", "memory_mover", ["--ipc-role", "consumer"]),
    ("library", "library", []),
    ("compiler_1", "compiler", []),
    ("compiler_2", "compiler", []),
    ("monitor", "monitor", []),
]

# Targeted role pairs (run_subset.sh) — (role_a, role_b, vram_pressure).
_ROLE_PAIRS: list[tuple[str, str, int]] = [
    ("compute", "compiler", 0),
    ("memory_mover", "library", 60),
    ("compute", "monitor", 0),
    ("compute", "profiler", 0),
]

# Single-role baselines — (role, vram_pressure).
_SINGLE_ROLES: list[tuple[str, int]] = [
    ("compute", 0),
    ("memory_mover", 60),
    ("library", 0),
    ("compiler", 0),
    ("monitor", 0),
    ("profiler", 0),
]


def _threshold_args(defaults: dict[str, str] | None = None) -> list[str]:
    """Return binary threshold flags for any tuning env vars that are set."""
    args: list[str] = []
    for env_name, flag in _THRESHOLD_ENV_TO_FLAG.items():
        value = os.environ.get(env_name) or (defaults or {}).get(env_name)
        if value:
            args += [flag, value]
    return args


def _role_health_args(role: str) -> list[str]:
    """Return role-specific health thresholds that can still be overridden by env vars."""
    if role == "memory_mover" and not os.environ.get("ROCM_TEST_ROCK_MPS_RSS_WARN"):
        return ["--rss-warn", _MEMORY_MOVER_RSS_WARN_MB]
    return []


def _base_dir(run_ctx, request) -> str:
    """Node-local scratch dir, unique per test invocation (param id included)."""
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    return f"/tmp/rock_mps_test_{run_ctx.run_id}_{tag}"  # nosec B108 - node-local scratch


def _role_command(
    *, ld: str, binary: str, gpu_index: int, duration: int, results_dir: str, label: str, role: str, extra: list[str]
) -> str:
    """Build a node-side command that redirects a role's output beside its CSVs."""
    return (
        f"env LD_LIBRARY_PATH={ld} {binary}"
        f" --role {role} --gpu {gpu_index} --duration {duration}"
        f" --results {results_dir} --verbose{' ' + ' '.join(extra) if extra else ''}"
        f" > {results_dir}/{label}_stdout.log 2> {results_dir}/{label}_stderr.log"
    )


def _await_baseline(target_executor, results_dir: str, proc, duration: int) -> None:
    """Wait for the profiler's uncontended baseline signal (or its early exit)."""
    signal = f"{results_dir}/profiler_baseline_done"
    deadline = time.monotonic() + min(duration, 45)
    while time.monotonic() < deadline:
        if target_executor.run(f"test -f {signal}").ok:
            return
        if not proc.is_alive:  # profiler exited before signalling
            return
        time.sleep(1.0)


def _await_completion(
    target_executor,
    procs: dict[tuple[int, str], AbstractBackgroundProcess],
    results_dirs: dict[int, str],
    duration: int,
) -> None:
    """Wait for the fixed-duration workload to finish, emitting a periodic live tail.

    Roles redirect their own stdout/stderr to node-side files, so nothing streams
    back automatically during the (potentially very long) contention phase.  To
    keep the console informative, every ``ROCM_TEST_ROCK_MPS_PROGRESS_SECS``
    (default 30 s) this logs the elapsed time, how many roles are still running,
    and the last line of each role's stdout log.  Set the env var to ``0`` to
    silence it.
    """
    deadline = time.monotonic() + duration + 180
    start = time.monotonic()
    try:
        report_every = float(os.environ.get(_ENV_PROGRESS_SECS, "30"))
    except ValueError:
        report_every = 30.0
    # procs keys are (gpu_idx, role_label) tuples; sort gives stable GPU-then-alpha ordering.
    logs = [f"{results_dirs[idx]}/{label}_stdout.log" for (idx, label) in sorted(procs)]
    last_report = 0.0
    while time.monotonic() < deadline:
        elapsed = time.monotonic() - start
        alive = [key for key, proc in procs.items() if proc.is_alive]
        if not alive:
            return
        if report_every > 0 and elapsed - last_report >= report_every:  # pylint: disable=chained-comparison
            logger.info(
                "rock_mps progress: elapsed %.0fs/%ds — %d/%d roles running; last log lines:",
                elapsed,
                duration,
                len(alive),
                len(procs),
            )
            if logs:
                with contextlib.suppress(Exception):
                    target_executor.run(f"tail -n 1 -v {' '.join(logs)} 2>/dev/null || true")
            last_report = elapsed
        time.sleep(2.0)


def _stage_and_check(target_executor, results_dir: str):
    """Stage the vendored analyzer onto the node and run it against *results_dir*."""
    remote_checker = f"{results_dir}/check_results.py"
    staged = target_executor.run(f"echo {_CHECKER_B64} | base64 -d > {remote_checker}")
    assert staged.ok, f"failed to stage check_results.py in {results_dir}: {staged.stderr[:400]}"
    return target_executor.run(f"python3 {remote_checker} {results_dir} --no-color --junit {results_dir}/results.xml")


def _run_suite(  # noqa: C901  # pylint: disable=too-many-locals,too-many-branches
    *,
    target_executor,
    ld: str,
    binary: str,
    base_dir: str,
    gpu_indices: list[int],
    duration: int,
    vram_pressure: int,
    phase2_roles: list[tuple[str, str, list[str]]],
    run_profiler: bool,
    enable_ipc: bool = False,
) -> None:
    """Orchestrate the two-phase concurrent workload and gate on check_results.py.

    Launches an uncontended profiler baseline (when *run_profiler*), then the
    Phase-2 contention roles per GPU (adding ipc_xfer on multi-GPU when
    *enable_ipc*), waits for completion, and asserts the analyzer's verdict for
    every GPU's results directory.
    """
    duration = int(os.environ.get(_ENV_DURATION, duration))
    vram_pressure = int(os.environ.get(_ENV_VRAM_PRESSURE, vram_pressure))
    assert 0 <= vram_pressure <= 90, f"{_ENV_VRAM_PRESSURE} must be 0-90, got {vram_pressure}"
    pressure_args = ["--vram-pressure", str(vram_pressure)] if vram_pressure > 0 else []
    thresholds = _threshold_args()
    profiler_defaults = _CONTENTION_PROFILER_DEFAULTS if phase2_roles else None
    profiler_thresholds = _threshold_args(profiler_defaults)
    count = len(gpu_indices)
    multi = count > 1

    def rdir(idx: int) -> str:
        return f"{base_dir}/gpu{idx}" if multi else base_dir

    def cmd(idx: int, label: str, role: str, extra: list[str]) -> str:
        return _role_command(
            ld=ld,
            binary=binary,
            gpu_index=idx,
            duration=duration,
            results_dir=rdir(idx),
            label=label,
            role=role,
            extra=extra,
        )

    stop_grace = duration + 60
    procs: dict[tuple[int, str], AbstractBackgroundProcess] = {}
    verdicts: dict[int, ExecutionResult] = {}
    results: dict[tuple[int, str], ExecutionResult] = {}
    try:
        for idx in gpu_indices:
            # 0700 preserves the original's shared-system security posture.
            setup = target_executor.run(f"mkdir -p {rdir(idx)} && chmod 700 {rdir(idx)}")
            assert setup.ok, f"could not create results dir {rdir(idx)}: {setup.stderr[:400]}"

        # --- Phase 1: profiler alone per GPU, for an uncontended baseline ---
        # Thresholds apply; VRAM pressure never does (mirrors the scripts).
        if run_profiler:
            for idx in gpu_indices:
                procs[(idx, "profiler")] = target_executor.start_background(
                    cmd(idx, "profiler", "profiler", profiler_thresholds),
                    timeout=stop_grace,
                    console_label=f"gpu{idx}/profiler",
                )
            for idx in gpu_indices:
                _await_baseline(target_executor, rdir(idx), procs[(idx, "profiler")], duration)

        # --- Phase 2: contention roles per GPU, launched simultaneously ---
        for pos, idx in enumerate(gpu_indices):
            roles = list(phase2_roles)
            if enable_ipc and multi:
                peer = gpu_indices[(pos + 1) % count]
                roles.append(("ipc_xfer", "ipc_xfer", ["--peer-gpu", str(peer)]))
            for label, role, extra in roles:
                procs[(idx, label)] = target_executor.start_background(
                    cmd(idx, label, role, extra + pressure_args + thresholds + _role_health_args(role)),
                    timeout=stop_grace,
                    console_label=f"gpu{idx}/{label}",
                )

        _await_completion(target_executor, procs, {idx: rdir(idx) for idx in gpu_indices}, duration)
        results = {key: proc.stop(timeout=30.0) for key, proc in procs.items()}

        # --- Gate: run the vendored analyzer on each GPU's results directory ---
        for idx in gpu_indices:
            verdicts[idx] = _stage_and_check(target_executor, rdir(idx))
    finally:
        for proc in procs.values():
            with contextlib.suppress(Exception):
                proc.stop(timeout=15.0)
        target_executor.run(f"rm -rf {base_dir}")

    assert verdicts, "check_results.py never ran (workload setup failed)"
    for idx, verdict in verdicts.items():
        exit_codes = {lbl: results[(i, lbl)].exit_code for (i, lbl) in results if i == idx}
        assert verdict.ok, (
            f"rock_mps check_results reported FAILURE on gpu{idx} (exit={verdict.exit_code}).\n"
            f"Per-role exit codes: {exit_codes}\n"
            f"--- check_results report ---\n{verdict.stdout[-3500:]}\n"
            f"--- stderr ---\n{verdict.stderr[-600:]}"
        )


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
@pytest.mark.parametrize(("duration", "vram"), [(60, 0), (600, 60)])
def test_hip_multiprocess_app(target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request, duration, vram):
    """Full role suite on one GPU: all CLR roles in concurrent contention, with VRAM pressure sweep."""
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=[0],
        duration=duration,
        vram_pressure=vram,
        phase2_roles=_FULL_SUITE_ROLES,
        run_profiler=True,
    )


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.parametrize(("duration", "vram"), [(3600, 50), (86400, 60)])
def test_hip_multiprocess_app_soak(
    target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request, duration, vram
):
    """Full role suite on one GPU, soak durations (run_all_roles.sh 0 3600/86400)."""
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=[0],
        duration=duration,
        vram_pressure=vram,
        phase2_roles=_FULL_SUITE_ROLES,
        run_profiler=True,
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
@pytest.mark.gpu_count(2)
@pytest.mark.parametrize(
    ("duration", "vram"),
    [
        pytest.param(600, 0, marks=pytest.mark.ci.nightly),
        pytest.param(600, 60, marks=pytest.mark.ci.weekly),
    ],
)
def test_hip_multiprocess_all_gpus(
    target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request, duration, vram
):
    """Full suite on every GPU incl. cross-GPU ipc_xfer (run_all_gpus.sh)."""
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=list(range(2)),
        duration=duration,
        vram_pressure=vram,
        phase2_roles=_FULL_SUITE_ROLES,
        run_profiler=True,
        enable_ipc=True,
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.gpu_count(8)
@pytest.mark.parametrize(("duration", "vram"), [(86400, 60)])
def test_hip_multiprocess_all_gpus_soak(
    target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request, duration, vram
):
    """24 h fleet soak on 8 GPUs, 60% VRAM pressure (run_all_gpus.sh 86400 60)."""
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=list(range(8)),
        duration=duration,
        vram_pressure=vram,
        phase2_roles=_FULL_SUITE_ROLES,
        run_profiler=True,
        enable_ipc=True,
    )


@pytest.mark.hw.gpu
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
@pytest.mark.parametrize(
    ("role_a", "role_b", "vram"),
    [
        pytest.param("compute", "compiler", 0, marks=pytest.mark.ci.nightly, id="compute-compiler"),
        pytest.param("memory_mover", "library", 60, marks=pytest.mark.ci.weekly, id="memory_mover-library"),
        pytest.param("compute", "monitor", 0, marks=pytest.mark.ci.nightly, id="compute-monitor"),
        pytest.param("compute", "profiler", 0, marks=pytest.mark.ci.nightly, id="compute-profiler"),
    ],
)
def test_hip_multiprocess_role_pair(
    target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request, role_a, role_b, vram
):
    """Targeted two-role interaction on one GPU, 300 s (run_subset.sh)."""
    roles = [role_a, role_b]
    phase2 = [(r, r, []) for r in roles if r != "profiler"]
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=[0],
        duration=300,
        vram_pressure=vram,
        phase2_roles=phase2,
        run_profiler="profiler" in roles,
    )


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.parametrize(("role", "vram"), _SINGLE_ROLES, ids=[r for r, _ in _SINGLE_ROLES])
def test_hip_multiprocess_single_role(
    target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request, role, vram
):
    """Single-role baseline on one GPU, 60 s (rock_mps_test --role R)."""
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=[0],
        duration=60,
        vram_pressure=vram,
        phase2_roles=[(role, role, [])],
        run_profiler=False,
    )


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_hip_multiprocess_ipc(target_executor, ld_path: dict, rock_mps_binary: str, run_ctx, request):
    """Cross-process shared GPU memory: memory_mover producer + consumer, 120 s."""
    phase2 = [
        ("memory_mover_producer", "memory_mover", ["--ipc-role", "producer"]),
        ("memory_mover_consumer", "memory_mover", ["--ipc-role", "consumer"]),
    ]
    _run_suite(
        target_executor=target_executor,
        ld=ld_path["LD_LIBRARY_PATH"],
        binary=rock_mps_binary,
        base_dir=_base_dir(run_ctx, request),
        gpu_indices=[0],
        duration=120,
        vram_pressure=0,
        phase2_roles=phase2,
        run_profiler=False,
    )
