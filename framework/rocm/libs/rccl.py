# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
rccl.py -- RCCL collective communication benchmark wrappers.

Provides helpers for launching RCCL collective operations (AllReduce, Broadcast,
AllGather) via ``rccl-tests`` binaries and parsing their bandwidth/latency output
for assertions and baseline comparisons.

All helpers delegate execution to the provided executor so they work in local
(LocalExecutor), remote (SshExecutor with gpu_indices), and container (ContainerExecutor)
contexts without change.

Usage::

    from framework.rocm.libs.rccl import check_rccl_available, run_allreduce, RcclResult

    def test_allreduce(target_executor):
        assert check_rccl_available(target_executor), "RCCL not installed"
        result = run_allreduce(target_executor, size_mb=256, n_gpus=2)
        assert result.bandwidth_gbps > 100
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

# stdout substrings that unambiguously indicate an rccl-tests run failed,
# independent of the bandwidth numbers (used by correctness_ok()).
_FAILURE_MARKERS: tuple[str, ...] = (
    "aborted",
    "test failure",
    "segmentation fault",
    "out of memory",
)


@dataclass
class RcclResult:
    """Parsed output from an RCCL collective operation benchmark.

    Attributes:
        operation:      Collective name (e.g. ``"allreduce"``).
        size_mb:        Message size in MB.
        n_gpus:         Number of GPUs used.
        bandwidth_gbps: Bus bandwidth in GB/s.
        latency_us:     Latency in microseconds.
        passed:         True if the operation completed without error.
        raw_output:     Full stdout from the benchmark run.
    """

    operation: str
    size_mb: int
    n_gpus: int
    bandwidth_gbps: float = 0.0
    latency_us: float = 0.0
    passed: bool = False
    raw_output: str = ""


def check_rccl_available(executor: AbstractExecutor) -> bool:
    """Return True if the RCCL shared library is detectable on the executor host.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        True if RCCL is installed and loadable via ``ctypes``.
    """
    script = (
        "import ctypes, ctypes.util; "
        "lib = ctypes.util.find_library('rccl'); "
        "print('RCCL_FOUND' if lib else 'RCCL_MISSING')"
    )
    result = executor.run(f'python3 -c "{script}"')
    return result.ok and "RCCL_FOUND" in result.stdout


def rccl_version(executor: AbstractExecutor) -> str | None:
    """Return the RCCL library version string.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        Version string, or None if unavailable.
    """
    script = (
        "import ctypes, ctypes.util; "
        "lib = ctypes.util.find_library('rccl'); "
        "rccl = ctypes.CDLL(lib) if lib else None; "
        "ver = ctypes.c_int(0); "
        "rccl.ncclGetVersion(ctypes.byref(ver)) if rccl else None; "
        "v = ver.value; "
        "print(f'{v//10000}.{(v//100)%100}.{v%100}') if v else print('')"
    )
    result = executor.run(f'python3 -c "{script}"')
    if result.ok and result.stdout.strip():
        return result.stdout.strip()
    return None


def run_perf(
    executor: AbstractExecutor,
    binary: str,
    *,
    n_gpus: int = 2,
    extra_args: str = "",
    env: dict[str, str] | None = None,
    launcher: str = "",
    operation: str = "perf",
    size_mb: int = 0,
    timeout: float = 300.0,
) -> RcclResult:
    """Run any ``rccl-tests`` ``*_perf`` binary and return parsed results.

    Single generic runner shared by every rccl-tests-based test (multi-GPU,
    HIP-graph, PAT-algorithm, plus the ``run_allreduce`` / ``run_broadcast`` /
    ``run_allgather`` convenience wrappers below). It only builds the command
    string and delegates execution to *executor*, so it works unchanged across
    local, SSH, and container contexts.

    The benchmark binary is invoked single-process / multi-thread via ``-g
    <n_gpus>`` (one thread per visible GPU). Environment variables (e.g.
    ``LD_LIBRARY_PATH`` for a TheRock install, ``NCCL_ALGO=PAT``) are prepended
    with ``env`` so callers never have to format the command themselves.

    Args:
        executor:   Any executor / ``NodeExecutorGroup`` with a ``.run()`` method.
        binary:     Path to (or PATH name of) the ``*_perf`` binary.
        n_gpus:     Number of GPU devices to use (maps to ``-g``).
        extra_args: Verbatim extra rccl-tests flags (e.g. ``"-b 16 -e 1G -f 2 -G 1"``).
        env:        Environment variables to export before the binary runs.
        launcher:   Optional command prefix such as ``"mpirun -np 2"``.
        operation:  Label stored on the returned ``RcclResult`` (for reporting).
        size_mb:    Message size in MB recorded on the result (metadata only).
        timeout:    Maximum seconds to wait.

    Returns:
        ``RcclResult`` with parsed bandwidth/latency and the full ``raw_output``.
    """
    cmd = f"{binary} -g {n_gpus}"
    if launcher:
        cmd = f"{launcher} {cmd}"
    if extra_args:
        cmd = f"{cmd} {extra_args}"
    if env:
        prefix = "env " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
        cmd = f"{prefix} {cmd}"
    result = executor.run(cmd, timeout=timeout)
    return _parse_rccl_output(operation, size_mb, n_gpus, result.stdout, result.ok)


def correctness_ok(result: RcclResult) -> bool:
    """Return True when an rccl-tests run completed *and* passed data validation.

    Distinct from ``RcclResult.passed`` (which is bandwidth-based): this checks
    the data-correctness path used by the functional tests (HIP-graph, PAT,
    multi-GPU). With checking enabled (``-c 1``) rccl-tests prints ``OK`` per
    row / ``# Out of bounds values : 0 OK`` and exits cleanly; any abort,
    mismatch (``FAILED``), or known failure marker means the run did not pass.

    Args:
        result: A ``RcclResult`` produced by :func:`run_perf`.

    Returns:
        True if the run looks correct, False otherwise.
    """
    out = result.raw_output
    low = out.lower()
    if any(marker in low for marker in _FAILURE_MARKERS):
        return False
    if "FAILED" in out:
        return False
    return "OK" in out


def run_allreduce(
    executor: AbstractExecutor,
    size_mb: int = 256,
    n_gpus: int = 2,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> RcclResult:
    """Run RCCL AllReduce benchmark via ``rccl-tests`` and return parsed results.

    Args:
        executor:  Any executor with a ``.run()`` method.
        size_mb:   Message size in MB.
        n_gpus:    Number of GPU devices to use.
        n_warmup:  Warm-up iterations (excluded from bandwidth calculation).
        n_iters:   Measurement iterations.

    Returns:
        RcclResult with bandwidth_gbps, latency_us, and pass/fail status.
        Requires ``all_reduce_perf`` from ``rccl-tests`` on the runner PATH.
    """
    size_bytes = size_mb * 1024 * 1024
    return run_perf(
        executor,
        "all_reduce_perf",
        n_gpus=n_gpus,
        extra_args=f"-b {size_bytes} -e {size_bytes} -f 2 -w {n_warmup} -n {n_iters}",
        operation="allreduce",
        size_mb=size_mb,
        timeout=120.0,
    )


def run_broadcast(
    executor: AbstractExecutor,
    size_mb: int = 256,
    n_gpus: int = 2,
) -> RcclResult:
    """Run RCCL Broadcast benchmark and return parsed results.

    Args:
        executor: Any executor with a ``.run()`` method.
        size_mb:  Message size in MB.
        n_gpus:   Number of GPU devices to use.

    Returns:
        RcclResult with bandwidth and latency metrics.
        Requires ``broadcast_perf`` from ``rccl-tests``.
    """
    size_bytes = size_mb * 1024 * 1024
    return run_perf(
        executor,
        "broadcast_perf",
        n_gpus=n_gpus,
        extra_args=f"-b {size_bytes} -e {size_bytes} -f 2",
        operation="broadcast",
        size_mb=size_mb,
        timeout=120.0,
    )


def run_allgather(
    executor: AbstractExecutor,
    size_mb: int = 256,
    n_gpus: int = 2,
) -> RcclResult:
    """Run RCCL AllGather benchmark and return parsed results.

    Args:
        executor: Any executor with a ``.run()`` method.
        size_mb:  Message size in MB.
        n_gpus:   Number of GPU devices to use.

    Returns:
        RcclResult with bandwidth and latency metrics.
        Requires ``all_gather_perf`` from ``rccl-tests``.
    """
    size_bytes = size_mb * 1024 * 1024
    return run_perf(
        executor,
        "all_gather_perf",
        n_gpus=n_gpus,
        extra_args=f"-b {size_bytes} -e {size_bytes} -f 2",
        operation="allgather",
        size_mb=size_mb,
        timeout=120.0,
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _parse_rccl_output(
    operation: str,
    size_mb: int,
    n_gpus: int,
    stdout: str,
    ok: bool,
) -> RcclResult:
    """Parse ``rccl-tests`` stdout and extract bandwidth + latency.

    ``rccl-tests`` output column order::

        # Size  Count  Type  Redop  Root  Time  AlgBW  BusBW

    The last non-comment data line is used (peak measurement).

    Args:
        operation: Collective name.
        size_mb:   Message size in MB (for result metadata).
        n_gpus:    Number of GPUs used.
        stdout:    Raw stdout from the benchmark run.
        ok:        Whether the command exited cleanly.

    Returns:
        RcclResult with parsed or zero-value metrics.
    """
    bw_gbps = 0.0
    latency_us = 0.0

    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                latency_us = float(parts[5])
                bw_gbps = float(parts[7])  # BusBW column
                break
            except (ValueError, IndexError):
                continue

    return RcclResult(
        operation=operation,
        size_mb=size_mb,
        n_gpus=n_gpus,
        bandwidth_gbps=bw_gbps,
        latency_us=latency_us,
        passed=ok and bw_gbps > 0,
        raw_output=stdout,
    )
