# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HIP binary invocation and assertion helpers shared across test suites.

Provides executor-transparent helpers for running prebuilt HIP binaries,
asserting pass/fail outcomes, checking binary availability, and detecting
hard GPU faults. Shared across hip_runtime, hwq_heuristic, rocprim, rocm_libs,
rocwmma, rocsolver, compiler, and rocm_examples test suites.
"""

from __future__ import annotations

import logging
import os
import shlex

import pytest

logger = logging.getLogger(__name__)

# Hard GPU fault signatures that indicate a kernel-level crash.
# A result.ok check alone does not catch these on all GPU configurations.
GPU_FAULT_MARKERS: tuple[str, ...] = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal error",
)

# Default success sentinels recognised across HIP sample and test binaries.
_DEFAULT_PASS_SENTINELS: tuple[str, ...] = (
    "PASSED",
    "Passed",
    "[PASS]",
    "All tests passed",
    "cpu_time_us",
)


def run_binary_with_ld(
    executor,
    binary: str,
    ld_library_path: str,
    args: str = "",
    timeout: float = 300.0,
    extra_env: str = "",
):
    """Run a prebuilt HIP binary with LD_LIBRARY_PATH set.

    Args:
        executor:          Any executor with a ``.run()`` method.
        binary:            Absolute path to the binary (will be shell-quoted).
        ld_library_path:   Value for ``LD_LIBRARY_PATH``.
        args:              Additional CLI arguments appended after the binary.
        timeout:           Seconds before the run is abandoned.
        extra_env:         Optional additional ``VAR=value`` pairs prefixed with ``env``.

    Returns:
        Execution result with ``.ok``, ``.stdout``, ``.stderr``, ``.exit_code``.
    """
    ld = shlex.quote(ld_library_path)
    b = shlex.quote(binary)
    env_prefix = f"env LD_LIBRARY_PATH={ld}"
    if extra_env:
        env_prefix = f"{env_prefix} {extra_env}"
    cmd = f"{env_prefix} {b}"
    if args:
        cmd = f"{cmd} {args}"
    logger.info("Running HIP binary: %s", cmd)
    return executor.run(cmd, timeout=timeout)


def assert_binary_passed(
    result,
    label: str,
    sentinels: tuple[str, ...] = _DEFAULT_PASS_SENTINELS,
) -> None:
    """Assert the binary exited 0 and printed a success sentinel in stdout.

    Args:
        result:    Execution result from ``executor.run()``.
        label:     Short name used in failure messages.
        sentinels: Tuple of success strings; at least one must appear in stdout.
    """
    assert result.ok, (
        f"{label} failed (exit={result.exit_code}):\n" f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert any(s in result.stdout for s in sentinels), (
        f"{label} exited 0 but no success sentinel {sentinels} found in stdout:\n" f"{result.stdout[:2000]}"
    )


def require_preinstalled_binary(
    binary_path: str,
    label: str,
    executor=None,
    *,
    fail_not_skip: bool = True,
) -> None:
    """Fail or skip when a binary is absent or non-executable on the executor.

    Handles both local (``os.access``) and remote (``test -x`` via executor)
    paths transparently.

    Args:
        binary_path:   Absolute path to the binary on the target node.
        label:         Human-readable name used in the failure/skip message.
        executor:      Remote executor; ``None`` means local.
        fail_not_skip: When ``True`` (default) calls ``pytest.fail``; when
                       ``False`` calls ``pytest.skip``.  Use ``False`` only for
                       optional features; use ``True`` for mandatory infrastructure.
    """
    reason = (
        f"{label} not found or not executable at {binary_path} — "
        f"ensure the required package is installed and --rock-dir points to the correct ROCm path."
    )
    if executor is not None:
        ok = executor.run(f"test -x {shlex.quote(binary_path)}", timeout=15.0).ok
    else:
        ok = os.path.isfile(binary_path) and os.access(binary_path, os.X_OK)

    if not ok:
        if fail_not_skip:
            pytest.fail(reason)
        else:
            pytest.skip(reason)


def assert_no_gpu_fault(output: str, label: str = "") -> None:
    """Fail when combined stdout/stderr contains hard GPU fault markers.

    GTest or a binary may exit 0 or 1 after a GPU fault without surfacing the
    cause in its own summary.  This guard catches kernel-level crashes before
    the caller inspects the numeric result.

    Args:
        output: Combined stdout + stderr from the run.
        label:  Short name used in the failure message.
    """
    for marker in GPU_FAULT_MARKERS:
        if marker in output:
            prefix = f"{label}: " if label else ""
            pytest.fail(f"{prefix}GPU fault detected — '{marker}' found in output.\n" f"output tail: {output[-2000:]}")


def assert_no_error_lines(
    output: str,
    label: str = "",
    patterns: tuple[str, ...] = (
        "Memory corruption",
        "out-of-bound",
        "Out of memory",
        "Memory access fault",
        "hang",
        "hung",
        "panic",
        "memleak",
    ),
) -> None:
    """Fail when any output line matches a known failure pattern.

    Args:
        output:   Combined stdout + stderr.
        label:    Short name used in the failure message.
        patterns: Tuple of substrings that indicate a failure condition.
    """
    matched = [line.strip()[:150] for line in output.splitlines() if any(p in line for p in patterns)]
    if matched:
        prefix = f"{label}: " if label else ""
        pytest.fail(f"{prefix}{len(matched)} error line(s) detected:\n" + "\n".join(f"  -> {ln}" for ln in matched[:5]))
