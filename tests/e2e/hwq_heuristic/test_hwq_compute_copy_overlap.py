# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hwq_compute_copy_overlap.py -- HIP async compute and memcpy overlap validation.

Validates that GPU compute kernels and async memory copies execute concurrently
on separate HIP streams. With --check-overlap: fails if the measured mixed
execution time is >= the sum of individual compute-only and copy-only phases,
proving the hardware is actually overlapping compute and copy in parallel.

The test exercises the CLR queue heuristic's ability to assign compute kernels
and DMA transfers to separate GPU hardware queues.

Regression for SWDEV-567580 (CLR hardware queue heuristic defects).

Binary compiled via CMake from:
    tests/e2e/hwq_heuristic/src/hwq_compute_copy_overlap_test.cpp

Args: --compute-streams=4 --copy-streams=4 --elems=1048576 --iters=64
      --rounds=4 --check-overlap
Parametrized over DEBUG_HIP_DYNAMIC_QUEUES modes 0, 1, 2.

runtime.fast is declared explicitly.
"""

import pytest


@pytest.mark.runtime.fast
@pytest.mark.parametrize("dq_mode", [0, 1, 2])
def test_hwq_compute_copy_overlap(
    target_executor,
    ld_path: dict,
    hwq_compute_copy_overlap_binary: str,
    dq_mode: int,
):
    """Validate concurrent compute + memcpy execution on separate HIP streams.

    Uses --check-overlap to make pass/fail deterministic: fails if
    t_mixed >= t_compute + t_copy (i.e., operations are not truly overlapping).
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" DEBUG_HIP_DYNAMIC_QUEUES={dq_mode}"
        f" {hwq_compute_copy_overlap_binary}"
        f" --compute-streams=4 --copy-streams=4"
        f" --elems=1048576 --iters=64 --rounds=4 --check-overlap"
    )
    assert result.ok, (
        f"hwq_compute_copy_overlap dq_mode={dq_mode} failed"
        f" (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Binary writes "FAIL: mixed time not below sum of parts" to stderr when
    # --check-overlap detects insufficient overlap (exits 1).  Assert here as a
    # belt-and-suspenders guard independent of the exit code.
    assert "FAIL" not in result.stderr, (
        f"hwq_compute_copy_overlap dq_mode={dq_mode}: unexpected FAIL in stderr "
        f"(exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASS" in result.stdout, (
        f"hwq_compute_copy_overlap dq_mode={dq_mode}: expected 'PASS' in stdout:\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
