# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hwq_null_stream_protection.py -- HIP null stream queue protection regression.

Validates the HIP CLR null stream queue is not released or corrupted after
heavy explicit-stream workloads. Three-phase test:

Phase 1 — Fill kernel on null stream, verify all elements == fill_value_1 (7).
Phase 2 — Launch heavy FMA kernels on N explicit streams to stress queue heuristics.
Phase 3 — Fill kernel on null stream again, verify all elements == fill_value_2 (42).

If the null stream's queue is incorrectly released between Phase 1 and Phase 3,
Phase 3 will write incorrect values and the test fails immediately with the
failing index and expected vs. actual values.

Regression for SWDEV-567580 (CLR hardware queue heuristic defects).

Binary compiled via CMake from:
    tests/e2e/hwq_heuristic/src/hwq_null_stream_protection_regr.cpp

Args: --n=8 --elems=262144 --iters=32
Parametrized over DEBUG_HIP_DYNAMIC_QUEUES modes 0, 1, 2.

runtime.fast is declared explicitly.
"""

import pytest


@pytest.mark.runtime.fast
@pytest.mark.parametrize("dq_mode", [0, 1, 2])
def test_hwq_null_stream_protection(
    target_executor,
    ld_path: dict,
    hwq_null_stream_protection_binary: str,
    dq_mode: int,
):
    """Validate null stream queue integrity after heavy explicit-stream workload."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" DEBUG_HIP_DYNAMIC_QUEUES={dq_mode}"
        f" {hwq_null_stream_protection_binary}"
        f" --n=8 --elems=262144 --iters=32"
    )
    assert result.ok, (
        f"hwq_null_stream_protection dq_mode={dq_mode} failed"
        f" (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Binary writes "FAIL: first/second null stream fill incorrect at index N"
    # to stderr when the fill values are wrong (exits 1).  Assert here as a
    # belt-and-suspenders guard independent of the exit code.
    assert "FAIL" not in result.stderr, (
        f"hwq_null_stream_protection dq_mode={dq_mode}: unexpected FAIL in stderr "
        f"(exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Positive progression check: all three phases must complete and print PASS.
    for phase_marker in ("phase1:", "phase2:", "phase3:", "PASS"):
        assert phase_marker in result.stdout, (
            f"hwq_null_stream_protection dq_mode={dq_mode}: expected '{phase_marker}' in stdout "
            f"(test may have aborted early):\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
