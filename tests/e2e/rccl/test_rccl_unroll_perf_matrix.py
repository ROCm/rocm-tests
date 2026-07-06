# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RCCL unroll-factor performance matrix.

The first-party binary benchmarks ``all_reduce_perf`` across message sizes and
forced unroll factors, then self-checks its embedded 10%/20% tolerance rules.
This test asserts the binary's own ``0 failed`` result instead of adding another
performance threshold in Python.
"""

import os

import pytest

from tests.e2e.rccl._workload import RESULTS_PASS_RE, all_reduce_perf_env


@pytest.mark.hw.gpu
@pytest.mark.runtime.medium
def test_rccl_unroll_perf_matrix(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    require_rccl,
    rccl_tests_build: str,
    rccl_unroll_perf_matrix_binary: str,
):
    """rccl_unroll_perf_matrix_test: all unroll-factor perf sub-tests must pass (exit 0)."""
    env = all_reduce_perf_env(
        rock_dir=rock_dir,
        ld_library_path=ld_path["LD_LIBRARY_PATH"],
        all_reduce_perf=os.path.join(rccl_tests_build, "all_reduce_perf"),
    )
    result = target_executor.run(f"{env} {rccl_unroll_perf_matrix_binary}", timeout=1800)
    assert result.ok, (
        f"rccl_unroll_perf_matrix_test failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:1000]}"
    )
    assert RESULTS_PASS_RE.search(result.stdout), (
        "rccl_unroll_perf_matrix_test did not report '0 failed':\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:1000]}"
    )
