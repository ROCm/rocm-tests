# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RCCL unroll-factor dual-kernel build validation.

The first-party binary self-validates kernel symbol presence, heuristic selection,
override handling, invalid override rejection, and single-GPU collective
correctness for forced unroll factors.  The runtime env points it at the shared
``rccl-tests`` build and installed ``librccl.so``.
"""

import os

import pytest

from tests.e2e.rccl._workload import RESULTS_PASS_RE, all_reduce_perf_env


@pytest.mark.hw.gpu
@pytest.mark.runtime.medium
def test_rccl_dual_kernel_build(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    require_rccl,
    rccl_tests_build: str,
    rccl_dual_kernel_build_binary: str,
):
    """rccl_dual_kernel_build_test: all 5 unroll-factor sub-tests must pass (exit 0)."""
    env = all_reduce_perf_env(
        rock_dir=rock_dir,
        ld_library_path=ld_path["LD_LIBRARY_PATH"],
        all_reduce_perf=os.path.join(rccl_tests_build, "all_reduce_perf"),
        rccl_lib=os.path.join(rock_dir, "lib", "librccl.so"),
    )
    result = target_executor.run(f"{env} {rccl_dual_kernel_build_binary}", timeout=900)
    assert result.ok, (
        f"rccl_dual_kernel_build_test failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:1000]}"
    )
    assert RESULTS_PASS_RE.search(result.stdout), (
        "rccl_dual_kernel_build_test did not report '0 failed':\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:1000]}"
    )
