# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_mini_residual_app.py -- hipBLASLt GEMM beta-path zero invariant smoke.

Validates D=0 when A=0 and C=0 across all heuristic algorithms (smoke: M=N=512, K=256, iters=5).
Binary: tests/e2e/hipblaslt/src/mini_residual_app.cpp
Checks: "[PASS]" in stdout.
"""

import pytest


@pytest.mark.runtime.fast
def test_mini_residual_app(
    target_executor,
    ld_path: dict,
    tensile_lib_path: str,
    mini_residual_app_binary: str,
):
    """Validate hipBLASLt GEMM beta-path zero invariant (FP8/BF16)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={tensile_lib_path}"
        f" {mini_residual_app_binary}"
        " --M 512 --N 512 --K 256 --iters 5",
        timeout=600.0,
    )
    assert result.ok, (
        f"mini_residual_app failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, f"Expected '[PASS]' in stdout:\n{result.stdout[:2000]}"
