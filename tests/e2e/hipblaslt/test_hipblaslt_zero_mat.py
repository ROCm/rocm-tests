# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_zero_mat.py -- hipBLASLt FP8/BF8 GEMM with a zero input matrix.

For ``D = alpha*A*B + beta*C`` (alpha=1, beta=0), zeroing either input makes the
product zero, so every element of the FP32 output must be exactly 0. Runs four
parametrized checks (E4M3 / E5M2 inputs x {A zero, B zero}) against the
``hipblaslt_zero_mat`` binary so a failure isolates to one dtype/operand.

runtime.fast is declared explicitly; layer.math_lib is injected by the area profile.
"""

import pytest

_CHECKS = ["e4m3_azero", "e4m3_bzero", "e5m2_azero", "e5m2_bzero"]


@pytest.mark.runtime.fast
@pytest.mark.parametrize("check", _CHECKS)
def test_hipblaslt_zero_mat(target_executor, ld_path: dict, hipblaslt_zero_mat_binary: str, check: str):
    """FP8/BF8 GEMM with one zero operand yields an all-zero output."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {hipblaslt_zero_mat_binary} {check}")
    assert result.ok, (
        f"hipblaslt_zero_mat {check!r} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        f"hipblaslt_zero_mat {check}: PASSED" in result.stdout
    ), f"missing PASSED sentinel for {check!r}:\n{result.stdout[:2000]}"
    assert "FAILED" not in result.stdout, f"a check FAILED for {check!r}:\n{result.stdout[:2000]}"
