# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_zero_mat.py -- hipBLASLt FP8/BF8 GEMM with a zero input matrix.

For ``D = alpha*A^T*B + beta*C`` (alpha=beta=1, C zeroed), zeroing either input
makes the product zero, so every element of the BFloat16 output must be exactly
0 and finite. A is FP8 (E4M3) and B is BF8 (E5M2); the run repeats the GEMM so a
transient artefact is caught rather than a single lucky pass.

runtime.medium is declared explicitly; the remaining markers are injected by the
area profile.
"""

import pytest

_CHECKS = ["azero", "bzero"]

# The repeated GEMM plus a full host-side scan of the 8192x32768 output runs well
# past the executor's 300s default.
_TIMEOUT = 1800.0


@pytest.mark.runtime.medium
@pytest.mark.parametrize("check", _CHECKS)
def test_hipblaslt_zero_mat(target_executor, ld_path: dict, hipblaslt_zero_mat_binary: str, check: str):
    """FP8/BF8 GEMM with one zero operand yields an all-zero output."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {hipblaslt_zero_mat_binary} {check}", timeout=_TIMEOUT)
    assert result.ok, (
        f"hipblaslt_zero_mat {check!r} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        f"hipblaslt_zero_mat {check}: PASSED" in result.stdout
    ), f"missing PASSED sentinel for {check!r}:\n{result.stdout[:2000]}"
    assert "FAILED" not in result.stdout, f"a check FAILED for {check!r}:\n{result.stdout[:2000]}"
