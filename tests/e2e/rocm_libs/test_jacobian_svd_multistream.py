# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_jacobian_svd_multistream.py -- hipSOLVER Jacobi SVD concurrent multi-stream e2e.

Validates Jacobi SVD correctness across 100 concurrent HIP streams (smoke: matrix_size=64).
Binary: tests/e2e/rocm_libs/src/jacobian_svd_multistream.cpp
Checks: "SUCCESS" in stdout.
"""

import pytest


@pytest.mark.runtime.medium
def test_jacobian_svd_multistream(
    target_executor,
    ld_path: dict,
    jacobian_svd_multistream_binary: str,
    rocblas_library_guard,
):
    """Validate Jacobi SVD correctness across 100 concurrent HIP streams."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {jacobian_svd_multistream_binary} 64",
        timeout=900.0,
    )
    assert result.ok, (
        f"jacobian_svd_multistream failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    assert "SUCCESS" in result.stdout, f"Expected 'SUCCESS' in stdout:\n{result.stdout[:2000]}"
