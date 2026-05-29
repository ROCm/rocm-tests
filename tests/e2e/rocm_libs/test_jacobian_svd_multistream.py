# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_jacobian_svd_multistream.py -- hipSOLVER Jacobi SVD concurrent multi-stream e2e.

Validates Jacobi SVD (GESVDJ) across 100 concurrent HIP streams, each processing
100 operations (10,000 SVD operations total). Verifies: (1) convergence in >= 1 sweep,
(2) reconstruction error ||A - U*S*V^T|| < 2e-5, (3) singular value sanity. Exercises
massive concurrent allocation and execution on real GPU hardware.

Binary compiled via CMake from:
    tests/e2e/rocm_libs/src/jacobian_svd_multistream.cpp

Smoke args: matrix_size=64 (64x64 matrices, ~1.2 GB VRAM, < 2 min)

runtime.medium is declared (100 streams x 100 SVDs may take 1-2 min on some GPUs).
"""

import pytest

from tests.e2e.rocm_libs.conftest import check_rocblas_library


@pytest.mark.runtime.medium
def test_jacobian_svd_multistream(
    target_executor,
    ld_path: dict,
    jacobian_svd_multistream_binary: str,
    rock_dir: str,
):
    """Validate Jacobi SVD correctness across 100 concurrent HIP streams."""
    check_rocblas_library(rock_dir)
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {jacobian_svd_multistream_binary} 64",
        timeout=180.0,
    )
    assert result.ok, (
        f"jacobian_svd_multistream failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    assert "SUCCESS" in result.stdout, f"Expected 'SUCCESS' in stdout:\n{result.stdout[:2000]}"
