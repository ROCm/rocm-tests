# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_small_sliding_contact_trsv.py -- rocBLAS strided-batched TRSV big-batch path.

Validates the rocBLAS strided-batched triangular solve (STRSV) for a contact
mechanics workload: 100,000 independent 3x3 triangular systems solved in parallel.
Exercises the rocBLAS big-batch TRSV kernel path (N < 128, batch_count > 16*N).
Validates against CPU reference solution with tolerance 1e-4. Also verifies a
physics identity check (K^-1 * K * gap = gap).

Binary compiled via CMake from:
    tests/e2e/rocm_libs/src/small_sliding_contact.cpp

Smoke args: N=3, BATCH_COUNT=100000 (< 30 s)

Markers auto-injected by CATEGORY_PROFILES (tests/e2e/rocm_libs):
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

runtime.fast is declared explicitly.
"""

import pytest

from tests.e2e.rocm_libs.conftest import check_rocblas_library


@pytest.mark.runtime.fast
def test_small_sliding_contact_trsv(
    target_executor,
    ld_path: dict,
    small_sliding_contact_binary: str,
    rock_dir: str,
):
    """Validate rocBLAS big-batch STRSV for contact mechanics (N=3, batch=100K).

    Args:
        target_executor:             Location-transparent GPU executor.
        ld_path:                     LD_LIBRARY_PATH dict for ROCm libs.
        small_sliding_contact_binary: Path to compiled binary.
        rock_dir:                    ROCm install root (for rocBLAS library check).
    """
    check_rocblas_library(rock_dir)
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {small_sliding_contact_binary} 3 100000")
    assert result.ok, (
        f"small_sliding_contact_trsv failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASSED" in result.stdout, f"Expected 'PASSED' in stdout:\n{result.stdout[:2000]}"
