# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_small_sliding_contact_trsv.py -- rocBLAS strided-batched TRSV big-batch smoke.

Validates 100,000 independent 3x3 triangular systems via rocBLAS STRSV (smoke: N=3, batch=100K).
Binary: tests/e2e/rocm_libs/src/small_sliding_contact.cpp
Checks: "PASSED" in stdout.
"""

import pytest


@pytest.mark.runtime.fast
def test_small_sliding_contact_trsv(
    target_executor,
    ld_path: dict,
    small_sliding_contact_binary: str,
    rocblas_library_guard,
):
    """Validate rocBLAS big-batch STRSV for contact mechanics (N=3, batch=100K)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {small_sliding_contact_binary} 3 100000", timeout=300.0)
    assert result.ok, (
        f"small_sliding_contact_trsv failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASSED" in result.stdout, f"Expected 'PASSED' in stdout:\n{result.stdout[:2000]}"
