# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_sparse_csrrf_analysis_reuse.py -- rocSOLVER sparse CSRRF analysis-reuse refactorization.

Binary: tests/e2e/rocm_libs/src/sparse_csrrf_analysis_reuse.cpp

Validates the sparse iterative refactorization optimization: symbolic analysis is
performed ONCE (csrrf_analysis) and reused across multiple Cholesky
refactorizations (csrrf_refactchol) with updated matrix values + solves
(csrrf_solve) -- the production pattern for Newton iterations and time-stepping.

Legacy PASS criteria (parse_pass_fail_re):
    "Iterative refactorization workflow PASSED!"
(FAIL on any of: FAILED, Error, ERROR, Aborted)
"""

import pytest

from tests.e2e.rocm_libs._workload import solver_workload


@pytest.mark.runtime.medium
def test_sparse_csrrf_analysis_reuse(
    target_executor,
    ld_path: dict,
    sparse_csrrf_analysis_reuse_binary: str,
    hip_mempool_env: str,
    rocblas_library_guard,
    workload_scale: str,
):
    """Validate rocSOLVER sparse CSRRF analysis-reuse across refactorization iterations."""
    ld = ld_path["LD_LIBRARY_PATH"]
    # Size knob via workload_scale (-m matrix-size, -n nnz, -i iterations); the pass
    # criterion below is identical regardless of size.
    wl = solver_workload("sparse_csrrf_analysis_reuse", workload_scale)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {hip_mempool_env} {sparse_csrrf_analysis_reuse_binary} {wl.args}",
        timeout=wl.timeout,
    )
    assert result.ok, (
        f"sparse_csrrf_analysis_reuse failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    assert (
        "Iterative refactorization workflow PASSED!" in result.stdout
    ), f"Expected 'Iterative refactorization workflow PASSED!' in stdout:\n{result.stdout[:3000]}"
