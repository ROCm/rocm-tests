# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_async_mixed_precision_workflow.py -- rocSOLVER async / mixed-precision LAPACK workflows.

Binary: tests/e2e/rocm_libs/src/async_mixed_precision_workflow.cpp

Validates three integrated rocSOLVER + rocBLAS workflows that stress async
operations, mixed precision, and numerical robustness:
    1. Async transfer-compute overlap pipeline (DPOTRF across HIP streams).
    2. Mixed-precision iterative refinement (SGETRF + DGETRF).
    3. Ill-conditioning robustness (progressive DGETRF condition degradation).

Legacy PASS criteria (parse_pass_fail_re, case-insensitive) -- the summary must show:
    "Total:  <N>"   (regex: total:\\s+\\d+)
    "Passed: <N>"   (regex: passed:\\s+\\d+)
    "Failed: 0"     (regex: failed:\\s+0)
(FAIL on any of: FAILED, Error, ERROR, Aborted, failed!)
"""

import re

import pytest

from tests.e2e.rocm_libs._workload import solver_workload


@pytest.mark.runtime.medium
def test_async_mixed_precision_workflow(
    target_executor,
    ld_path: dict,
    async_mixed_precision_workflow_binary: str,
    rocblas_library_guard,
    workload_scale: str,
):
    """Validate the rocSOLVER async/mixed-precision triple workflow (Failed: 0)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    # Size knob via workload_scale (--size/--matrices/--streams); the pass criteria
    # below are identical regardless of size.
    wl = solver_workload("async_mixed_precision_workflow", workload_scale)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {async_mixed_precision_workflow_binary} {wl.args}",
        timeout=wl.timeout,
    )
    assert result.ok, (
        f"async_mixed_precision_workflow failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    assert re.search(r"(?i)total:\s+\d+", result.stdout), f"Expected 'Total: <N>' in stdout:\n{result.stdout[:3000]}"
    assert re.search(r"(?i)passed:\s+\d+", result.stdout), f"Expected 'Passed: <N>' in stdout:\n{result.stdout[:3000]}"
    assert re.search(r"(?i)failed:\s+0", result.stdout), f"Expected 'Failed: 0' in stdout:\n{result.stdout[:3000]}"
