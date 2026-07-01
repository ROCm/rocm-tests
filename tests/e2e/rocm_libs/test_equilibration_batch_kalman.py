# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_equilibration_batch_kalman.py -- rocSOLVER equilibration / batch / Kalman workflows.

Binary: tests/e2e/rocm_libs/src/equilibration_batch_kalman.cpp

Validates three integrated rocSOLVER + rocBLAS workflows in one binary:
    1. Matrix equilibration and scaling (GETRF + GETRS, double precision).
    2. Asynchronous batched Cholesky across multiple HIP streams (ZPOTRF).
    3. Kalman filter update cycle (DGEMM + DPOTRF + DTRSM).

Legacy PASS criteria (parse_pass_fail_re) — all three lines must appear:
    "Equilibration workflow completed successfully!"
    "Async batch processing completed successfully!"
    "Kalman filter update completed successfully!"
(FAIL on any of: FAILED, Error, ERROR, Aborted, failed!)
"""

import pytest

from tests.e2e.rocm_libs._workload import solver_workload


@pytest.mark.runtime.medium
def test_equilibration_batch_kalman(
    target_executor,
    ld_path: dict,
    equilibration_batch_kalman_binary: str,
    rocblas_library_guard,
    workload_scale: str,
):
    """Validate the rocSOLVER equilibration/batch/Kalman triple workflow."""
    ld = ld_path["LD_LIBRARY_PATH"]
    # Size knob via workload_scale (--size/--nrhs/--batch/--streams/--m); the pass
    # criteria below are identical regardless of size.
    wl = solver_workload("equilibration_batch_kalman", workload_scale)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {equilibration_batch_kalman_binary} {wl.args}",
        timeout=wl.timeout,
    )
    assert result.ok, (
        f"equilibration_batch_kalman failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    for marker in (
        "Equilibration workflow completed successfully!",
        "Async batch processing completed successfully!",
        "Kalman filter update completed successfully!",
    ):
        assert marker in result.stdout, f"Expected '{marker}' in stdout:\n{result.stdout[:3000]}"
