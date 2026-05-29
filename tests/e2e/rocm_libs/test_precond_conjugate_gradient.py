# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_precond_conjugate_gradient.py -- Preconditioned Conjugate Gradient solver e2e.

Validates a full ILU(0)-preconditioned Conjugate Gradient solver pipeline that
exercises: hipBLAS (dot, axpy, scal -- dense vector ops), hipSPARSE (SpMV and SpSV
triangular solve), rocSOLVER (ILU(0) factorization analysis + numeric), and multi-GPU
P2P transfers (optional --ngpus 2 variant). The solver converges to CPU reference
solution with max error < 1e-5.

Binary compiled via CMake from:
    tests/e2e/rocm_libs/src/precond_conjugate_gradient.cpp

Smoke args: --size 128 --ngpus 1 (< 30 s)
Multi-GPU variant: --size 128 --ngpus 2 (requires 2 GPUs)

runtime.fast is declared explicitly on the single-GPU function.
"""

import re

import pytest


@pytest.mark.runtime.fast
def test_precond_conjugate_gradient_single_gpu(
    target_executor,
    ld_path: dict,
    precond_conjugate_gradient_binary: str,
):
    """Validate ILU(0) preconditioned CG convergence on a single GPU.

    Stack exercised: hipBLAS (dot/axpy/scal) + hipSPARSE (SpMV/SpSV) +
    rocSOLVER (ILU0 factorization) + HIP runtime + KFD/driver.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}" f" {precond_conjugate_gradient_binary} --size 128 --ngpus 1"
    )
    assert result.ok, (
        f"precond_conjugate_gradient failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2500]}\nstderr: {result.stderr[:600]}"
    )
    assert re.search(
        r"Total Errors:\s+0\b", result.stdout
    ), f"Expected zero total errors in output:\n{result.stdout[:2500]}"


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.fast
def test_precond_conjugate_gradient_multi_gpu(
    target_executor,
    ld_path: dict,
    precond_conjugate_gradient_binary: str,
):
    """Validate ILU(0) preconditioned CG with 2-GPU P2P transfers."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}" f" {precond_conjugate_gradient_binary} --size 128 --ngpus 2"
    )
    assert result.ok, (
        f"precond_conjugate_gradient multi-GPU failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2500]}\nstderr: {result.stderr[:600]}"
    )
    assert re.search(
        r"Total Errors:\s+0\b", result.stdout
    ), f"Expected zero total errors in output:\n{result.stdout[:2500]}"
