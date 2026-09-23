# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""test_hip_complex_her2.py — HIP complex API header validation via rocblas_cher2.

Validates that HIP complex number types (``hipFloatComplex``) interoperate correctly
with the rocBLAS public API when the ``ROCM_MATHLIBS_API_USE_HIP_COMPLEX`` API path is
active.  The workload allocates a 267x267 Hermitian matrix and two complex vectors on
the GPU, calls ``rocblas_cher2()`` (Hermitian rank-2 update), retrieves the result, and
verifies every upper-triangular element against the expected formula.  A "PASS" token on
stdout confirms success; any "FAIL" token or non-zero exit indicates a regression.
"""

import pytest


@pytest.mark.runtime.fast
def test_hip_complex_her2(
    target_executor,
    ld_path: dict,
    hip_complex_her2_binary: str,
    rocblas_library_guard,
) -> None:
    """Run the HIP complex API header workload and assert correctness.

    Exercises ``rocblas_cher2()`` through the ``hipFloatComplex``-typed API surface
    (``ROCM_MATHLIBS_API_USE_HIP_COMPLEX``) on a 267x267 upper-triangular system.
    The binary prints "PASS" on success and "FAIL" on any element mismatch.

    Args:
        target_executor: Framework executor dispatched to the assigned GPU node.
        ld_path: Dict containing ``LD_LIBRARY_PATH`` for the ROCm install.
        hip_complex_her2_binary: Path to the compiled ``hip_complex_her2`` binary.
        rocblas_library_guard: Session fixture; fails early if librocblas.so is absent.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {hip_complex_her2_binary}",
        timeout=120.0,
    )
    assert result.ok, f"hip_complex_her2 exited with non-zero status:\n{result.stderr}"
    assert "PASS" in result.stdout, f"Expected PASS token, got: {result.stdout!r}"
    assert "FAIL" not in result.stdout, f"Unexpected FAIL in output: {result.stdout!r}"
