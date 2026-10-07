# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hipblas_samples.py -- hipBLAS sample binary validation.

Runs 12 hipBLAS example binaries and asserts the expected pass token in
stdout.  Binaries are resolved by ``hipblas_samples_bin_dir`` (defined in
``conftest.py``) in this priority order:

1. ``rock_dir/bin`` — pre-installed via the ``hipblas-samples`` package.
2. A cmake build from a sparse clone of ``ROCm/rocm-libraries``, using the
   framework's ``external_build.clone_repo`` + ``cmake_build_dir`` APIs.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Sample binary definitions: (binary_name, expected_pass_token)
# ---------------------------------------------------------------------------

_SAMPLES: list[tuple[str, str]] = [
    ("hipblas-example-bfdot-hip-bfloat16", "BFDOT TEST PASSES"),
    ("hipblas-example-gemmEx-fortran", "GEMMEX TEST PASS"),
    ("hipblas-example-c", "SSCAL TEST PASSES"),
    ("hipblas-example-gemmEx", "PASS"),
    ("hipblas-example-hgemm-half", "PASS"),
    ("hipblas-example-hip-complex-her2", "PASS"),
    ("hipblas-example-scal-ex", "SCALEX TEST PASSES"),
    ("hipblas-example-sgemm", "PASS"),
    ("hipblas-example-sgemm-strided-batched", "PASS"),
    ("hipblas-example-sscal", "SSCAL TEST PASSES"),
    ("hipblas-example-sscal-fortran", "SSCAL TEST PASS"),
    ("hipblas-example-strmm", "PASS"),
]


@pytest.mark.runtime.fast
@pytest.mark.parametrize(("binary_name", "pass_token"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
def test_hipblas_sample(
    hipblas_samples_bin_dir: str,
    target_executor,
    ld_path: dict,
    binary_name: str,
    pass_token: str,
):
    """Run a single hipBLAS sample binary and assert the pass token in stdout."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = os.path.join(hipblas_samples_bin_dir, binary_name)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary}",
        timeout=300.0,
    )
    assert result.ok, (
        f"{binary_name} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert pass_token in result.stdout, (
        f"{binary_name}: expected '{pass_token}' in stdout:\n{result.stdout[:2000]}"
    )
