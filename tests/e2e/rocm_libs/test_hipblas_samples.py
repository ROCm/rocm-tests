# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hipblas_samples.py -- hipBLAS sample binary validation.

Runs 12 hipBLAS example binaries and asserts the expected pass token in
stdout.  Binaries are resolved by ``hipblas_samples_bin_dir`` (defined in
``conftest.py``) in this priority order:

1. ``rock_dir/bin`` -- pre-installed via the ``hipblas-samples`` package.
2. A cmake build from a sparse clone of ``ROCm/rocm-libraries``, using the
   framework's ``external_build.clone_repo`` + ``cmake_build_dir`` APIs.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# Sample binary definitions: (binary_name, pass_token, fail_token)
#
# fail_token mirrors the original test's explicit-failure check: binaries that
# report a relative error print either "PASS: max_relative_error =" or
# "FAIL: max_relative_error =" — both must be checked.  Other binaries only
# emit a fixed pass string; their fail_token is None.
# ---------------------------------------------------------------------------

_SAMPLES: list[tuple[str, str, str | None]] = [
    ("hipblas-example-bfdot-hip-bfloat16", "BFDOT TEST PASSES", None),
    ("hipblas-example-gemmEx-fortran", "GEMMEX TEST PASS", None),
    ("hipblas-example-c", "SSCAL TEST PASSES", None),
    ("hipblas-example-gemmEx", "PASS: max_relative_error =", "FAIL: max_relative_error ="),
    ("hipblas-example-hgemm-half", "PASS: max_relative_error =", "FAIL: max_relative_error ="),
    ("hipblas-example-hip-complex-her2", "PASS", None),
    ("hipblas-example-scal-ex", "SCALEX TEST PASSES", None),
    ("hipblas-example-sgemm", "PASS: max_relative_error =", "FAIL: max_relative_error ="),
    ("hipblas-example-sgemm-strided-batched", "PASS: max_relative_error =", "FAIL: max_relative_error ="),
    ("hipblas-example-sscal", "SSCAL TEST PASSES", None),
    ("hipblas-example-sscal-fortran", "SSCAL TEST PASS", None),
    ("hipblas-example-strmm", "PASS: max_relative_error =", "FAIL: max_relative_error ="),
]


@pytest.mark.runtime.fast  # type: ignore[attr-defined]
@pytest.mark.parametrize(
    ("binary_name", "pass_token", "fail_token"),
    _SAMPLES,
    ids=[s[0] for s in _SAMPLES],
)
def test_hipblas_sample(
    hipblas_samples_bin_dir: str,
    target_executor,
    ld_path: dict,
    binary_name: str,
    pass_token: str,
    fail_token: str | None,
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
    assert pass_token in result.stdout, f"{binary_name}: expected '{pass_token}' in stdout:\n{result.stdout[:2000]}"
    if fail_token is not None:
        assert (
            fail_token not in result.stdout
        ), f"{binary_name}: explicit failure token '{fail_token}' found in stdout:\n{result.stdout[:2000]}"
    assert "FAILED" not in result.stdout, f"{binary_name}: unexpected FAILED in stdout:\n{result.stdout[:2000]}"
