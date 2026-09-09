# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hipblas_samples.py -- hipBLAS pre-built sample binary validation.

Runs 12 hipBLAS example binaries shipped in ``rock_dir/bin`` and asserts the
expected pass token appears in stdout for each one.
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

_SENTINEL = "hipblas-example-bfdot-hip-bfloat16"


@pytest.fixture
def hipblas_samples_bin_dir(target_executor, ld_path: dict, rock_dir: str) -> str:
    """Return rock_dir/bin after verifying hipblas sample binaries are present."""
    ld = ld_path["LD_LIBRARY_PATH"]
    bin_dir = os.path.join(rock_dir, "bin")
    probe = target_executor.run(f"env LD_LIBRARY_PATH={ld} test -f {bin_dir}/{_SENTINEL} && echo OK")
    if not probe.ok or "OK" not in probe.stdout:
        pytest.skip(
            f"hipblas-samples not installed — binaries not found in {bin_dir}. "
            "Install the 'hipblas-samples' package from the ROCm repository."
        )
    return bin_dir


@pytest.mark.runtime.fast
@pytest.mark.parametrize(("binary_name", "pass_token"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
def test_hipblas_sample(
    hipblas_samples_bin_dir: str,
    target_executor,
    ld_path: dict,
    binary_name: str,
    pass_token: str,
):
    """Run a single pre-built hipBLAS sample binary and assert the pass token in stdout."""
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
