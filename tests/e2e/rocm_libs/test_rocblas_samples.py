# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocblas_samples.py -- ROCm rocBLAS example suite validation.

Validates the rocBLAS sample programs distributed with ROCm. Each sample
demonstrates a core rocBLAS operation: GEMM, TRSV, scalar operations, etc.
Tests verify that all samples execute successfully and produce expected output.
"""

import os
import re

import pytest

from framework.common import ExecutionResult

# Sample list — must be kept in sync between parametrized and static tests.
_ROCBLAS_SAMPLES = [
    "rocblas-example-c-dgeam",
    "rocblas-example-fortran-axpy",
    "rocblas-example-fortran-gemv",
    "rocblas-example-fortran-scal",
    "rocblas-example-gemv-graph-capture",
    "rocblas-example-hip-complex-her2",
    "rocblas-example-scal-multiple-strided-batch",
    "rocblas-example-scal-template",
    "rocblas-example-sgemm",
    "rocblas-example-sgemm-multiple-strided-batch",
    "rocblas-example-sgemm-strided-batched",
    "rocblas-example-solver",
    "rocblas-example-user-driven-tuning",
]


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.math_lib
@pytest.mark.runtime.medium
@pytest.mark.os.linux
@pytest.mark.parametrize("sample_name", _ROCBLAS_SAMPLES)
def test_rocblas_samples_dynamic(target_executor, rock_dir, sample_name):
    """Execute a single rocBLAS sample and validate its output."""
    cmd_dir = os.path.join(rock_dir, "bin")
    result = target_executor.run(f"cd {cmd_dir} && ./{sample_name}")

    assert result.ok, f"{sample_name} failed: {result.stderr}"
    _validate_rocblas_sample_output(result, sample_name)


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.math_lib
@pytest.mark.runtime.medium
@pytest.mark.os.linux
def test_rocblas_samples_static(target_executor, rock_dir):
    """Execute all rocBLAS samples in sequence and validate each output."""
    cmd_dir = os.path.join(rock_dir, "bin")

    for sample_name in _ROCBLAS_SAMPLES:
        result = target_executor.run(f"cd {cmd_dir} && ./{sample_name}")
        assert result.ok, f"{sample_name} failed: {result.stderr}"
        _validate_rocblas_sample_output(result, sample_name)


def _validate_rocblas_sample_output(result: ExecutionResult, test_case_name: str) -> None:
    """Validate sample output against sample-specific expected patterns."""
    data = result.stdout

    if test_case_name == "rocblas-example-user-driven-tuning":
        pattern = r"^(\d+)\s+solution\(s\)\s+found\s+that\s+can\s+solve\s+this\s+GEMM\."
        assert any(
            re.search(pattern, line) for line in data.splitlines()
        ), f"Expected solutions found message not in output for {test_case_name}"

    elif test_case_name == "rocblas-example-scal-template":
        lines = data.splitlines()
        found = False
        for lineno, line in enumerate(lines):
            if lineno + 1 < len(lines) and re.search(r"N\s+rocblas.*", line):
                if re.search(r"\d+\s+\d+.*", lines[lineno + 1]):
                    found = True
                    break
        assert found, f"Expected pattern not found in output for {test_case_name}"

    else:
        assert re.search(
            r"Pass", data, flags=re.IGNORECASE
        ), f"Expected 'Pass' not found in output for {test_case_name}"
