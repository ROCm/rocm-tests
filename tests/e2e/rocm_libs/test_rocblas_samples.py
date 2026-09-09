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
import shlex

import pytest

from framework.common import ExecutionResult

# Sample list for parametrized test execution.
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


@pytest.mark.runtime.medium
@pytest.mark.parametrize("sample_name", _ROCBLAS_SAMPLES)
def test_rocblas_samples_dynamic(target_executor, rock_dir, sample_name, rocblas_library_guard, ld_path: dict):
    """Execute a single rocBLAS sample and validate its output."""
    cmd_dir = os.path.join(rock_dir, "bin")
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={shlex.quote(ld)} {shlex.quote(f'{cmd_dir}/{sample_name}')}", timeout=900.0
    )

    assert (
        result.ok
    ), f"{sample_name} failed (exit={result.exit_code}): stdout: {result.stdout[:3000]} stderr: {result.stderr[:800]}"
    _validate_rocblas_sample_output(result, sample_name)


def _validate_rocblas_sample_output(result: ExecutionResult, test_case_name: str) -> None:
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
            if (
                lineno + 1 < len(lines)
                and re.search(r"N\s+rocblas.*", line)
                and re.search(r"\d+\s+\d+.*", lines[lineno + 1])
            ):
                found = True
                break
        assert found, f"Expected pattern not found in output for {test_case_name}"

    else:
        assert re.search(
            r"passed|pass|PASS|PASSED|all tests passed|All tests passed", data, flags=re.IGNORECASE
        ), f"Expected 'Pass' not found in output for {test_case_name}"
