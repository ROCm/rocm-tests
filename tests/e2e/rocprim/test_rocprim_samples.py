# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocprim_samples.py -- rocPRIM sample workflow system tests.

Uses the public rocPRIM sample-style workflows from the legacy sample suite into
the shared rocPRIM CMake/GTest binary used by tests/e2e/rocprim/.
"""

import pytest

_SAMPLE_FILTERS = [
    pytest.param("RocprimSampleTests.RunningStatistics", id="running-statistics"),
    pytest.param("RocprimSampleTests.TopKFrequency", id="topk-frequency"),
    pytest.param(
        "RocprimSampleTests.MlFeatureEngineering",
        id="ml-feature-engineering",
    ),
    pytest.param("RocprimSampleTests.EtlWorkflow", id="etl-workflow"),
]


@pytest.mark.runtime.fast
@pytest.mark.parametrize("gtest_filter", _SAMPLE_FILTERS)
def test_rocprim_samples(
    target_executor,
    ld_path: dict,
    rocprim_tests_binary: str,
    gtest_filter: str,
):
    """Run one rocPRIM sample workflow and validate its CPU-reference checks."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} " f"{rocprim_tests_binary} --gtest_filter={gtest_filter}",
        timeout=600.0,
    )
    assert result.ok, (
        f"rocprim sample {gtest_filter} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "FAILED" not in result.stdout, f"rocprim sample {gtest_filter} reported failures:\n{result.stdout[:2000]}"
    assert (
        "PASSED" in result.stdout or "All tests passed!" in result.stdout
    ), f"rocprim sample {gtest_filter} pass token not found in stdout:\n{result.stdout[:2000]}"
