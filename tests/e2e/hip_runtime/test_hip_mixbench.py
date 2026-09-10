# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hip_mixbench.py -- mixbench-hip mixed compute/memory microbenchmark validation.

Validates:
    1. The mixbench-hip benchmark builds, runs on an AMD GPU, and emits a
       well-formed CSV results section covering the single-, double-, and
       half-precision floating point plus integer throughput sweeps.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hip_runtime/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.medium
"""

from __future__ import annotations

import csv
from io import StringIO
import re

import pytest

_FLOAT_COLS_NAMES = ["Flops/byte", "execution time", "GFLOPS", "GB/sec"]
_INTEGER_COLS_NAMES = ["Iops/byte", "execution time", "GIOPS", "GB/sec"]
_VALUE_REGEX = r"([0-9]+(\.[0-9]+)?|inf)"


def _mixbench_failure(test_name: str, marker: str) -> dict:
    """Build the result mapping for a malformed mixbench CSV section."""
    return {
        marker: True,
        "testSuiteInfo": {"default": {test_name: {"status": False}}},
        "nTestSuites": 1,
        "nTestCases": 1,
        "status": False,
    }


def _parse_mixbench_output(text: str) -> dict:
    """Parse mixbench stdout into a structured throughput result mapping.

    The benchmark variant is inferred from the banner ("alternating" selects
    ``mixbench_alt``, "read-only" selects ``mixbench_ro``, otherwise ``sanity``).
    The CSV block begins three lines after the line containing "CSV" and excludes
    the final line; each column is sliced into single/double/half precision and
    integer throughput groups, and every field must match the numeric pattern.

    Args:
        text: Captured mixbench standard output.

    Returns:
        A result mapping whose ``status`` is ``True`` when the CSV section is
        well-formed, ``False`` when data is missing or non-numeric.
    """
    if "alternating" in text:
        test_name = "mixbench_alt"
    elif "read-only" in text:
        test_name = "mixbench_ro"
    else:
        test_name = "sanity"

    lines = text.splitlines()
    csv_index = [idx for idx, line in enumerate(lines) if "CSV" in line][0]  # noqa: RUF015
    csv_data = "\n".join(lines[csv_index + 3 : -1])

    reader = csv.reader(StringIO(csv_data), skipinitialspace=True)
    columns = [list(col) for col in zip(*list(reader), strict=False)]

    sp_data: list = []
    dp_data: list = []
    hp_data: list = []
    int_data: list = []
    for column in columns:
        try:
            sp_data.append(column[1:5])
            dp_data.append(column[5:9])
            hp_data.append(column[9:13])
            int_data.append(column[13:17])
        except IndexError:
            return _mixbench_failure(test_name, "data missing")
        for value in column:
            if re.search(_VALUE_REGEX, value) is None:
                return _mixbench_failure(test_name, "data incorrect")

    return {
        "single_precision": {"cols_names": _FLOAT_COLS_NAMES, "data": sp_data},
        "double_precision": {"cols_names": _FLOAT_COLS_NAMES, "data": dp_data},
        "half_precision": {"cols_names": _FLOAT_COLS_NAMES, "data": hp_data},
        "integer": {"cols_names": _INTEGER_COLS_NAMES, "data": int_data},
        "testSuiteInfo": {"default": {test_name: {"status": True}}},
        "nTestSuites": 1,
        "nTestCases": 1,
        "status": True,
    }


@pytest.mark.runtime.medium
def test_hip_mixbench(target_executor, ld_path: dict, mixbench_hip_binary: str):
    """Run the mixbench-hip benchmark and assert a well-formed CSV result.

    The session-scoped ``mixbench_hip_binary`` fixture clones and builds the
    benchmark; this test executes the resulting binary on an AMD GPU and parses
    its CSV throughput section, asserting the parsed status is valid.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {mixbench_hip_binary}")
    assert result.ok, (
        f"mixbench-hip failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )

    parsed = _parse_mixbench_output(result.stdout)
    assert parsed["status"] is True, f"mixbench-hip produced an invalid CSV result section:\n{result.stdout[:2000]}"
