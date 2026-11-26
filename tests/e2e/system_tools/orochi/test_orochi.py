# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Orochi unit test suite validation on Windows ROCm installations.

Orochi (https://github.com/GPUOpen-LibrariesAndSDKs/Orochi, MIT licensed) loads
the HIP and CUDA driver/runtime APIs dynamically at runtime. This test builds
Orochi against the ROCm install under test (see conftest.py) and runs its own
unit test suite, asserting that every test passes.

Markers:
    CATEGORY_PROFILES injects hw.gpu, ci.nightly, layer.runtime, e2e.stack,
    and os.windows. runtime.medium is declared on the test.
"""

from __future__ import annotations

import logging
import re

import pytest

logger = logging.getLogger(__name__)

# Matches Orochi's GTest-style summary line, e.g. "[  PASSED  ] 42 tests." or
# "[  FAILED  ] 1 test."
_RESULT_PATTERN = re.compile(r"\[\s*([A-Z]+)\s*\]\s*(\d+)\s*tests?[.,]")


@pytest.mark.runtime.medium
def test_orochi_unit_tests(target_executor, orochi_scripts_dir: str) -> None:
    """Run Orochi's unit test suite and assert that all tests pass."""
    result = target_executor.run(f'cd /d "{orochi_scripts_dir}" && unittest.bat', timeout=600.0)
    assert result.ok, (
        f"Orochi unittest.bat failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
    )

    match = _RESULT_PATTERN.search(result.stdout)
    assert match, (
        "Orochi: could not find a '[ PASSED/FAILED ] N tests.' summary line in output:\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
    )

    status, count = match.group(1).upper(), int(match.group(2))
    assert status == "PASSED", f"Orochi: {count} test(s) reported as {status}"
    assert count > 0, "Orochi: the unit-test runner reported success but ran zero tests"
    logger.info("Orochi: %s tests passed", count)
