# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmi_bm.py -- AMD SMI Library Benchmark Orchestration Suite.

Ported from: amd-smi-lib AMDSMI_BM class (lines 273-357).

Validates:
    - All 50+ amd-smi-lib API functions via the benchmark suite
    - GPU metrics: power, temperature, memory, clocks, throttle, ECC, PCIe
    - Power management: limits, overdrive, P-states, UBB
    - GPU reset and driver integration
    - Performance monitoring and workload tracking

Auto-injected by CATEGORY_PROFILES for tests/e2e/rocm_libs/:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.medium
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _dispatch_test_result(ret):
    """Evaluate return value to (passed: bool, message: str | None) tuple.

    Handles bool, Multi_Test_Status, or tuple return types matching the
    original AMDSMI_BM.execute() dispatch logic exactly.
    """
    if isinstance(ret, bool):
        return ret, None
    if isinstance(ret, tuple):
        status = ret[0] if ret else False
        message = ret[1] if len(ret) > 1 else None
        return status, message
    return bool(ret), None


def _call_test_function(test_name: str, amdsmi_testlist: list[str]):
    """Call a test function from globals with special-case parameter handling.

    Raises ValueError if function not found.
    """
    test_fn = globals().get(test_name)
    if test_fn is None:
        raise ValueError(f"Test function {test_name} not found in globals")

    if test_name in ("AMDSMI_monitor_qt_gpu_workload", "AMDSMI_monitor_qt_gpu_file_workload"):
        return test_fn(amd_smi_sleep_time=5)
    if test_name == "AMDSMI_node_power_management":
        return test_fn(len(amdsmi_testlist), len(amdsmi_testlist))
    return test_fn()


@pytest.mark.runtime.medium
def test_amdsmi_bm_suite(
    amdsmi_installed,
    amdsmi_testlist: list[str],
    amdsmi_app_version: str | None,
):
    """Execute the full amd-smi-lib benchmark suite orchestrator.

    Loops through all test functions in the suite, handles special-case
    parameters for specific tests (monitor_qt_gpu_workload variants,
    node_power_management), and tracks per-test execution time.

    Handles return types:
        - bool: True -> pass, False -> fail
        - Multi_Test_Status: pass-through status object
        - tuple: (status, optional_message)

    Fails the test if any subtest fails; reports individual results via
    logger.
    """
    del amdsmi_installed

    suite_start = time.time()
    all_passed = True
    failed_tests = []

    logger.info("Starting amd-smi-lib benchmark suite (app version: %s)", amdsmi_app_version)

    for test_name in set(amdsmi_testlist):
        logger.info("Running test: %s", test_name)
        subtest_start = time.time()

        try:
            ret = _call_test_function(test_name, amdsmi_testlist)
            test_passed, test_message = _dispatch_test_result(ret)

            if test_passed:
                msg = "" if not test_message else f" ({test_message})"
                logger.info("✓ %s: PASS%s", test_name, msg)
            else:
                msg = "" if not test_message else f" ({test_message})"
                logger.error("✗ %s: FAIL%s", test_name, msg)
                all_passed = False
                failed_tests.append((test_name, test_message or "Test failed"))

            logger.info("Subtest execution time: %.2f seconds", time.time() - subtest_start)

        except ValueError as exc:
            logger.error("Test function not found: %s", exc)
            all_passed = False
            failed_tests.append((test_name, str(exc)))
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Exception in %s: %s", test_name, exc)
            all_passed = False
            failed_tests.append((test_name, f"Exception: {exc}"))

    suite_elapsed = time.time() - suite_start

    logger.info("Suite execution time: %.2f seconds", suite_elapsed)
    if failed_tests:
        logger.error("Failed tests (%d):", len(failed_tests))
        for test_name, reason in failed_tests:
            logger.error("  - %s: %s", test_name, reason)

    assert all_passed, f"amd-smi-lib benchmark suite failed with {len(failed_tests)} test(s):\n" + "\n".join(
        f"  {name}: {reason}" for name, reason in failed_tests
    )
