# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmi_bm.py -- AMD SMI Library Benchmark Orchestration Suite.

Validates:
    - All amd-smi-lib API functions via the benchmark suite
    - GPU metrics: power, temperature, memory, clocks, throttle, ECC, PCIe
    - Power management: limits, overdrive, P-states, UBB
    - GPU reset and driver integration
    - Performance monitoring and workload tracking

Auto-injected by CATEGORY_PROFILES for tests/e2e/system_tools/amd_smi:
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


def _build_amd_smi_cmd(amd_smi: str, test_name: str) -> str:
    """Map test name to amd-smi CLI command.

    The AMDSMI_FULL_TESTS list maps 1:1 to amd-smi subcommands. Underscores
    in test names are converted to spaces for the CLI (e.g.
    "AMDSMI_metric_watch1" → "amd-smi metric watch1").

    Args:
        amd_smi: Path to the amd-smi executable.
        test_name: Test name from AMDSMI_FULL_TESTS.

    Returns:
        Full amd-smi CLI command string ready for target_executor.run().
    """
    parts = test_name.split("_")
    subcommand = " ".join(parts[1:]).lower()
    return f"{amd_smi} {subcommand}"


@pytest.mark.runtime.medium
def test_amdsmi_bm_suite(
    target_executor,
    amdsmi_installed,
    amdsmi_testlist: list[str],
    amdsmi_app_version: str | None,
    ld_path: dict,
):
    """Execute the full amd-smi-lib benchmark suite via direct amd-smi CLI invocations.

    Loops through all test subcommands in the suite, invokes each via
    target_executor.run(), and tracks per-test execution time and exit codes.

    Fails the test if any subcommand exits non-zero; reports individual
    results via logger.
    """
    del amdsmi_installed

    suite_start = time.time()
    all_passed = True
    failed_tests = []

    amd_smi = "amd-smi"
    ld = ld_path.get("LD_LIBRARY_PATH", "")
    env_prefix = f"env LD_LIBRARY_PATH={ld} " if ld else ""

    logger.info("Starting amd-smi-lib benchmark suite (app version: %s)", amdsmi_app_version)

    for test_name in set(amdsmi_testlist):
        logger.info("Running test: %s", test_name)
        subtest_start = time.time()

        try:
            cmd = _build_amd_smi_cmd(amd_smi, test_name)
            full_cmd = f"{env_prefix}{cmd}"
            result = target_executor.run(full_cmd)

            if result.ok:
                logger.info("✓ %s: PASS", test_name)
            else:
                logger.error("✗ %s: FAIL (exit_code=%d)", test_name, result.exit_code)
                all_passed = False
                failed_tests.append((test_name, f"exit_code={result.exit_code}"))

            logger.info("Subtest execution time: %.2f seconds", time.time() - subtest_start)

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
