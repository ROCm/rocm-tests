# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
artifacts_plugin.py -- Allure artifact capture and AllureReporter fixture.

Responsibilities:
    - Provide the ``artifacts_fixture``: captures GPU state on test failure
      and attaches it to Allure.
    - Provide the ``allure_reporter`` fixture: injects an AllureReporter instance
      for step-level reporting inside test functions.
    - Attach GPU state dumps (amd-smi output) to Allure on FAIL automatically
      via the autouse ``_auto_capture_artifacts`` fixture.

The ``allure_reporter`` fixture gives tests a structured API for steps, metrics,
and attachments without importing allure directly.

Usage in tests:
    def test_hip_runtime(target_executor, allure_reporter):
        with allure_reporter.step("Compile kernel"):
            result = target_executor.run("hipcc kernel.cpp -o kernel")
        allure_reporter.attach(result.stdout, name="compilation_output")
        allure_reporter.metric("COMPILE_TIME_S", result.duration)
"""

from __future__ import annotations

import logging
import subprocess

import pytest

from framework.reporting.allure_reporter import AllureReporter, attach_text, step

logger = logging.getLogger(__name__)


def _capture_gpu_state(gpu_index: int | None = None) -> str:
    """Run amd-smi to capture current GPU state for Allure attachment.

    Args:
        gpu_index: Specific GPU to query, or None for all GPUs.

    Returns:
        amd-smi output as a string, or an error message if unavailable.
    """
    cmd = ["amd-smi", "metric"]
    if gpu_index is not None:
        cmd += ["--gpu", str(gpu_index)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        return result.stdout or result.stderr or "(no amd-smi output)"
    except FileNotFoundError:
        return "amd-smi not found — GPU state unavailable"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"GPU state capture failed: {exc}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def allure_reporter(request):
    """Inject an AllureReporter for step-level Allure reporting in tests.

    The reporter is bound to the current test's node ID so log messages and
    Allure steps are correctly attributed.

    Args:
        request: pytest request object providing test metadata.

    Returns:
        AllureReporter: Ready-to-use reporter with .step(), .attach(), .metric().
    """
    return AllureReporter(test_name=request.node.nodeid)


@pytest.fixture
def artifacts_fixture(request, _framework_config):
    """Capture GPU state and attach it to Allure on test failure.

    Yields control to the test. On teardown, if the test failed, captures
    GPU diagnostic output via amd-smi and attaches it to the Allure report.

    Args:
        request:          pytest request object (provides test outcome).
        framework_config: Session config for artifact directory path.

    Yields:
        None
    """
    yield
    # Post-test: attach GPU state on failure
    rep = getattr(request.node, "rep_call", None)
    if rep is None or not rep.failed:
        return

    with step("Capture GPU state (post-failure)"):
        state_dump = _capture_gpu_state()
        attach_text(state_dump, name="gpu_state_on_failure.txt")
        logger.info("Attached GPU state dump for failed test: %s", request.node.nodeid)


@pytest.fixture(autouse=True)
def _store_test_outcome():
    """Autouse: store the test call phase outcome on the node for artifacts_fixture.

    pytest does not expose test outcomes in teardown by default. This fixture
    hooks pytest_runtest_makereport to save the call phase result on the node.

    Yields:
        None
    """
    return
    # rep_call is set by the hookimpl below


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Store test phase reports on the item node for post-test fixture access."""
    outcome = yield
    rep = outcome.get_result()
    if call.when == "call":
        item.rep_call = rep
