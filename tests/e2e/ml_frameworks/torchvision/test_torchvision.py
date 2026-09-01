# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TorchVision P1 image-transform correctness UT suite.

Runs the cuda-tagged tensor UT suites in a container against torchvision ops built
once per session; the JUnit report is parsed so the assertion names any failing case.
"""

import logging
import os

import pytest

from tests.e2e.ml_frameworks.torchvision._constants import (
    CONTAINER_MOUNT_FLAGS,
    GPU_COUNT_ARG,
    PYTEST_SELECTOR,
    RUN_TIMEOUT,
    TEST_FILES,
)
from tests.e2e.ml_frameworks.torchvision._result_parser import parse_junit_xml

logger = logging.getLogger(__name__)

# Sentinels bracketing the JUnit XML report catted onto stdout after the run.
_JUNIT_START = "__TV_JUNIT_START__"
_JUNIT_END = "__TV_JUNIT_END__"

# Hard-crash signatures: a match means the run aborted mid-way (the process died
# before writing its report), so the run is a failure regardless of parsed cases.
_CRASH_MARKERS = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal Python error",
)


def _extract_junit(text: str) -> str:
    """Return the JUnit XML report bracketed by the sentinels in *text*, or ``""``."""
    start = text.find(_JUNIT_START)
    end = text.find(_JUNIT_END)
    if start == -1 or end == -1 or end < start:
        return ""
    return text[start + len(_JUNIT_START) : end].strip()


@pytest.mark.container(extra_run_flags=CONTAINER_MOUNT_FLAGS)
@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.runtime.medium
@pytest.mark.parametrize("test_file", TEST_FILES, ids=lambda f: os.path.basename(f)[len("test_") : -len(".py")])
def test_torchvision_p1_ut_suite(target_executor, torchvision_repo, test_file):
    """Run one cuda-tagged UT suite against the pre-built torchvision ops; assert it passes.

    The ops are built once per session by the ``torchvision_repo`` fixture.  Each
    parametrized call runs only the pytest collection and execution step, then
    parses the JUnit report so the assertion names any failing GPU case.
    """
    suite = os.path.basename(test_file)[len("test_") : -len(".py")]
    junit = f"junit_{suite}.xml"

    # The ops are already built by the session-scoped torchvision_repo fixture.
    # Each parametrized test case only runs pytest; no rebuild or git-clean here,
    # which prevents the second test from wiping the .so the first test built.
    cmd = "\n".join(
        (
            f"cd {torchvision_repo}",
            f"python -m pytest {test_file} -v -k {PYTEST_SELECTOR} --junitxml={junit} -p no:cacheprovider",
            "rc=$?",
            f"echo {_JUNIT_START}",
            f"cat {junit} 2>/dev/null",
            f"echo {_JUNIT_END}",
            "exit $rc",
        )
    )

    logger.info("TorchVision P1 UT suite starting: file=%s", test_file)
    result = target_executor.run(cmd, timeout=RUN_TIMEOUT)

    combined = f"{result.stdout}\n{result.stderr}"
    summary = parse_junit_xml(_extract_junit(combined))
    crash_markers = [m for m in _CRASH_MARKERS if m in combined]

    logger.info(
        "TorchVision UT results [%s]: passed=%d skipped=%d failed=%d errored=%d (exit=%s, crash_markers=%s)",
        test_file,
        summary.passed,
        summary.skipped,
        summary.failed,
        summary.errored,
        result.exit_code,
        crash_markers or "none",
    )

    # Crash markers take priority: if the process was killed by a signal, report
    # the crash explicitly rather than the secondary "no results" symptom.
    assert not crash_markers, (
        f"TorchVision UT runner crashed for {test_file} "
        f"(exit={result.exit_code}, crash_markers={crash_markers}):\n"
        f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-4000:]}"
    )

    # No parsed results means the ops build, nms import gate, or runner never
    # produced a report (failed to start).
    assert summary.total > 0, (
        f"TorchVision UT suite produced no test results for {test_file} (exit={result.exit_code}); "
        f"the ops build, nms import check, or runner likely failed to start:\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
    )

    # Clean run: no failed/errored cases and zero exit.
    completed_cleanly = summary.is_clean and result.exit_code == 0
    assert completed_cleanly, (
        f"TorchVision UT suite did not complete cleanly for {test_file} "
        f"(exit={result.exit_code}, failed={summary.failed}, errored={summary.errored}, "
        f"passed={summary.passed}, skipped={summary.skipped}):\n"
        f"failed: {summary.failed_names[:50]}\n"
        f"errored: {summary.errored_names[:50]}\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
