# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apex fused-kernel L0 unit-test suite validation.

Runs the Apex L0 ``unittest`` suite through ``target_executor`` in a container and
asserts every sub-test passes; the ``apex_repo`` fixture provides the checkout.
"""

import logging
import sys

import pytest

from tests.e2e.ml_frameworks.apex._constants import (
    APEX_NUM_GPUS,
    CONTAINER_MOUNT_FLAGS,
    GPU_COUNT_ARG,
    L0_SUBDIR,
    RUN_SCRIPT,
    RUN_TIMEOUT,
)
from tests.e2e.ml_frameworks.apex._result_parser import parse_unittest_output

logger = logging.getLogger(__name__)

# Hard-crash signatures: a match means the runner aborted mid-run, so the suite
# fails regardless of how many sub-tests parsed as passing.
_CRASH_MARKERS = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal Python error",
)


def _print_summary(summary, exit_code, crash_markers):
    """Print a formatted result table to stdout after the Apex L0 suite runs."""
    status = "PASSED" if summary.is_clean and exit_code == 0 and not crash_markers else "FAILED"
    lines = [
        "",
        "=" * 60,
        "  Apex L0 Suite Results",
        "=" * 60,
        f"  Status    : {status}",
        f"  Passed    : {summary.passed}",
        f"  Skipped   : {summary.skipped}",
        f"  Failed    : {summary.failed}",
        f"  Errored   : {summary.errored}",
        f"  Unresolved: {len(summary.unresolved_names)}",
        f"  Total run : {summary.ran_total}",
        f"  Exit code : {exit_code}",
        f"  Crashes   : {', '.join(crash_markers) if crash_markers else 'none'}",
        "=" * 60,
        "",
    ]
    print("\n".join(lines), file=sys.stdout, flush=True)


@pytest.mark.container(ipc="host", extra_run_flags=CONTAINER_MOUNT_FLAGS)
@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.runtime.soak
def test_apex_l0_suite(target_executor, apex_repo):
    """Run the Apex L0 suite in the container and assert it completes cleanly.

    Compiles the fused kernels on first run, drives the ``unittest`` suite, and
    parses the verbose output per sub-test so the assertion names any failure.
    """
    cmd = f"cd {apex_repo}/{L0_SUBDIR} && bash ./{RUN_SCRIPT}"

    gpu_label = "all" if APEX_NUM_GPUS is None else APEX_NUM_GPUS
    logger.info("Apex L0 suite starting (num_gpus=%s)", gpu_label)
    result = target_executor.run(cmd, timeout=RUN_TIMEOUT)

    # unittest writes its verbose result lines to stderr; parse both streams.
    combined = f"{result.stdout}\n{result.stderr}"
    summary = parse_unittest_output(combined)
    crash_markers = [m for m in _CRASH_MARKERS if m in combined]

    logger.info(
        "Apex L0 results: passed=%d skipped=%d failed=%d errored=%d unresolved=%d "
        "(ran_total=%d, exit=%s, crash_markers=%s)",
        summary.passed,
        summary.skipped,
        summary.failed,
        summary.errored,
        len(summary.unresolved_names),
        summary.ran_total,
        result.exit_code,
        crash_markers or "none",
    )

    # Print a human-readable summary table to stdout so it appears in the
    # pytest live log regardless of log level configuration.
    _print_summary(summary, result.exit_code, crash_markers)

    # No parsed results means the kernel build or runner never produced sub-tests.
    assert summary.total > 0 or summary.ran_total > 0, (
        f"Apex L0 suite produced no test results (exit={result.exit_code}); "
        f"the kernel build or runner likely failed to start:\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-4000:]}"
    )

    # Clean run: no failed/errored/unresolved sub-tests, zero exit, no crash. The
    # exit code and crash markers are backstops -- a mid-test fault can leave the
    # last sub-test without an outcome and must never be reported as a pass.
    completed_cleanly = summary.is_clean and result.exit_code == 0 and not crash_markers
    assert completed_cleanly, (
        f"Apex L0 suite did not complete cleanly "
        f"(exit={result.exit_code}, crash_markers={crash_markers or 'none'}, "
        f"failed={summary.failed}, errored={summary.errored}, "
        f"unresolved={len(summary.unresolved_names)}, "
        f"passed={summary.passed}, skipped={summary.skipped}):\n"
        f"failed: {summary.failed_names[:50]}\n"
        f"errored: {summary.errored_names[:50]}\n"
        f"unresolved (crashed mid-test): {summary.unresolved_names[:50]}\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-3000:]}"
    )
