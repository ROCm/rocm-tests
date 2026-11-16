# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RDC diagnostic run against an unauthenticated daemon.

``rdci diag -u -r 2 -g <group>`` asks rdcd to run its run-level-2 battery over
a GPU group and prints a verdict per check::

    LOG: DiagnosticRun started
    LOG: Test 1 / 5
    ...
    LOG: DiagnosticRun finished
    No compute process:       Pass
    Node topology check:      Pass
    GPU parameters check:     Pass
    Compute Queue ready:      Pass
    System memory check:      Pass

A run is accepted when four things hold: the command exited successfully, its
output carries no crash marker, it reached ``DiagnosticRun finished``, and no
check reported ``Fail``. RDC has four verdicts -- Pass, Skip, Warn and Fail --
and only Fail marks a regression: Skip means the check does not apply to this
hardware configuration, which can happen legitimately on qualifying hardware.

The exit-code and crash-marker checks matter on their own. A diagnostic that
printed its summary and then died on the way out still reaches
``DiagnosticRun finished`` with every check reporting ``Pass``, so the output
alone would call that run healthy.

Both tests drive the same command. RDC used to report RVS module sections in
this output and the two differed in which sections each required; builds since
report the five named checks above instead, so the two now cover the same
ground under their original names.

Each test reserves the whole node. rdcd is a node-wide singleton that is
stopped and restarted here, which would disturb anything else talking to RDC,
and the run is held to the node's full GPU inventory.
"""

from __future__ import annotations

import logging
import re
import shlex

import pytest

from framework.reporting.allure_reporter import attach_text, step

logger = logging.getLogger(__name__)

# Run level 2 is documented as "tests take a few minutes"; level 1 is seconds
# and level 3 is half an hour or more.
_RUN_LEVEL = 2
_DIAG_TIMEOUT = 900.0

_FINISHED_RE = re.compile(r"DiagnosticRun\s+finished")
# Verdicts print as "<check name>:<padding><verdict>".
_FAIL_RE = re.compile(r":\s+Fail\b")
# Signs that the tool died rather than reported, matched case-insensitively
# because they reach the log from the shell, libc and the binary alike.
_CRASH_RE = re.compile(
    r"core dumped|Aborted|Segmentation fault|No such file or directory|EXIT_CODE=134",
    re.IGNORECASE,
)


def _run_diag(executor, rdci: str, group_id: str) -> tuple[str, int]:
    """Run the diagnostic over *group_id* and return its output and exit code."""
    command = f"sudo -n {shlex.quote(rdci)} diag -u -r {_RUN_LEVEL} -g {group_id}"
    logger.info("Running RDC diagnostic: %s", command)
    result = executor.run(command, timeout=_DIAG_TIMEOUT)
    return (result.stdout or "") + (result.stderr or ""), result.exit_code


def _assert_diag_passed(output: str, exit_code: int, label: str) -> None:
    """Require the diagnostic to have run cleanly and no check to have failed.

    The tool-level checks come first: once the binary has crashed, a missing
    ``DiagnosticRun finished`` is a consequence rather than the fault worth
    reporting.
    """
    assert exit_code == 0, f"{label}: rdci diag exited {exit_code}:\n{output[-2000:]}"

    crash = _CRASH_RE.search(output)
    assert crash is None, f"{label}: rdci diag output carries the crash marker {crash.group()!r}:\n{output[-2000:]}"

    assert _FINISHED_RE.search(output), (
        f"{label}: the diagnostic never reached 'DiagnosticRun finished':\n{output[-2000:]}"
    )

    failed = [line.strip() for line in output.splitlines() if _FAIL_RE.search(line)]
    assert not failed, "{}: the diagnostic reported a failing check:\n{}".format(label, "\n".join(failed))

    logger.info("%s: diagnostic completed with no failing check", label)


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_rdc_rvs_memspeed(target_executor, rdc_binaries, rdc_group):
    """The run-level-2 diagnostic completes with no check reporting Fail."""
    label = "RDC_RVS_MemSpeed"
    group_id = rdc_group("test")

    with step(f"Run {label}: rdci diag -u -r {_RUN_LEVEL}"):
        output, exit_code = _run_diag(target_executor, rdc_binaries.rdci, group_id)

    with step("Qualify diagnostic verdicts"):
        attach_text(output, name=f"{label} diagnostic output")
        _assert_diag_passed(output, exit_code, label)


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_rdc_rvs_bwtest(target_executor, rdc_binaries, rdc_group):
    """The run-level-2 diagnostic completes with no check reporting Fail."""
    label = "RDC_RVS_BWTest"
    group_id = rdc_group("test")

    with step(f"Run {label}: rdci diag -u -r {_RUN_LEVEL}"):
        output, exit_code = _run_diag(target_executor, rdc_binaries.rdci, group_id)

    with step("Qualify diagnostic verdicts"):
        attach_text(output, name=f"{label} diagnostic output")
        _assert_diag_passed(output, exit_code, label)
