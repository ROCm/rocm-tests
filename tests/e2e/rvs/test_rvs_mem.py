# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS MEM -- GPU memory qualification: run the memtest suite on every GPU.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_mem.py`` and its parser
``logParser/tests/RVS/rvs_mem_parse.py``. MEM runs unprivileged. It walks a set
of numbered memory tests (walking-1-bit, own-address, moving inversions, block
move, random patterns) on each GPU in parallel and reports one verdict per test
per GPU::

    [INFO  ] [412959.703844] [action_1] mem 9354  Starting the Memory stress test
    [RESULT] [412960.246988] [action_1] mem Test 1 : PASS

The verdict lines carry no GPU id, so a test that passed on seven of eight GPUs
looks exactly like one that only ran on seven -- both just yield fewer ``PASS``
lines. The participating GPUs are therefore taken from the ``Starting the Memory
stress test`` lines, and every numbered test must report one pass per GPU.

Corrections against the original parser:

* ``FAIL`` verdicts are recognised. The original matched only ``: PASS``, and its
  final-status scan looked for ``FALSE``/``ABORT``/``ERROR`` -- none of which
  occur in a ``mem Test 4 : FAIL`` line. A real memory failure was dropped from
  the results and left the run green.
* Verdicts are read for whichever actions the config declares. The original
  hard-coded ``[action_1]`` into its regex, making any further action invisible.
* Per-GPU coverage is enforced. The original collapsed all GPUs into a single key
  per test number and never checked how many GPUs reported in.

Runtime is about 40 s for the shipped config on 8 GPUs.
"""

from __future__ import annotations

import collections
import logging
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric, step
from tests.e2e.rvs._rvs_log import (
    assert_log_sane,
    assert_summary_passed,
    declared_actions,
    parse_summary,
    run_rvs,
)

logger = logging.getLogger(__name__)

_CONF_NAME = "mem.conf"
_LABEL = "MEM"
# num_passes/num_iter scale the walk, and the serial actions scale with GPU count.
_RVS_TIMEOUT = 1800.0

# ``[INFO  ] [ ts ] [<action>] mem <gpu> Starting the Memory stress test``
_START_RE = re.compile(r"\[([^\]]+)\]\s*mem\s+(\d+)\s+Starting the Memory stress test")
# ``[RESULT] [ ts ] [<action>] mem Test <n> : PASS|FAIL``. Requiring the verdict
# token keeps the descriptive "mem Test 1: Change one bit memory addresss" lines,
# which share the prefix, out of the results.
_VERDICT_RE = re.compile(r"\[\s*RESULT\s*\].*\[([^\]]+)\]\s*mem\s+Test\s+(\d+)\s*:\s*(PASS|FAIL)\b")


def _parse_gpus_per_action(text: str) -> dict[str, set[str]]:
    """Return ``{action: {gpu_id}}`` for the GPUs that entered the stress test."""
    gpus: dict[str, set[str]] = collections.defaultdict(set)
    for line in text.splitlines():
        if match := _START_RE.search(line):
            gpus[match.group(1)].add(match.group(2))
    return dict(gpus)


def _parse_verdicts(text: str) -> dict[tuple, collections.Counter]:
    """Return ``{(action, test_number): Counter({PASS: n, FAIL: m})}``."""
    verdicts: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in text.splitlines():
        if match := _VERDICT_RE.search(line):
            verdicts[(match.group(1), int(match.group(2)))][match.group(3)] += 1
    return dict(verdicts)


def _assert_no_failures(verdicts: dict[tuple, collections.Counter]) -> None:
    """Any ``FAIL`` verdict is a memory defect and fails the run outright."""
    failed = sorted(
        f"[{action}] Test {number}: {counts['FAIL']} FAIL"
        for (action, number), counts in verdicts.items()
        if counts["FAIL"]
    )
    assert not failed, "RVS MEM reported failing memory test(s):\n{}".format("\n".join(failed))


def _assert_full_coverage(verdicts: dict[tuple, collections.Counter], gpus: dict[str, set[str]]) -> None:
    """Every numbered test must have passed once per GPU that started the action."""
    short = []
    for (action, number), counts in sorted(verdicts.items()):
        expected = len(gpus.get(action, ()))
        if expected and counts["PASS"] < expected:
            short.append(f"[{action}] Test {number}: {counts['PASS']} pass(es) for {expected} GPU(s)")
    assert not short, (
        "RVS MEM did not pass on every GPU; a memory test that silently covered "
        "fewer GPUs qualifies less than it appears to:\n{}".format("\n".join(short))
    )


def _report_mem_metrics(verdicts: dict[tuple, collections.Counter], gpus: dict[str, set[str]]) -> None:
    """Publish test and GPU coverage so a shrinking run shows up in reports."""
    passes = sum(counts["PASS"] for counts in verdicts.values())
    covered = set().union(*gpus.values()) if gpus else set()
    report_metric("RVS_MEM_TESTS_CHECKED", float(len(verdicts)))
    report_metric("RVS_MEM_PASS_VERDICTS", float(passes))
    report_metric("RVS_MEM_GPUS_COVERED", float(len(covered)))
    logger.info(
        "MEM: %d numbered test(s) across %d GPU(s), %d pass verdict(s); per-action GPUs: %s",
        len(verdicts),
        len(covered),
        passes,
        {action: len(ids) for action, ids in sorted(gpus.items())},
    )


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_rvs_mem(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every MEM test must pass on every participating GPU."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))

    with step(f"Run RVS MEM ({_CONF_NAME})"):
        output, exit_code = run_rvs(target_executor, rvs_env, binary, conf_path, _RVS_TIMEOUT, _LABEL)

    with step("Qualify memory test verdicts"):
        assert_log_sane(output, exit_code, _LABEL)

        gpus = _parse_gpus_per_action(output)
        assert gpus, (
            f"No GPU entered the MEM stress test; expected 'Starting the Memory "
            f"stress test' lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        verdicts = _parse_verdicts(output)
        # Without this an empty or unparseable log would assert vacuously.
        assert verdicts, (
            f"No MEM verdicts found; expected '[<action>] mem Test <n> : PASS|FAIL' "
            f"lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        _report_mem_metrics(verdicts, gpus)
        _assert_no_failures(verdicts)
        _assert_full_coverage(verdicts, gpus)

    assert_summary_passed(parse_summary(output), declared_actions(target_executor, conf_path), _LABEL, conf)
