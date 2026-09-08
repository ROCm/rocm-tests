# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS IET -- input EDPp test: drive a GEMM load and qualify the power response.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_iet.py`` and its parser
``logParser/tests/RVS/rvs_iet_parse.py``. IET runs unprivileged. It ramps a GEMM
workload towards ``target_power`` and reports one verdict per (action, GPU)
alongside a stream of power samples::

    [RESULT] [413362.955511] [action_1] [GPU::  9354] Power(W) 40.000000
    [RESULT] [413413.15384 ] [action_1] [GPU:: 25466] pass: TRUE

Both signals matter, for the same reason they do in TST: a power stress test that
reports ``pass: TRUE`` while never reading a usable wattage has qualified
nothing, so every reported sample must be positive.

Corrections against the original parser:

* The Multi-Chip Module special case is dropped as unsound rather than ported.
  The original set ``gpu_count = len(verdicts) / 2`` on MCM hosts and then
  required ``gpu_count == verdicts.count("TRUE")``. That equality cannot hold
  when every verdict is ``TRUE`` (``len/2 != len`` for any non-empty list), so a
  fully passing MCM host was reported as failing. Keying verdicts by
  ``(action, GPU)`` and requiring each to be ``TRUE`` needs no die-count
  correction and is right on both single-die and MCM parts.
* Repeated verdicts for one ``(action, GPU)`` are combined with AND, so a GPU
  that reports both a pass and a fail counts as failing instead of depending on
  which line came last.

The shipped MI210 config runs five 50 s actions, two of them serial across all
GPUs, so expect roughly 16 minutes on an 8-GPU host.
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

_CONF_NAME = "iet_single.conf"
_LABEL = "IET"
# Two of the five shipped actions are serial (parallel: false), so wall time
# scales with GPU count: ~50 s per GPU each, plus three parallel 50 s actions.
_RVS_TIMEOUT = 3600.0

# ``[RESULT] [ ts ] [<action>] [GPU:: <id>] pass: TRUE|FALSE``. Case-sensitive on
# the verdict token so prose containing "pass: true" cannot be mistaken for one.
_VERDICT_RE = re.compile(r"\[\s*RESULT\s*\].*\[([^\]]+)\]\s*\[GPU::\s*(\d+)\]\s*pass:\s*(TRUE|FALSE)\b")
_POWER_RE = re.compile(r"\[([^\]]+)\]\s*\[GPU::\s*(\d+)\]\s*Power\(W\)\s*([\d.]+)")


def _parse_verdicts(text: str) -> dict[tuple, bool]:
    """Return ``{(action, gpu_id): passed}``, combining repeats with AND."""
    verdicts: dict[tuple, bool] = {}
    for line in text.splitlines():
        if match := _VERDICT_RE.search(line):
            key = (match.group(1), match.group(2))
            verdicts[key] = verdicts.get(key, True) and match.group(3) == "TRUE"
    return verdicts


def _parse_power(text: str) -> list[tuple]:
    """Return ``(action, gpu_id, watts)`` for every power sample."""
    samples = []
    for line in text.splitlines():
        if match := _POWER_RE.search(line):
            samples.append((match.group(1), match.group(2), float(match.group(3))))
    return samples


def _assert_power_usable(samples: list[tuple]) -> None:
    """Every reported wattage must be positive, so a blind pass cannot slip through."""
    assert samples, (
        "RVS IET reported no power samples; a power stress test that never read a "
        "wattage cannot be considered to have qualified anything."
    )
    bad = [f"[{action}] GPU {gpu} Power(W)={watts}" for action, gpu, watts in samples if watts <= 0]
    assert not bad, "{} of {} power sample(s) were not positive:\n{}".format(
        len(bad), len(samples), "\n".join(bad[:10])
    )


def _assert_all_passed(verdicts: dict[tuple, bool]) -> None:
    """Fail with the specific (action, GPU) pairs that reported ``pass: FALSE``."""
    failed = sorted(f"[{action}] GPU {gpu}" for (action, gpu), passed in verdicts.items() if not passed)
    per_action = collections.Counter(action for action, _ in verdicts)
    assert not failed, "{} of {} IET verdict(s) reported 'pass: FALSE' (per-action totals {}):\n{}".format(
        len(failed), len(verdicts), dict(per_action), "\n".join(failed[:20])
    )


def _report_iet_metrics(verdicts: dict[tuple, bool], samples: list[tuple]) -> None:
    """Publish verdict and power coverage so a shrinking run surfaces in reports."""
    values = [watts for _, _, watts in samples]
    covered = {gpu for _, gpu in verdicts}
    report_metric("RVS_IET_VERDICTS", float(len(verdicts)))
    report_metric("RVS_IET_GPUS_COVERED", float(len(covered)))
    report_metric("RVS_IET_POWER_SAMPLES", float(len(samples)))
    if values:
        report_metric("RVS_IET_POWER_MAX_W", max(values), "W")
        report_metric("RVS_IET_POWER_MIN_W", min(values), "W")
    logger.info(
        "IET: %d verdicts across %d GPUs; %d power samples (%.1f-%.1f W)",
        len(verdicts),
        len(covered),
        len(samples),
        min(values) if values else 0.0,
        max(values) if values else 0.0,
    )


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.soak
def test_rvs_iet(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every IET action must pass on every GPU, backed by usable power readings."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))

    with step(f"Run RVS IET ({_CONF_NAME})"):
        output, exit_code = run_rvs(target_executor, rvs_env, binary, conf_path, _RVS_TIMEOUT, _LABEL)

    with step("Qualify verdicts and power samples"):
        assert_log_sane(output, exit_code, _LABEL)

        verdicts = _parse_verdicts(output)
        # Without this an empty or unparseable log would assert vacuously.
        assert verdicts, (
            f"No IET verdicts found; expected '[<action>] [GPU:: <id>] pass: TRUE|FALSE' "
            f"lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        samples = _parse_power(output)
        _assert_power_usable(samples)
        _report_iet_metrics(verdicts, samples)

    assert_summary_passed(parse_summary(output), declared_actions(target_executor, conf_path), _LABEL, conf)
    _assert_all_passed(verdicts)
