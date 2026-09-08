# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS BABEL -- BabelStream memory bandwidth qualification across all GPUs.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_babel.py`` and its parser
``logParser/tests/RVS/rvs_babel_parse.py``. BABEL runs unprivileged. Unlike the
other RVS modules it emits no per-GPU ``pass:`` verdict; it launches the
BabelStream kernels and prints one throughput table covering every GPU::

    [RESULT] [413001.791587] [babel-256MiB] [GPU:: 9354] Starting the Babel memory stress test
    ...
    GPU Id      Function    MiBytes/sec    Max MiB/s      Min MiB/s      Avg MiB/s
    46764       Copy        789869.309     789869.309     671527.371     724259.937

The qualification is therefore structural: every GPU that started must report a
row for each enabled kernel, at a positive throughput. A missing row means the
kernel never completed on that GPU.

Corrections against the original parser:

* Passing actions are no longer discarded when a different action fails. The
  original built its result dict inside ``if failures: ... else: ...``, so as
  soon as any GPU was short a kernel the dict held *only* the failing actions and
  every passing one vanished from the report.
* An action that produced no table at all is a failure rather than a pass. The
  original's ``else`` branch marked every announced action ``TRUE`` whenever the
  failure list was empty, which includes the case where no rows were parsed.
* The expected kernel set is read from the config's ``copy``/``mul``/``add``/
  ``triad``/``dot`` flags instead of being hard-coded to all five, so a config
  that disables a kernel is not failed for the missing row.

Runtime is about 80 s for the shipped MI210 config on 8 GPUs.
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

_CONF_NAME = "babel.conf"
_LABEL = "BABEL"
_RVS_TIMEOUT = 1800.0

_ALL_OPS = ("Copy", "Mul", "Add", "Triad", "Dot")

_ACTION_NAME_RE = re.compile(r"Action name\s*:\s*(\S+)")
_START_RE = re.compile(r"\[([^\]]+)\]\s*\[GPU::\s*(\d+)\]\s*Starting the Babel memory stress test")
# ``<gpu id>  <Function>  <MiBytes/sec>  <Max>  <Min>  <Avg>``. Anchored on the
# known kernel names so no other numeric line in the log can be read as a row.
_ROW_RE = re.compile(
    rf"^\s*(\d+)\s+({'|'.join(_ALL_OPS)})\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
    re.MULTILINE,
)


def _expected_ops(conf_text: str) -> set[str]:
    """Return the kernels the config enables.

    A kernel counts as enabled unless explicitly set ``false``, which matches
    RVS's own defaulting and avoids demanding rows for a disabled kernel.
    """
    enabled = set(_ALL_OPS)
    for op in _ALL_OPS:
        if re.search(rf"^\s*{op.lower()}\s*:\s*false\b", conf_text, re.MULTILINE | re.IGNORECASE):
            enabled.discard(op)
    return enabled


def _parse_started(text: str) -> dict[str, set[str]]:
    """Return ``{action: {gpu_id}}`` for the GPUs that entered the stress test."""
    started: dict[str, set[str]] = collections.defaultdict(set)
    for line in text.splitlines():
        if match := _START_RE.search(line):
            started[match.group(1)].add(match.group(2))
    return dict(started)


def _parse_rows(text: str) -> dict[str, dict[str, dict[str, float]]]:
    """Return ``{action: {gpu_id: {op: mib_per_sec}}}``.

    Rows carry no action name, so they are attributed to the most recently
    announced action -- the order RVS prints them in.
    """
    rows: dict[str, dict[str, dict[str, float]]] = collections.defaultdict(lambda: collections.defaultdict(dict))
    current = ""
    for line in text.splitlines():
        if match := _ACTION_NAME_RE.search(line):
            current = match.group(1)
        elif (match := _ROW_RE.search(line)) and current:
            rows[current][match.group(1)][match.group(2)] = float(match.group(3))
    return {action: dict(gpus) for action, gpus in rows.items()}


def _assert_rows_complete(
    rows: dict[str, dict[str, dict[str, float]]],
    started: dict[str, set[str]],
    expected: set[str],
) -> None:
    """Every started GPU must report every enabled kernel at a positive rate."""
    problems = []
    for action, gpus in sorted(started.items()):
        measured = rows.get(action, {})
        for gpu in sorted(gpus):
            ops = measured.get(gpu, {})
            if missing := sorted(expected - set(ops)):
                problems.append(f"[{action}] GPU {gpu} missing kernel(s): {', '.join(missing)}")
            problems.extend(
                f"[{action}] GPU {gpu} {op} reported {rate} MiBytes/sec"
                for op, rate in sorted(ops.items())
                if rate <= 0
            )
    assert not problems, "RVS BABEL results are incomplete:\n{}".format("\n".join(problems[:20]))


def _report_babel_metrics(rows: dict[str, dict[str, dict[str, float]]], expected: set[str]) -> None:
    """Publish throughput coverage and range so a shrinking run surfaces in reports."""
    rates = [rate for gpus in rows.values() for ops in gpus.values() for rate in ops.values()]
    covered = {gpu for gpus in rows.values() for gpu in gpus}
    report_metric("RVS_BABEL_KERNEL_RESULTS", float(len(rates)))
    report_metric("RVS_BABEL_GPUS_COVERED", float(len(covered)))
    if rates:
        report_metric("RVS_BABEL_RATE_MAX_MIBPS", max(rates), "MiB/s")
        report_metric("RVS_BABEL_RATE_MIN_MIBPS", min(rates), "MiB/s")
    logger.info(
        "BABEL: %d kernel result(s) across %d GPU(s), kernels %s (%.0f-%.0f MiBytes/sec)",
        len(rates),
        len(covered),
        sorted(expected),
        min(rates) if rates else 0.0,
        max(rates) if rates else 0.0,
    )


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_rvs_babel(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every GPU must report every enabled BabelStream kernel at a positive rate."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))
    conf_text = target_executor.run(f"cat {conf_path}").stdout or ""

    with step(f"Run RVS BABEL ({_CONF_NAME})"):
        output, exit_code = run_rvs(target_executor, rvs_env, binary, conf_path, _RVS_TIMEOUT, _LABEL)

    with step("Qualify BabelStream throughput results"):
        assert_log_sane(output, exit_code, _LABEL)

        started = _parse_started(output)
        assert started, (
            f"No GPU entered the BABEL stress test; expected 'Starting the Babel "
            f"memory stress test' lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        rows = _parse_rows(output)
        # Without this an empty or unparseable table would assert vacuously.
        assert rows, (
            f"No BABEL throughput rows found; expected '<gpu> <Copy|Mul|Add|Triad|Dot> "
            f"<rate> ...' rows from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        expected = _expected_ops(conf_text)
        _report_babel_metrics(rows, expected)
        _assert_rows_complete(rows, started, expected)

    assert_summary_passed(parse_summary(output), declared_actions(target_executor, conf_path), _LABEL, conf)
