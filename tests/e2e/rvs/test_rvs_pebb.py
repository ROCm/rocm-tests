# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS PEBB -- PCIe bandwidth qualification between every CPU and every GPU.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_pebb.py`` and its parser
``logParser/tests/RVS/rvs_pebb_parse.py``. PEBB runs unprivileged. It walks each
CPU/GPU combination and reports the measured host-to-device and device-to-host
throughput::

    [RESULT] [413133.743591] [h2d-sequential-51MB] pcie-bandwidth [ 1/16] [CPU:: 0] \
[GPU:: 2 - 9354 - 0000:63:00.0] h2d::true d2h::true 14.254 GBps duration: 0.079026 secs

The ``h2d::``/``d2h::`` tokens echo the action's configured directions rather than
reporting a result: the shipped ``h2d-b2b-51MB`` action sets
``device_to_host: false`` and duly logs ``d2h::false`` on every line. The
qualification signal is the bandwidth figure, so that is what this test checks.

Corrections against the original parser:

* Direction flags are no longer read as verdicts. The original recorded a pass for
  ``(true, true)``, ``(false, true)`` and ``(true, false)``, which means it passed
  an action whenever at least one direction was enabled -- a property of the
  config file, true before the test ran. Bandwidth values are validated instead.
* Errors are not silently swallowed. The original wrapped its verdict loop in a
  bare ``except:`` that discarded every exception, so a parsing failure produced
  an empty result set rather than a failure.

The enumeration is complete even when a measurement is skipped -- entries appear
as ``(not measured)`` -- so full GPU coverage per action is enforced, while the
per-sample check applies only to values actually reported.

Runtime is about 2.5 minutes for the shipped MI210 config on 8 GPUs.
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

_CONF_NAME = "pebb_single.conf"
_LABEL = "PEBB"
_RVS_TIMEOUT = 1800.0

# ``[RESULT] [ ts ] [<action>] pcie-bandwidth [ n/m] [CPU:: c] [GPU:: idx - id - bdf]
#  h2d::x d2h::y <gbps|(not measured)> duration: ... secs``
#
# Only ``[RESULT]`` lines are read. RVS also emits interim ``[INFO  ]`` progress
# lines for the same CPU/GPU pairs, and on this config 1041 of them carry
# ``-nan GBps`` while the running average is still settling. Those are not
# results, and folding them in would either fail a healthy run or -- worse -- slip
# through, since NaN compares false against any threshold.
_SAMPLE_RE = re.compile(
    r"\[\s*RESULT\s*\].*\[([^\]]+)\]\s*pcie-bandwidth\s*\[[^\]]*\]\s*\[CPU::\s*(\d+)\]\s*"
    r"\[GPU::\s*\d+\s*-\s*(\d+)\s*-\s*[^\]]+\]\s*"
    r"h2d::(?:true|false)\s+d2h::(?:true|false)\s+(\(not measured\)|\(pending\)|-?nan|-?inf|[\d.]+)"
)


def _parse_samples(text: str) -> tuple[list[tuple], dict[str, set[str]], int, list[str]]:
    """Return samples, ``{action: {gpu_id}}`` enumerated, skipped count and bad readings."""
    samples: list[tuple] = []
    enumerated: dict[str, set[str]] = collections.defaultdict(set)
    skipped = 0
    invalid: list[str] = []
    for line in text.splitlines():
        match = _SAMPLE_RE.search(line)
        if not match:
            continue
        action, cpu, gpu, value = match.groups()
        enumerated[action].add(gpu)
        if value.startswith("("):
            skipped += 1
        elif value.lstrip("-") in ("nan", "inf"):
            invalid.append(f"[{action}] CPU {cpu} -> GPU {gpu}: {value} GBps")
        else:
            samples.append((action, cpu, gpu, float(value)))
    return samples, dict(enumerated), skipped, invalid


def _assert_bandwidth_usable(samples: list[tuple], skipped: int, invalid: list[str]) -> None:
    """Reported bandwidths must be finite and positive, and something must be measured."""
    # NaN compares false against every threshold, so it has to be rejected by name
    # rather than left to the positivity check below.
    assert not invalid, "RVS PEBB reported {} non-finite PCIe bandwidth value(s) on result lines:\n{}".format(
        len(invalid), "\n".join(invalid[:10])
    )
    assert samples, (
        f"RVS PEBB measured no PCIe bandwidth at all ({skipped} sample(s) reported "
        f"'(not measured)'); the link qualification produced no throughput evidence."
    )
    bad = [f"[{action}] CPU {cpu} -> GPU {gpu}: {gbps} GBps" for action, cpu, gpu, gbps in samples if gbps <= 0]
    assert not bad, "{} of {} PCIe bandwidth sample(s) were not positive:\n{}".format(
        len(bad), len(samples), "\n".join(bad[:10])
    )


def _assert_coverage(samples: list[tuple], enumerated: dict[str, set[str]], declared: set[str]) -> None:
    """Every declared action must enumerate all GPUs and measure at least one."""
    all_gpus = set().union(*enumerated.values()) if enumerated else set()
    measured = {action for action, _, _, _ in samples}
    problems = []
    for action in sorted(declared):
        if action not in enumerated:
            problems.append(f"[{action}] produced no PCIe bandwidth entries")
            continue
        if gaps := sorted(all_gpus - enumerated[action]):
            problems.append(f"[{action}] never enumerated GPU(s): {', '.join(gaps)}")
        if action not in measured:
            problems.append(f"[{action}] enumerated GPUs but measured no bandwidth on any of them")
    assert not problems, "RVS PEBB coverage is incomplete:\n{}".format("\n".join(problems))


def _report_pebb_metrics(samples: list[tuple], enumerated: dict[str, set[str]], skipped: int) -> None:
    """Publish bandwidth coverage and range so a regression shows up in reports."""
    rates = [gbps for _, _, _, gbps in samples]
    covered = set().union(*enumerated.values()) if enumerated else set()
    report_metric("RVS_PEBB_BANDWIDTH_SAMPLES", float(len(samples)))
    report_metric("RVS_PEBB_GPUS_COVERED", float(len(covered)))
    if rates:
        report_metric("RVS_PEBB_BANDWIDTH_MAX_GBPS", max(rates), "GBps")
        report_metric("RVS_PEBB_BANDWIDTH_MIN_GBPS", min(rates), "GBps")
    logger.info(
        "PEBB: %d bandwidth sample(s) across %d action(s) and %d GPU(s) (%.2f-%.2f GBps), %d not measured",
        len(samples),
        len(enumerated),
        len(covered),
        min(rates) if rates else 0.0,
        max(rates) if rates else 0.0,
        skipped,
    )


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_rvs_pebb(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every PEBB action must measure positive PCIe bandwidth across all GPUs."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))

    with step(f"Run RVS PEBB ({_CONF_NAME})"):
        output, exit_code = run_rvs(target_executor, rvs_env, binary, conf_path, _RVS_TIMEOUT, _LABEL)

    with step("Qualify PCIe bandwidth results"):
        assert_log_sane(output, exit_code, _LABEL)

        samples, enumerated, skipped, invalid = _parse_samples(output)
        # Without this an empty or unparseable log would assert vacuously.
        assert enumerated, (
            f"No PEBB bandwidth entries found; expected '[<action>] pcie-bandwidth "
            f"[n/m] [CPU:: c] [GPU:: ...]' lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        _report_pebb_metrics(samples, enumerated, skipped)
        _assert_bandwidth_usable(samples, skipped, invalid)

    declared = declared_actions(target_executor, conf_path)
    _assert_coverage(samples, enumerated, declared)
    assert_summary_passed(parse_summary(output), declared, _LABEL, conf)
