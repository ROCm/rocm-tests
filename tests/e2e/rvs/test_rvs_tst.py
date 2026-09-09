# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS TST -- thermal stress test: drive GEMM load and qualify the thermal response.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_tst.py`` and its parser
``logParser/tests/RVS/rvs_tst_parse.py``. TST runs unprivileged. It ramps a GEMM
workload towards ``target_temp`` and reports one verdict per (action, GPU)
alongside a stream of temperature samples::

    [INFO  ] [ 73532.520487] [action_1] tst GPU 9354 Current edge temperature is :  43.000000
    [RESULT] [ 73562.603180] [action_1] [GPU:: 9354] pass: TRUE

Both signals matter. The verdict alone is not enough: TST can report ``pass:
TRUE`` while never reading a usable temperature, so the original also required
every reported temperature to be greater than zero. That check is kept here,
with the same gfx942 exemption -- MI300 legitimately reports 0 for these sensors.

Corrections against the original parser:

* Verdicts are keyed by ``(action, gpu)``. The original stored them under the
  action name alone, so on an 8-GPU host seven of the eight results were
  overwritten and only the last GPU's verdict survived.
* The reported ``TRUE``/``FALSE`` is actually honoured. The original tested
  ``if status and flag``, where ``status`` is the non-empty string ``"TRUE"`` or
  ``"FALSE"`` and therefore always truthy, so a genuine ``pass: FALSE`` was
  recorded as a pass. It only failed the run via a separate scan of the raw log
  text for the substring ``FALSE``.

Note this suite also drives TST through ``tests/e2e/gpu_monitored`` with amd-smi
telemetry wrapped around it. This test is the standalone equivalent: same binary
and config, no monitoring pipeline, and it additionally qualifies the temperature
samples that the monitored validator does not inspect.
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
    declared_actions,
    run_rvs,
)

logger = logging.getLogger(__name__)

_CONF_NAME = "tst_single.conf"
_LABEL = "TST"
# action_1 is serial (parallel: false), so wall time scales with GPU count:
# ~30 s per GPU on the MI210 config, ~120 s per GPU on the generic one.
_RVS_TIMEOUT = 1800.0

# ``[RESULT] [ ts ] [<action>] [GPU:: <id>] pass: TRUE|FALSE``. Case-sensitive on
# the verdict token so prose containing "pass: true" cannot be mistaken for one.
_VERDICT_RE = re.compile(r"\[\s*RESULT\s*\].*\[([^\]]+)\]\s*\[GPU::\s*(\d+)\]\s*pass:\s*(TRUE|FALSE)\b")
_TEMP_RE = re.compile(
    r"\[([^\]]+)\]\s*tst\s+GPU\s+(\d+)\s+Current\s+(edge|junction)\s+temperature\s+is\s*:\s*([\d.-]+)"
)
_GFX_RE = re.compile(r"gfx[0-9a-f]+", re.IGNORECASE)

# MI300 reports 0 for the edge/junction sensors TST reads, so the
# temperature-above-zero rule is waived there rather than failing the platform.
_ZERO_TEMP_OK_ARCHS = ("gfx942",)


def _target_arch(executor, rock_dir: str, gpu_arch: str | None) -> str:
    """Return the GPU architecture, preferring an explicit ``--gpu-arch``."""
    if gpu_arch:
        return gpu_arch.lower()
    result = executor.run(f"{shlex.quote(rock_dir)}/bin/rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+'")
    match = _GFX_RE.search(result.stdout or "")
    return match.group(0).lower() if match else ""


def _parse_verdicts(text: str) -> dict[tuple, bool]:
    """Return ``{(action, gpu_id): passed}`` -- one entry per GPU per action."""
    return {
        (match.group(1), match.group(2)): match.group(3) == "TRUE"
        for match in (_VERDICT_RE.search(line) for line in text.splitlines())
        if match
    }


def _parse_temperatures(text: str) -> list[tuple]:
    """Return ``(action, gpu_id, sensor, value)`` for every temperature sample."""
    samples = []
    for line in text.splitlines():
        if match := _TEMP_RE.search(line):
            samples.append((match.group(1), match.group(2), match.group(3), float(match.group(4))))
    return samples


def _assert_temperatures_usable(samples: list[tuple], arch: str) -> None:
    """Every reported temperature must be positive, so a blind pass cannot slip through."""
    assert samples, (
        "RVS TST reported no temperature samples; a thermal stress test that never "
        "read a temperature cannot be considered to have qualified anything."
    )
    if arch.startswith(_ZERO_TEMP_OK_ARCHS):
        logger.info("Skipping the temperature>0 rule on %s (MI300 reports 0 for these sensors)", arch)
        return
    bad = [f"[{action}] GPU {gpu} {sensor}={value}" for action, gpu, sensor, value in samples if value <= 0]
    assert not bad, "{} of {} temperature sample(s) were not positive on {}:\n{}".format(
        len(bad), len(samples), arch or "unknown arch", "\n".join(bad[:10])
    )


def _report_tst_metrics(verdicts: dict[tuple, bool], samples: list[tuple]) -> None:
    """Publish verdict and thermal coverage so a shrinking run surfaces in reports."""
    values = [value for _, _, _, value in samples]
    report_metric("RVS_TST_VERDICTS", float(len(verdicts)))
    report_metric("RVS_TST_GPUS_COVERED", float(len({gpu for _, gpu in verdicts})))
    report_metric("RVS_TST_TEMP_SAMPLES", float(len(samples)))
    if values:
        report_metric("RVS_TST_TEMP_MAX_C", max(values), "C")
        report_metric("RVS_TST_TEMP_MIN_C", min(values), "C")
    logger.info(
        "TST: %d verdicts across %d GPUs; %d temperature samples (%.1f-%.1f C)",
        len(verdicts),
        len({gpu for _, gpu in verdicts}),
        len(samples),
        min(values) if values else 0.0,
        max(values) if values else 0.0,
    )


def _assert_all_passed(verdicts: dict[tuple, bool]) -> None:
    """Fail with the specific (action, GPU) pairs that reported ``pass: FALSE``."""
    failed = sorted(f"[{action}] GPU {gpu}" for (action, gpu), passed in verdicts.items() if not passed)
    per_action = collections.Counter(action for action, _ in verdicts)
    assert not failed, "{} of {} TST verdict(s) reported 'pass: FALSE' (per-action totals {}):\n{}".format(
        len(failed), len(verdicts), dict(per_action), "\n".join(failed[:20])
    )


def _assert_actions_covered(executor, conf_path: str, conf: str, verdicts: dict[tuple, bool]) -> None:
    """Every action the config declares must have produced verdicts."""
    missing = sorted(declared_actions(executor, conf_path) - {action for action, _ in verdicts})
    assert not missing, f"TST action(s) declared in {conf} produced no verdict: {', '.join(missing)}"


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_rvs_tst(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env, rock_dir, gpu_arch):
    """Every TST action must pass on every GPU, backed by usable temperature readings."""
    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))
    arch = _target_arch(target_executor, rock_dir, gpu_arch)

    with step(f"Run RVS TST ({_CONF_NAME}) on {arch or 'unknown arch'}"):
        output, exit_code = run_rvs(target_executor, rvs_env, binary, conf_path, _RVS_TIMEOUT, _LABEL)

    with step("Qualify verdicts and temperature samples"):
        assert_log_sane(output, exit_code, _LABEL)

        verdicts = _parse_verdicts(output)
        # Without this an empty or unparseable log would assert vacuously.
        assert verdicts, (
            f"No TST verdicts found; expected '[<action>] [GPU:: <id>] pass: TRUE|FALSE' "
            f"lines from {conf} (exit={exit_code}):\n{output[-2000:]}"
        )

        samples = _parse_temperatures(output)
        _assert_temperatures_usable(samples, arch)
        _report_tst_metrics(verdicts, samples)

    _assert_actions_covered(target_executor, conf_path, conf, verdicts)
    _assert_all_passed(verdicts)
