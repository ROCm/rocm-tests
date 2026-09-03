# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS PEQT -- PCIe Qualification Tool checks across all AMD GPUs.

Builds ``rvs -c <conf> -d <level>``, runs it under sudo, and derives the
per-action verdicts from the log.

PEQT reads PCIe capability registers (link speed and width, slot power, atomic-op
completers, kernel driver, ...) and matches them against the regular expressions
each action declares in ``peqt_single.conf``. Two properties of the tool drive
the shape of this test:

* ``rvs`` exits 0 even when actions report ``peqt false`` -- an unprivileged run
  on gfx90a returned rc=0 with 11 of 17 actions failing. The verdict therefore
  has to be parsed out of the log and the return code is only a crash signal.
* The capability registers are readable only as root. Without privilege they
  read ``NOT SUPPORTED``, so every action carrying a regex fails. That is an
  environment limitation rather than a PCIe defect, so a host without
  passwordless sudo skips instead of reporting spurious failures.

``peqt_single`` maps to ``./`` in ``rvs_config_mapping.csv`` (no per-GPU
subdirectory) because the checks are device-agnostic; ``rvs_find_conf`` still
tries the GPU-specific directory first so a device that ships its own PEQT
config is picked up automatically.
"""

from __future__ import annotations

import logging
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric, step

logger = logging.getLogger(__name__)

_CONF_NAME = "peqt_single.conf"
_RVS_DEBUG_LEVEL = 3
# PEQT only reads config space, so it finishes in well under a second per action
# on an 8-GPU host. The ceiling is here to fail a wedged PCIe read rather than
# hang the session.
_RVS_TIMEOUT = 300.0

# Action names are of the form ``pcie_act_<n>``, matched as ``\w+_\d+`` so a
# config that renames its actions still parses.
_ACTION_NAME_RE = re.compile(r"\[\s*RESULT\s*\]\s*\[\s*\d+\.?\d*\s*\]\s*Action\s*name\s*:\s*(\w+_\d+)")
_VERDICT_RE = re.compile(r"\[\s*RESULT\s*\].*\[(\w+_\d+)\]\s+peqt\s+(true|false)\b", re.IGNORECASE)
_RVS_ERROR_RE = re.compile(r"RVS-ERROR\s.+\s*\[(\w+_\d+)\]")
# Case-sensitive: RVS emits its own uppercase ABORT, and a lowercase "abort()"
# from libc elsewhere in the log must not promote the run to FAIL.
_ABORT_RE = re.compile(r"\bABORT\b")


def _parse_peqt_actions(text: str) -> dict[str, bool]:
    """Return ``{action_name: passed}`` in log discovery order.

    The precedence matters because the three signals disagree on a partially
    failing run:

    1. ``[<action>] peqt true|false`` is authoritative wherever present.
    2. ``RVS-ERROR ... [<action>]`` fails an action only when that action never
       produced a verdict line -- RVS logs an error *and* a verdict for the same
       action, so letting the error win would double-count it as a failure.
    3. An action that was announced but produced neither is treated as passing.
    """
    announced: list[str] = []
    errored: set[str] = set()
    verdicts: dict[str, bool] = {}

    for line in text.splitlines():
        if match := _ACTION_NAME_RE.search(line):
            announced.append(match.group(1))
        if match := _RVS_ERROR_RE.search(line):
            errored.add(match.group(1))
        if match := _VERDICT_RE.search(line):
            verdicts[match.group(1)] = match.group(2).lower() == "true"

    results: dict[str, bool] = {}
    for name in (*announced, *verdicts, *errored):
        if name in results:
            continue
        results[name] = verdicts.get(name, name not in errored)
    return results


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_rvs_peqt(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every PEQT action in ``peqt_single.conf`` must report ``peqt true``."""
    if not target_executor.run("sudo -n true").ok:
        pytest.skip(
            "Passwordless sudo is not available for the test user. PEQT reads PCIe "
            "capability registers, which return NOT SUPPORTED unprivileged and would "
            "fail every action that asserts a capability regex."
        )

    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    # Absolute paths: the fixtures may hand back repo-relative ones, and sudo
    # resolving them against the caller's cwd is a trap on a remote executor.
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))
    cmd = f"sudo -n env {rvs_env} {binary} -c {conf_path} -d {_RVS_DEBUG_LEVEL}"

    with step(f"Run RVS PEQT ({_CONF_NAME})"):
        logger.info("Running PEQT: %s", cmd)
        result = target_executor.run(cmd, timeout=_RVS_TIMEOUT)
        output = (result.stdout or "") + (result.stderr or "")

    with step("Parse per-action PEQT verdicts"):
        assert output.strip(), f"RVS PEQT produced no output (exit={result.exit_code})"
        assert not _ABORT_RE.search(output), f"RVS PEQT reported ABORT:\n{output[-2000:]}"

        actions = _parse_peqt_actions(output)
        # A config that matched no action would otherwise assert vacuously, the
        # same zero-denominator hole the RVS and memtest validators close.
        assert actions, (
            f"No PEQT actions found in RVS output; expected '[<action>] peqt true|false' "
            f"lines from {conf} (exit={result.exit_code}):\n{output[-2000:]}"
        )

        failed = sorted(name for name, passed in actions.items() if not passed)
        report_metric("RVS_PEQT_ACTIONS_TOTAL", float(len(actions)))
        report_metric("RVS_PEQT_ACTIONS_PASSED", float(len(actions) - len(failed)))
        logger.info("PEQT actions: %d total, %d failed", len(actions), len(failed))

    assert (
        not failed
    ), f"{len(failed)}/{len(actions)} PEQT action(s) reported 'peqt false': {', '.join(failed)}\n{output[-2000:]}"
