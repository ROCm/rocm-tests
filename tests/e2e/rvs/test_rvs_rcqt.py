# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS RCQT -- ROCm configuration qualification: metapackage and package-list validation.

Ported from ROCmTest ``tests/TOOLS/RVS/rvs_rcqt.py`` and its parser
``logParser/tests/RVS/rvs_rcqt_parse.py``. RCQT runs unprivileged and
``rcqt_single`` maps to ``./`` in the config mapping, so it is device-agnostic.

``rcqt_single.conf`` contains only package checks: one metapackage action and one
deb/rpm package-list action. RVS reports counters per action rather than a
verdict::

    Packages install validation complete :
        Missing packages      : 35
        Installed packages    : 0

The original parser has two modes and picks between them on install type. On a
deb/rpm host it counts those numbers, and an action passes only when nothing is
missing or version-mismatched. On a TheRock install it instead scrapes the RVS
summary table -- which is a problem, because RVS prints ``PASS`` there even when
every package is missing, as observed on this hardware (0 installed, 35 missing,
summary ``PASS``). That branch cannot fail for any reason short of RVS itself
printing FAIL.

This port keeps the real counter-based validation and replaces the vacuous branch
with a skip: if ROCm was not installed by the package manager there is genuinely
nothing for RCQT to qualify, and saying so is more honest than reporting a pass.
The summary table is still consulted, but only as an extra way to fail -- an
explicit FAIL row is honoured even if the counters look clean.
"""

from __future__ import annotations

import logging
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric, step

logger = logging.getLogger(__name__)

_CONF_NAME = "rcqt_single.conf"
_RVS_DEBUG_LEVEL = 3
_RVS_TIMEOUT = 600.0

_ACTION_NAME_RE = re.compile(r"\[\s*RESULT\s*\]\s*\[\s*[\d.]+\s*\]\s*Action\s*name\s*:\s*(\S+)")
_INSTALLED_RE = re.compile(r"Installed packages\s*:\s*(\d+)")
_MISSING_RE = re.compile(r"Missing packages\s*:\s*(\d+)")
_MISMATCH_RE = re.compile(r"Version mismatch packages\s*:\s*(\d+)")
_META_MISSING = "Meta package not installed"
_RVS_ERROR_RE = re.compile(r"RVS-ERROR.*", re.IGNORECASE)
_ABORT_RE = re.compile(r"\bABORT\b")

# Summary rows arrive colourised: "| <action> | RCQT | \x1b[32mPASS\x1b[0m |".
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_SUMMARY_ROW_RE = re.compile(r"^\|\s*([\w\-]+)\s*\|\s*RCQT\s*\|\s*(PASS|FAIL|FAILED)\s*\|", re.IGNORECASE)

# ROCm component name prefixes as shipped by the deb/rpm repositories. Used only
# to decide whether ROCm is package-managed at all, never as a pass criterion.
_PKG_PREFIXES = (
    "rocm|hip|hsa|amd-smi|rocblas|miopen|rccl|migraphx|rocminfo|comgr|rpp|"
    "roctracer|rocprofiler|openmp-extras|composablekernel|hiptensor|half"
)

# Counts installed ROCm packages via dpkg, falling back to rpm. A TheRock or
# tarball install yields 0 while the package database is still perfectly
# readable, which is what distinguishes "nothing to check" from "all broken".
_PKG_PROBE_CMD = (
    "c=0; "
    'if command -v dpkg-query >/dev/null 2>&1; then c=$(dpkg-query -W -f="${Package} ${Status}\\n" 2>/dev/null '
    f"| awk '/install ok installed/{{print $1}}' | grep -cE '^({_PKG_PREFIXES})' || true); fi; "
    'if [ "$c" = "0" ] && command -v rpm >/dev/null 2>&1; then '
    f"c=$(rpm -qa 2>/dev/null | grep -cE '^({_PKG_PREFIXES})' || true); fi; "
    'echo "ROCM_PKG_COUNT=$c"'
)


def _installed_rocm_packages(executor) -> int:
    """Return how many ROCm packages the system package manager reports installed."""
    result = executor.run(_PKG_PROBE_CMD, timeout=120.0)
    match = re.search(r"ROCM_PKG_COUNT=(\d+)", result.stdout or "")
    if not match:
        pytest.skip(
            "Could not query the package database to determine whether ROCm is "
            f"package-managed (exit={result.exit_code}): {(result.stderr or '')[:300]}"
        )
    return int(match.group(1))


def _parse_rcqt_actions(text: str) -> dict[str, dict]:
    """Accumulate RCQT's per-action package counters in log order.

    Counters are summed rather than overwritten because an action may validate
    several package lists (deb and rpm) and report a block for each.
    """
    actions: dict[str, dict] = {}
    current: dict | None = None
    for line in text.splitlines():
        if match := _ACTION_NAME_RE.search(line):
            current = actions.setdefault(
                match.group(1),
                {"installed": 0, "missing": 0, "mismatch": 0, "meta_missing": 0, "error": False},
            )
            continue
        if current is None:
            continue
        if match := _INSTALLED_RE.search(line):
            current["installed"] += int(match.group(1))
        elif match := _MISSING_RE.search(line):
            current["missing"] += int(match.group(1))
        elif match := _MISMATCH_RE.search(line):
            current["mismatch"] += int(match.group(1))
        elif _META_MISSING in line:
            # Counted per action; the original used a single counter shared
            # across actions, which leaked one action's failures into the next.
            current["meta_missing"] += 1
        elif _RVS_ERROR_RE.search(line):
            current["error"] = True
    return actions


def _failure_reasons(actions: dict[str, dict]) -> dict[str, str]:
    """Return ``{action: reason}`` for every action that fails the counter rules."""
    failures: dict[str, str] = {}
    for name, counters in actions.items():
        if counters["error"]:
            failures[name] = "RVS-ERROR reported"
            continue
        problems = [
            f"{counters[key]} {label}"
            for key, label in (
                ("missing", "missing package(s)"),
                ("mismatch", "version-mismatched package(s)"),
                ("meta_missing", "missing metapackage(s)"),
            )
            if counters[key]
        ]
        if problems:
            failures[name] = f"{', '.join(problems)} (installed={counters['installed']})"
    return failures


def _summary_failures(text: str) -> list[str]:
    """Return actions the RVS summary table explicitly marks FAIL."""
    failed = []
    for line in text.splitlines():
        match = _SUMMARY_ROW_RE.match(_ANSI_RE.sub("", line))
        if match and match.group(2).upper() != "PASS":
            failed.append(match.group(1))
    return failed


@pytest.mark.hw.gpu
@pytest.mark.gpu_count(1)
@pytest.mark.runtime.fast
def test_rvs_rcqt(target_executor, rvs_binary, rvs_find_conf, gpu_conf_dir, rvs_env):
    """Every RCQT action must report all required ROCm packages installed."""
    with step("Determine whether ROCm is package-managed"):
        package_count = _installed_rocm_packages(target_executor)
        logger.info("Package manager reports %d installed ROCm package(s)", package_count)
        if package_count == 0:
            pytest.skip(
                "ROCm is not installed via the system package manager on this host "
                "(TheRock/tarball install), so RCQT's deb/rpm package validation has "
                "nothing to qualify. Note RVS's own summary reports PASS regardless, "
                "which would make this test pass vacuously."
            )

    conf = rvs_find_conf(_CONF_NAME, gpu_conf_dir=gpu_conf_dir)
    binary = shlex.quote(str(pathlib.Path(rvs_binary).resolve()))
    conf_path = shlex.quote(str(pathlib.Path(conf).resolve()))

    with step(f"Run RVS RCQT ({_CONF_NAME})"):
        cmd = f"env {rvs_env} {binary} -c {conf_path} -d {_RVS_DEBUG_LEVEL}"
        logger.info("Running RCQT: %s", cmd)
        result = target_executor.run(cmd, timeout=_RVS_TIMEOUT)
        output = (result.stdout or "") + (result.stderr or "")

    with step("Validate per-action package counters"):
        assert output.strip(), f"RVS RCQT produced no output (exit={result.exit_code})"
        assert not _ABORT_RE.search(output), f"RVS RCQT reported ABORT:\n{output[-2000:]}"

        actions = _parse_rcqt_actions(output)
        # Without this an unparseable log would assert vacuously -- the exact
        # failure mode the TheRock summary branch suffers from.
        assert actions, (
            f"No RCQT actions found; expected '[RESULT] ... Action name :<name>' lines "
            f"from {conf} (exit={result.exit_code}):\n{output[-2000:]}"
        )

        failures = _failure_reasons(actions)
        summary_failed = _summary_failures(output)
        report_metric("RVS_RCQT_ACTIONS_TOTAL", float(len(actions)))
        report_metric("RVS_RCQT_PACKAGES_INSTALLED", float(sum(a["installed"] for a in actions.values())))
        logger.info("RCQT actions: %s", actions)

    assert not failures, "RCQT package validation failed for {}/{} action(s):\n{}".format(
        len(failures), len(actions), "\n".join(f"  {name}: {reason}" for name, reason in sorted(failures.items()))
    )
    # Counters looked clean, so only an explicit FAIL row can still fail the run.
    assert not summary_failed, f"RVS summary table reports FAIL for: {', '.join(summary_failed)}"
