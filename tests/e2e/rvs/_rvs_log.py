# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared log handling for the RVS module tests.

Every RVS module wraps its module-specific output in the same envelope: a stream
of ``[RESULT]``/``[INFO  ]`` lines followed by a summary table holding the
per-action verdict RVS itself stands behind::

    +=====================================================================+
    | Action Name                      | Module         | Result          |
    +=====================================================================+
    | action_1                         | IET            | PASS            |
    +---------------------------------------------------------------------+

Each test asserts on that table and then cross-checks the module's own detail
lines, because the two can disagree in ways that matter. RCQT, for instance,
reports ``PASS`` here while its detail lines list 35 missing packages. Keeping
the envelope in one place stops the RVS tests from drifting apart on what counts
as a failure.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

RVS_DEBUG_LEVEL = 3

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Verdict cells are colour-coded, so ANSI is stripped before matching. The module
# column is upper-case alphanumeric, which is what keeps the header row ("Result")
# and the GPU inventory rows above the table from matching.
_SUMMARY_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*([A-Z0-9_]+)\s*\|\s*(PASS|FAIL)\s*\|", re.MULTILINE)
_ACTION_DECL_RE = re.compile(r"^\s*-\s*name\s*:\s*(\S+)", re.MULTILINE)
# ``0000:03:00.0 - GPU[ 4 - 53592] AMD Instinct MI210`` from ``rvs -g``.
_GPU_LIST_RE = re.compile(r"GPU\[\s*(\d+)\s*-\s*(\d+)\s*\]")
# Anchored on RVS's own prefix so an unrelated libc abort() cannot fail the run.
_ABORT_RE = re.compile(r"\bABORT\b")
_RVS_ERROR_RE = re.compile(r"RVS-ERROR.*", re.IGNORECASE)


def strip_ansi(text: str) -> str:
    """Remove ANSI colour escapes so summary cells can be matched literally."""
    return _ANSI_RE.sub("", text)


def run_rvs(executor, rvs_env: str, binary: str, conf_path: str, timeout: float, label: str) -> tuple[str, int]:
    """Run one RVS config and return its combined output and exit code."""
    cmd = f"env {rvs_env} {binary} -c {conf_path} -d {RVS_DEBUG_LEVEL}"
    logger.info("Running %s: %s", label, cmd)
    result = executor.run(cmd, timeout=timeout)
    return (result.stdout or "") + (result.stderr or ""), result.exit_code


def list_gpu_ids(executor, rvs_env: str, binary: str, timeout: float = 300.0) -> list[str]:
    """Return the RVS GPU ids the binary can see, via ``rvs -g``."""
    result = executor.run(f"env {rvs_env} {binary} -g", timeout=timeout)
    output = (result.stdout or "") + (result.stderr or "")
    return [gpu_id for _node, gpu_id in _GPU_LIST_RE.findall(output)]


def assert_log_sane(output: str, exit_code: int, label: str) -> None:
    """Reject crashes and RVS-level errors before interpreting any verdict."""
    assert output.strip(), f"RVS {label} produced no output (exit={exit_code})"
    assert not _ABORT_RE.search(output), f"RVS {label} reported ABORT:\n{output[-2000:]}"
    errors = _RVS_ERROR_RE.findall(output)
    assert not errors, "RVS {} logged {} error(s):\n{}".format(label, len(errors), "\n".join(errors[:10]))


def parse_summary(text: str) -> dict[str, bool]:
    """Return ``{action: passed}`` from the run's summary table."""
    return {action: verdict == "PASS" for action, _module, verdict in _SUMMARY_ROW_RE.findall(strip_ansi(text))}


def declared_actions(executor, conf_path: str) -> set[str]:
    """Return the action names the config declares."""
    return set(_ACTION_DECL_RE.findall(executor.run(f"cat {conf_path}").stdout or ""))


def assert_summary_passed(summary: dict[str, bool], declared: set[str], label: str, conf: str) -> None:
    """Require a summary verdict for every declared action, and all of them PASS.

    The missing-action check is what stops a config whose actions silently never
    ran from passing on an empty table.
    """
    assert summary, f"RVS {label} printed no summary table; expected one row per action from {conf}"
    missing = sorted(declared - set(summary))
    assert not missing, f"RVS {label} action(s) declared in {conf} produced no summary row: {', '.join(missing)}"
    failed = sorted(action for action, passed in summary.items() if not passed)
    assert not failed, f"RVS {label} reported FAIL for {len(failed)} of {len(summary)} action(s): {', '.join(failed)}"
