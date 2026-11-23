# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GoogleTest binary invocation and result-parsing helpers.

Provides executor-transparent helpers for running GTest binaries with
``--gtest_filter``, parsing ``[PASSED N]``/``[FAILED N]`` summary lines,
and asserting a clean run.  Shared across rocprim, hipblaslt, ucc, hpc/ucx,
and hwq_heuristic test suites.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import shlex

import pytest

logger = logging.getLogger(__name__)

# GTest summary line patterns.
_RAN_RE = re.compile(r"\[=+\]\s+(\d+)\s+tests?\s+from\s+\d+\s+test\s+suites?\s+ran", re.IGNORECASE)
_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+([1-9][0-9]*)\s+tests?\.", re.IGNORECASE | re.MULTILINE)
_FAILED_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\d+)\s+tests?", re.IGNORECASE)

# Binary crash signatures that appear independent of the numeric summary.
_CRASH_MARKERS: tuple[str, ...] = (
    "segmentation fault",
    "core dumped",
    "terminate called",
    "aborted",
)


@dataclass
class GtestSummary:
    """Parsed outcome of a GoogleTest run.

    Attributes:
        ok:        Exit 0 and zero FAILED cases.
        exit_code: Process exit code.
        total:     Tests that ran (0 when binary crashed before announcing).
        passed:    Tests reported PASSED.
        failed:    Tests reported FAILED.
        output:    Combined stdout + stderr.
    """

    ok: bool
    exit_code: int
    total: int = 0
    passed: int = 0
    failed: int = 0
    output: str = ""


def run_gtest_filter(
    executor,
    binary: str,
    ld_library_path: str,
    gtest_filter: str,
    timeout: float = 1800.0,
    extra_env: str = "",
) -> tuple[str, int]:
    """Run a GTest binary with ``--gtest_filter`` and return ``(output, exit_code)``.

    Detects binary crash markers in the output before returning so callers can
    surface a more specific error than a bare non-zero exit code.

    Args:
        executor:          Any executor with a ``.run()`` method.
        binary:            Absolute path to the GTest binary.
        ld_library_path:   Value for ``LD_LIBRARY_PATH``.
        gtest_filter:      GTest filter expression (e.g. ``"SystemMultiStreamTests.*"``).
        timeout:           Seconds before the run is abandoned.
        extra_env:         Optional additional ``VAR=value`` pairs to prepend.

    Returns:
        ``(combined_stdout_stderr, exit_code)``
    """
    ld = shlex.quote(ld_library_path)
    b = shlex.quote(binary)
    env_prefix = f"env LD_LIBRARY_PATH={ld}"
    if extra_env:
        env_prefix = f"{env_prefix} {extra_env}"
    cmd = f"{env_prefix} {b} --gtest_filter={gtest_filter}"
    logger.info("Running GTest: %s", cmd)
    result = executor.run(cmd, timeout=timeout)
    output = (result.stdout or "") + (result.stderr or "")
    return output, result.exit_code


def parse_gtest_summary(output: str) -> GtestSummary | None:
    """Parse ``[PASSED N]``/``[FAILED N]`` summary lines from GTest output.

    Returns ``None`` when the binary crashed before announcing the test suite,
    since an empty summary is indistinguishable from a zero-test partial run.

    Args:
        output: Combined stdout + stderr from the GTest run.
    """
    ran = _RAN_RE.search(output)
    if ran is None:
        return None

    total = int(ran.group(1))
    passed_match = _PASSED_RE.search(output)
    failed_match = _FAILED_RE.search(output)
    passed = int(passed_match.group(1)) if passed_match else 0
    failed = int(failed_match.group(1)) if failed_match else 0
    ok = failed == 0 and passed > 0

    return GtestSummary(ok=ok, exit_code=0, total=total, passed=passed, failed=failed, output=output)


def assert_gtest_passed(
    output: str,
    exit_code: int,
    label: str,
) -> None:
    """Assert the GTest run produced zero failures and at least one passing test.

    Checks for binary crash markers first, then parses the summary and asserts
    the expected outcome.  Failure messages include passed/failed counts.

    Args:
        output:    Combined stdout + stderr from the GTest run.
        exit_code: Process exit code.
        label:     Short name used in failure messages.
    """
    output_lower = output.lower()
    for marker in _CRASH_MARKERS:
        if marker in output_lower:
            pytest.fail(f"{label}: binary crash detected — '{marker}' in output.\n" f"output tail: {output[-2000:]}")

    summary = parse_gtest_summary(output)
    if summary is None:
        pytest.fail(
            f"{label}: GTest binary exited {exit_code} without announcing a test suite "
            f"— binary may have crashed before starting.\n"
            f"output tail: {output[-2000:]}"
        )

    assert summary.failed == 0, (
        f"{label}: GTest reported {summary.failed} failure(s) — "
        f"passed={summary.passed} total={summary.total} exit={exit_code}.\n"
        f"output tail: {output[-2000:]}"
    )
    assert summary.passed > 0, (
        f"{label}: GTest reported 0 tests passed — partial run or empty filter? "
        f"total={summary.total} exit={exit_code}.\n"
        f"output tail: {output[-2000:]}"
    )


def assert_no_gtest_crashes(output: str, label: str = "") -> None:
    """Fail when output contains binary crash signatures independent of the summary.

    Useful when the caller wants to detect a crash before inspecting the
    numeric pass/fail counts.

    Args:
        output: Combined stdout + stderr.
        label:  Short name used in the failure message.
    """
    output_lower = output.lower()
    for marker in _CRASH_MARKERS:
        if marker in output_lower:
            prefix = f"{label}: " if label else ""
            pytest.fail(f"{prefix}crash marker '{marker}' detected in GTest output.\n" f"output tail: {output[-2000:]}")
