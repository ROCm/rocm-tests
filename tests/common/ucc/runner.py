# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""UCC MPI test report parsing and assertion helpers.

Parses the ``===== UCC MPI TEST REPORT =====`` / ``===== UCC MPI TEST SUMMARY =====``
blocks emitted by ``ucc_test_mpi`` and asserts a clean collective run.
Shared across current and future UCC test suites.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)

_REPORT_MARKER = "===== UCC MPI TEST REPORT ====="
_SUMMARY_MARKER = "===== UCC MPI TEST SUMMARY ====="
_ALL_SKIPPED_MARKER = "All tests have been skipped"

_ROW_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z_]*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$",
    re.MULTILINE,
)
_SUMMARY_FIELDS: dict[str, re.Pattern[str]] = {
    "total": re.compile(r"^\s*total tests:\s*(\d+)\s*$", re.MULTILINE),
    "passed": re.compile(r"^\s*passed:\s*(\d+)\s*$", re.MULTILINE),
    "skipped": re.compile(r"^\s*skipped:\s*(\d+)\s*$", re.MULTILINE),
    "failed": re.compile(r"^\s*failed:\s*(\d+)\s*$", re.MULTILINE),
}


@dataclass(frozen=True)
class CollCounts:
    """Counts from one ``UCC MPI TEST REPORT`` table row."""

    collective: str
    tests: int
    passed: int
    failed: int
    skipped: int


@dataclass(frozen=True)
class UccReport:
    """Parsed ``ucc_test_mpi`` report and summary."""

    per_coll: dict[str, CollCounts]
    total: int
    passed: int
    skipped: int
    failed: int

    def counts_for(self, collective: str) -> CollCounts | None:
        """Return the row for *collective*; the table uses ``Gather``, callers use ``gather``."""
        return self.per_coll.get(collective.lower())


def parse_ucc_report(output: str) -> UccReport | None:
    """Parse UCC ``TEST REPORT``/``TEST SUMMARY`` blocks from *output*.

    Returns ``None`` when no report was printed — the workload died before
    reaching its summary, which callers must treat as a failure rather than
    an absence of failures.

    Args:
        output: Combined stdout + stderr from the ``ucc_test_mpi`` run.
    """
    report_at = output.find(_REPORT_MARKER)
    summary_at = output.find(_SUMMARY_MARKER, report_at + 1 if report_at >= 0 else 0)
    if report_at < 0 or summary_at < 0:
        return None

    per_coll: dict[str, CollCounts] = {}
    for match in _ROW_RE.finditer(output[report_at:summary_at]):
        name, tests, passed, failed, skipped = match.groups()
        per_coll[name.lower()] = CollCounts(
            collective=name,
            tests=int(tests),
            passed=int(passed),
            failed=int(failed),
            skipped=int(skipped),
        )

    summary_text = output[summary_at:]
    values: dict[str, int] = {}
    for field, pattern in _SUMMARY_FIELDS.items():
        found = pattern.search(summary_text)
        if found is None:
            return None
        values[field] = int(found.group(1))

    return UccReport(per_coll=per_coll, **values)


def assert_ucc_all_passed(output: str, report: UccReport | None, label: str = "") -> None:
    """Assert the UCC report shows total > 0, passed > 0, failed == 0, no all-skipped.

    Args:
        output: Combined stdout + stderr (used in failure messages).
        report: Parsed ``UccReport``; ``None`` means the binary crashed before summary.
        label:  Short test name used in failure messages.
    """
    prefix = f"{label}: " if label else ""
    tail = output[-4000:]

    assert _ALL_SKIPPED_MARKER not in output, f"{prefix}ucc_test_mpi reported that every test was skipped:\n{tail}"
    assert report is not None, (
        f"{prefix}ucc_test_mpi printed no test report — " f"the run died before reaching its summary:\n{tail}"
    )
    assert not report.failed, f"{prefix}summary reports {report.failed} failed test(s):\n{tail}"
    assert report.total, f"{prefix}summary reports 0 total tests — nothing ran:\n{tail}"
    assert report.passed, f"{prefix}summary reports 0 passed tests ({report.skipped} skipped):\n{tail}"
    assert report.skipped != report.total, f"{prefix}all {report.total} tests were skipped:\n{tail}"
