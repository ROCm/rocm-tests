# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run UCC gather/scatter collectives over ROCm device buffers on four MPI ranks.

``conftest.py`` builds UCX, OpenMPI and UCC from source; this file runs
``ucc_test_mpi`` with ``--mtypes rocm`` over gather/gatherv/scatter/scatterv and
parses its per-collective report. Every requested collective must have actually
run and passed -- reporting no failures is not enough, since the workload also
reports none when it skipped everything.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re

import pytest

logger = logging.getLogger(__name__)

COLLS: tuple[str, ...] = tuple(
    coll.strip()
    for coll in os.environ.get("ROCM_TEST_UCC_COLLS", "gather,gatherv,scatter,scatterv").split(",")
    if coll.strip()
)

# Four is the minimum that exercises the half/reverse/odd-even sub-teams the
# workload builds from the world communicator; hence the original's 4-GPU gate.
RANKS = int(os.environ.get("ROCM_TEST_UCC_RANKS", "4"))

# ``rocm`` is the point of the port: it moves device memory, not host buffers.
MTYPES = os.environ.get("ROCM_TEST_UCC_MTYPES", "rocm")

MPIRUN_EXPORTS: tuple[str, ...] = (
    "UCC_TL_UCP_TUNE=rocm:0",
    "UCX_TLS=tcp,sm,self",
)

# Bounds the collective run only; the dependency build has its own timeout.
RUN_TIMEOUT = float(os.environ.get("ROCM_TEST_UCC_RUN_TIMEOUT", "1800"))

_REPORT_MARKER = "===== UCC MPI TEST REPORT ====="
_SUMMARY_MARKER = "===== UCC MPI TEST SUMMARY ====="
ALL_SKIPPED_MARKER = "All tests have been skipped"

# Narrow by design: UCX and OpenMPI print benign "No such file or directory"
# warnings while probing transports, so only unambiguous crashes are listed.
_CRASH_PATTERNS: tuple[str, ...] = (
    "core dumped",
    "Segmentation fault",
    "exited on signal",
    "ucs_fatal_error",
    "Caught signal",
)

# A table row is a collective name followed by tests/passed/failed/skipped.
_ROW_RE = re.compile(r"^\s*([A-Za-z][A-Za-z_]*)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$", re.MULTILINE)
_SUMMARY_FIELDS = {
    "total": re.compile(r"^\s*total tests:\s*(\d+)\s*$", re.MULTILINE),
    "passed": re.compile(r"^\s*passed:\s*(\d+)\s*$", re.MULTILINE),
    "skipped": re.compile(r"^\s*skipped:\s*(\d+)\s*$", re.MULTILINE),
    "failed": re.compile(r"^\s*failed:\s*(\d+)\s*$", re.MULTILINE),
}


@dataclass(frozen=True)
class CollCounts:
    """One ``===== UCC MPI TEST REPORT =====`` table row."""

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
        """Return the row for *collective*; the table says ``Gather``, ``--colls`` says ``gather``."""
        return self.per_coll.get(collective.lower())


def build_run_command(mpirun: str, ucc_test_mpi: str) -> str:
    """Return the ``mpirun`` command line for the ported workload."""
    exports = " ".join(f"-x {assignment}" for assignment in MPIRUN_EXPORTS)
    return (
        f"{mpirun} {exports} -np {RANKS} {ucc_test_mpi} "
        f"--mtypes {MTYPES} --set_device 1 --inplace 2 --verbose "
        f"--colls {','.join(COLLS)}"
    )


def parse_report(output: str) -> UccReport | None:
    """Parse the UCC report/summary blocks out of *output*.

    Returns ``None`` when no report was printed at all -- the workload died
    before reaching its summary, which the caller must treat as a failure
    rather than as an absence of failures.
    """
    report_at = output.find(_REPORT_MARKER)
    summary_at = output.find(_SUMMARY_MARKER, report_at + 1 if report_at >= 0 else 0)
    if report_at < 0 or summary_at < 0:
        return None

    # Rows live strictly between the two markers, which keeps the permissive
    # row regex away from the ``--verbose`` per-test chatter surrounding them.
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


def find_crashes(output: str) -> list[str]:
    """Return crash/abort evidence found in *output*."""
    return [pattern for pattern in _CRASH_PATTERNS if pattern.lower() in output.lower()]


@pytest.mark.gpu_count(RANKS)
@pytest.mark.runtime.medium
def test_ucc_mpi_collectives(target_executor, ucc_stack, requested_gpu_count: int) -> None:
    """Run ``ucc_test_mpi`` on ROCm buffers and require every collective to pass."""
    if requested_gpu_count < RANKS:
        pytest.skip(f"UCC MPI needs at least {RANKS} GPUs; {requested_gpu_count} acquired")

    command = build_run_command(ucc_stack.mpirun, ucc_stack.ucc_test_mpi)
    logger.info("Running UCC MPI collectives: %s", command)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ucc_stack.ld_library_path} {command}",
        timeout=RUN_TIMEOUT,
    )
    output = (result.stdout or "") + (result.stderr or "")
    tail = output[-4000:]

    crashes = find_crashes(output)
    assert not crashes, f"ucc_test_mpi crashed ({', '.join(crashes)}):\n{tail}"

    # Fatal on its own: the summary still says "failed: 0" when nothing ran.
    assert ALL_SKIPPED_MARKER not in output, f"ucc_test_mpi reported that every test was skipped:\n{tail}"

    report = parse_report(output)
    assert report is not None, (
        f"ucc_test_mpi printed no test report (exit={result.exit_code}); "
        f"the run died before reaching its summary:\n{tail}"
    )

    # --- summary block: necessary, but on its own not sufficient -------------
    assert not report.failed, f"summary reports {report.failed} failed test(s):\n{tail}"
    assert report.total, f"summary reports 0 total tests — nothing ran:\n{tail}"
    assert report.passed, f"summary reports 0 passed tests ({report.skipped} skipped):\n{tail}"
    assert report.skipped != report.total, f"all {report.total} tests were skipped:\n{tail}"
    assert report.total == report.passed + report.skipped + report.failed, (
        f"summary counts do not add up: total={report.total} != passed={report.passed} + "
        f"skipped={report.skipped} + failed={report.failed}"
    )

    # --- per collective: each requested one must have run and passed ---------
    for collective in COLLS:
        counts = report.counts_for(collective)
        assert counts is not None, (
            f"{collective}: requested but absent from the report table "
            f"(saw {', '.join(sorted(report.per_coll)) or 'nothing'})"
        )
        assert counts.tests, f"{collective}: 0 tests ran"
        assert not counts.failed, f"{collective}: {counts.failed} of {counts.tests} tests failed"
        assert counts.passed, f"{collective}: 0 tests passed ({counts.skipped} skipped)"

    # A collective the workload chose to run on its own must not fail either.
    requested = {collective.lower() for collective in COLLS}
    other_failures = [
        f"{counts.collective}: {counts.failed} failed"
        for counts in report.per_coll.values()
        if counts.failed and counts.collective.lower() not in requested
    ]
    assert not other_failures, "unrequested collectives reported failures:\n  " + "\n  ".join(other_failures)

    assert result.exit_code == 0, (
        f"ucc_test_mpi reported {report.passed}/{report.total} passed but exited {result.exit_code}:\n{tail}"
    )

    logger.info(
        "UCC MPI: %d/%d tests passed (%d skipped) across %s",
        report.passed,
        report.total,
        report.skipped,
        ", ".join(sorted(report.per_coll)),
    )
