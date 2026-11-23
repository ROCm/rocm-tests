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

import logging
import os

import pytest

from tests.common.ucc.runner import assert_ucc_all_passed, parse_ucc_report

logger = logging.getLogger(__name__)

COLLS: tuple[str, ...] = tuple(
    coll.strip()
    for coll in os.environ.get("ROCM_TEST_UCC_COLLS", "gather,gatherv,scatter,scatterv").split(",")
    if coll.strip()
)

# Four is the minimum that exercises the half/reverse/odd-even sub-teams the
# workload builds from the world communicator; hence the 4-GPU gate.
RANKS = int(os.environ.get("ROCM_TEST_UCC_RANKS", "4"))

# ``rocm`` is the point of the test: it moves device memory, not host buffers.
MTYPES = os.environ.get("ROCM_TEST_UCC_MTYPES", "rocm")

MPIRUN_EXPORTS: tuple[str, ...] = (
    "UCC_TL_UCP_TUNE=rocm:0",
    "UCX_TLS=tcp,sm,self",
)

# Bounds the collective run only; the dependency build has its own timeout.
RUN_TIMEOUT = float(os.environ.get("ROCM_TEST_UCC_RUN_TIMEOUT", "1800"))

_CRASH_PATTERNS: tuple[str, ...] = (
    "core dumped",
    "Segmentation fault",
    "exited on signal",
    "ucs_fatal_error",
    "Caught signal",
)


def build_run_command(mpirun: str, ucc_test_mpi: str) -> str:
    """Return the ``mpirun`` command line for the workload."""
    exports = " ".join(f"-x {assignment}" for assignment in MPIRUN_EXPORTS)
    return (
        f"{mpirun} --allow-run-as-root {exports} -np {RANKS} {ucc_test_mpi} "
        f"--mtypes {MTYPES} --set_device 1 --inplace 2 --verbose "
        f"--colls {','.join(COLLS)}"
    )


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

    report = parse_ucc_report(output)
    assert_ucc_all_passed(output, report, label="ucc_test_mpi")

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

    assert (
        result.exit_code == 0
    ), f"ucc_test_mpi reported {report.passed}/{report.total} passed but exited {result.exit_code}:\n{tail}"

    logger.info(
        "UCC MPI: %d/%d tests passed (%d skipped) across %s",
        report.passed,
        report.total,
        report.skipped,
        ", ".join(sorted(report.per_coll)),
    )
