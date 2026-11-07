# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functional RAJA HIP unit tests.

Builds LLNL/RAJA with HIP and requires ctest to pass. This is the functional
suite. It does not collect or compare RAJAPerf scores.

Markers other than ``runtime.soak`` come from the CATEGORY_PROFILE for
``tests/e2e/hpc/raja/`` (``ci.weekly``). The full ctest run exceeds 30 minutes.
"""

import logging
import re
import shlex

import pytest

from tests.e2e.hpc.raja._workload import CTEST_RUN_TIMEOUT, CTEST_TIMEOUT

logger = logging.getLogger(__name__)

_CTEST_FAIL_BANNER = "The following tests FAILED"
_PASSED_RE = re.compile(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)")


@pytest.mark.runtime.soak  # type: ignore[attr-defined]
def test_raja_functional(target_executor, rock_dir: str, ld_path: dict, raja_functional_build):
    """RAJA HIP unit tests must build and pass ctest."""
    build = raja_functional_build
    ld = ld_path["LD_LIBRARY_PATH"]
    ctest_log = f"{build.build_dir}/results.log"
    # Apt Open MPI is already on PATH. A provisioned runtime is not.
    mpi_bin = f"{shlex.quote(build.mpi_bin)}:" if build.mpi_bin else ""
    mpi_lib = f"{shlex.quote(build.mpi_lib)}:" if build.mpi_lib else ""
    # The e2e container runs as root and has no sshd. Open MPI refuses root
    # unless these are set, and its default rsh/ssh launcher hangs every
    # MPI test. isolated forks ranks on this node.
    cmd = (
        f"env ROCM_PATH={shlex.quote(rock_dir)} "
        f"PATH={mpi_bin}{shlex.quote(rock_dir)}/bin:$PATH "
        f"LD_LIBRARY_PATH={mpi_lib}{shlex.quote(ld)}:$LD_LIBRARY_PATH "
        f"OMPI_ALLOW_RUN_AS_ROOT=1 "
        f"OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 "
        f"OMPI_MCA_plm=isolated "
        f"OMPI_MCA_btl=tcp,self "
        f"OMPI_MCA_pml=ob1 "
        f"OMPI_MCA_rmaps_base_oversubscribe=1 "
        f"OMPI_MCA_hwloc_base_binding_policy=none "
        f"ctest --test-dir {shlex.quote(build.build_dir)} --output-on-failure "
        f"--timeout {CTEST_TIMEOUT} --output-log {shlex.quote(ctest_log)}"
    )
    logger.info("RAJA functional ctest starting; full log -> %s", ctest_log)
    result = target_executor.run(cmd, timeout=CTEST_RUN_TIMEOUT)

    stdout = result.stdout or ""
    no_tests = f"RAJA ctest ran no tests — build produced no test targets:\n{stdout[-2000:]}"
    assert "No tests were found" not in stdout, no_tests
    summary = _PASSED_RE.search(stdout)
    assert summary, f"RAJA ctest did not report a tests-passed count:\n{stdout[-2000:]}"
    percent, failed, total = (int(summary.group(1)), int(summary.group(2)), int(summary.group(3)))
    detail = f"RAJA ctest reported {percent}% tests passed, {failed} failed out of {total}:\n{stdout[-2000:]}"
    assert total > 0, detail
    assert failed == 0, detail
    assert percent == 100, detail
    if _CTEST_FAIL_BANNER in stdout or not result.ok:
        pytest.fail(
            f"RAJA functional ctest failed (exit={result.exit_code}):\n"
            f"stdout: {stdout[-4000:]}\nstderr: {(result.stderr or '')[-1000:]}"
        )
