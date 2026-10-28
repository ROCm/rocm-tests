# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functional RAJA HIP unit tests (qa-forge hpc-raja, ROCmTest raja_ut).

Builds LLNL/RAJA with HIP and requires ctest to pass. This is the functional
suite. It does not collect or compare RAJAPerf scores.

Markers other than ``runtime.soak`` come from the CATEGORY_PROFILE for
``tests/e2e/hpc/raja/`` (``ci.weekly``). The full ctest run exceeds 30 minutes.
"""

import logging
import shlex

import pytest

from tests.e2e.hpc.raja._workload import CTEST_RUN_TIMEOUT, CTEST_TIMEOUT

logger = logging.getLogger(__name__)

_CTEST_FAIL_BANNER = "The following tests FAILED"


@pytest.mark.runtime.soak  # type: ignore[attr-defined]
def test_raja_functional(target_executor, rock_dir: str, ld_path: dict, raja_functional_build):
    """RAJA HIP unit tests must build and pass ctest."""
    build = raja_functional_build
    ld = ld_path["LD_LIBRARY_PATH"]
    ctest_log = f"{build.build_dir}/results.log"
    cmd = (
        f"env ROCM_PATH={shlex.quote(rock_dir)} "
        f"PATH={shlex.quote(rock_dir)}/bin:$PATH "
        f"LD_LIBRARY_PATH={shlex.quote(ld)}:$LD_LIBRARY_PATH "
        f"ctest --test-dir {shlex.quote(build.build_dir)} --output-on-failure "
        f"--timeout {CTEST_TIMEOUT} --output-log {shlex.quote(ctest_log)}"
    )
    logger.info("RAJA functional ctest starting; full log -> %s", ctest_log)
    result = target_executor.run(cmd, timeout=CTEST_RUN_TIMEOUT)

    no_tests = f"RAJA ctest ran no tests — build produced no test targets:\n{result.stdout[-2000:]}"
    assert "No tests were found" not in (result.stdout or ""), no_tests
    assert (result.stdout or "").strip(), no_tests
    if _CTEST_FAIL_BANNER in (result.stdout or "") or not result.ok:
        pytest.fail(
            f"RAJA functional ctest failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-1000:]}"
        )
