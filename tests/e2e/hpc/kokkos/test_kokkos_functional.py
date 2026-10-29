# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functional Kokkos HIP tests (RDC off).

Builds Kokkos 5.1.1 with HIP tests enabled and relocatable device code off,
then requires the ctest suite to pass. This is the functional suite. It does
not compare benchmark scores, and it is not the HIP RDC (``-fgpu-rdc``) test.

Markers other than ``runtime.soak`` come from the CATEGORY_PROFILE for
``tests/e2e/hpc/kokkos/`` (``ci.weekly``). The full ctest run exceeds 30 minutes.
"""

import logging
import re
import shlex

import pytest

from tests.e2e.hpc.kokkos._workload import CTEST_RUN_TIMEOUT, CTEST_TIMEOUT

logger = logging.getLogger(__name__)

_CTEST_FAIL_BANNER = "The following tests FAILED"

# Known upstream Kokkos 5.1.1 bug: hipMemcpy called with the wrong direction
# in TestHIP_AsyncLauncher.cpp, so Kokkos_CoreUnitTest_HIP aborts. Same pin
# and same xfail as test_gpu_rdc_kokkos. Becomes XPASS when a patched Kokkos
# release is pinned.
_KNOWN_XFAIL_TESTS = frozenset({"Kokkos_CoreUnitTest_HIP"})


@pytest.mark.runtime.soak  # type: ignore[attr-defined]
def test_kokkos_functional(
    target_executor,
    rock_dir: str,
    ld_path: dict,
    kokkos_functional_build,
):
    """Kokkos HIP functional ctest must pass. Relocatable device code stays off."""
    build = kokkos_functional_build
    assert not build.rdc_enabled, "functional Kokkos build must keep relocatable device code off"

    ld = ld_path["LD_LIBRARY_PATH"]
    ctest_log = f"{build.build_dir}/results.log"
    _qr = shlex.quote(rock_dir)
    _qb = shlex.quote(build.build_dir)
    _qc = shlex.quote(ctest_log)
    cmd = (
        f"env ROCM_PATH={_qr} "
        f"PATH={_qr}/bin:$PATH "
        f"LD_LIBRARY_PATH={shlex.quote(ld)}:$LD_LIBRARY_PATH "
        f"ctest --test-dir {_qb} --output-on-failure "
        f"--timeout {CTEST_TIMEOUT} --output-log {_qc}"
    )
    logger.info("Kokkos functional ctest starting; full log -> %s", ctest_log)
    result = target_executor.run(cmd, timeout=CTEST_RUN_TIMEOUT)

    no_tests = f"Kokkos ctest ran no tests — build produced no test targets:\n{result.stdout[-2000:]}"
    assert "No tests were found" not in result.stdout, no_tests
    assert result.stdout.strip(), no_tests

    if _CTEST_FAIL_BANNER in result.stdout or not result.ok:
        failed = set(re.findall(r"^\s+\d+ - (\S+)", result.stdout, re.MULTILINE))
        if not failed:
            pytest.fail(
                f"Kokkos ctest exited non-zero (exit={result.exit_code}) but "
                f"reported no named failing targets.\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-1000:]}"
            )
        unknown = failed - _KNOWN_XFAIL_TESTS
        if unknown:
            pytest.fail(f"Kokkos functional ctest reported unexpected failures {unknown}:\n{result.stdout[-4000:]}")
        pytest.xfail(
            f"Kokkos functional ctest failures are known upstream bugs in Kokkos 5.1.1 "
            f"on this platform: {failed}. "
            "Will become XPASS when a patched Kokkos release is pinned."
        )
