# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functional Kokkos HIP tests (RDC off).

Builds Kokkos 5.1.1 with HIP tests enabled and relocatable device code off,
then requires the ctest suite to pass. Benchmarks stay off. This is not the
HIP RDC (``-fgpu-rdc``) test.

Two Kokkos 5.1.1 targets are excluded. ``Kokkos_CoreUnitTest_HIP`` aborts in
``TestHIP_AsyncLauncher.cpp`` (wrong ``hipMemcpy`` direction).
``Kokkos_CoreUnitTest_HIPInterOpGraph`` fails on gfx942. The names are
anchored so ``Kokkos_CoreUnitTest_HIP_ViewSupport`` still runs.

Markers other than ``runtime.soak`` come from the CATEGORY_PROFILE for
``tests/e2e/hpc/kokkos/`` (``ci.weekly``).
"""

import logging
import shlex

import pytest

from tests.e2e.hpc.kokkos._workload import CTEST_RUN_TIMEOUT, CTEST_TIMEOUT

logger = logging.getLogger(__name__)

_CTEST_FAIL_BANNER = "The following tests FAILED"

# Anchored so the HIP prefix does not also drop HIP_ViewSupport / HIPInterOpInit.
_EXCLUDED_TESTS = r"^Kokkos_CoreUnitTest_HIP$|^Kokkos_CoreUnitTest_HIPInterOpGraph$"


@pytest.mark.runtime.soak  # type: ignore[attr-defined]
def test_kokkos_hip_functional_ctest(
    target_executor,
    rock_dir: str,
    ld_path: dict,
    kokkos_functional_build,
):
    """Kokkos 5.1.1 HIP ctest must pass with relocatable device code off."""
    build = kokkos_functional_build
    assert (
        not build.rdc_enabled
    ), "functional Kokkos build must keep relocatable device code off"

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
        f"-E {shlex.quote(_EXCLUDED_TESTS)} "
        f"--timeout {CTEST_TIMEOUT} --output-log {_qc}"
    )
    logger.info("Kokkos functional ctest starting; full log -> %s", ctest_log)
    result = target_executor.run(cmd, timeout=CTEST_RUN_TIMEOUT)

    no_tests = f"Kokkos ctest ran no tests \u2014 build produced no test targets:\n{result.stdout[-2000:]}"
    assert "No tests were found" not in (result.stdout or ""), no_tests
    assert (result.stdout or "").strip(), no_tests
    detail = (
        f"Kokkos functional ctest failed (exit={result.exit_code}):\n"
        f"stdout: {(result.stdout or '')[-4000:]}\nstderr: {(result.stderr or '')[-1000:]}"
    )
    assert result.ok, detail
    assert _CTEST_FAIL_BANNER not in (result.stdout or ""), detail
