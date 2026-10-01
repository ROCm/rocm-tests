# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_gpu_rdc_kokkos.py -- HIP Relocatable Device Code (RDC) end-to-end validation.

Validates:
    1. The HIP compiler actually performs RDC compilation: with
       ``Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON``, the ``-fgpu-rdc`` flag
       appears in the VERBOSE build output of the Kokkos
       (https://github.com/kokkos/kokkos) HPC library. This is the AUTHORITATIVE
       pass criterion.
    2. The Kokkos ctest suite (built with RDC on) runs on the GPU and every test
       passes — evidence the RDC-compiled binaries execute correctly. The ctest
       gate is strict: any failing test fails this test (as the original did by
       parsing ctest results into its verdict).

The clone + custom CMake configure/VERBOSE-build/install and the ``-fgpu-rdc``
scan are handled by the session-scoped ``kokkos_rdc_build`` fixture in
``conftest.py``. This test asserts the RDC verdict first, then drives the ctest
suite on the GPU node via ``target_executor`` with the ROCm runtime environment
injected as an ``env VAR=... cmd`` prefix (never via ``os.environ``).

Markers (declared explicitly here: only runtime.*; the rest are injected by the
CATEGORY_PROFILE for tests/e2e/hpc/kokkos/):
    hw.gpu        -- single GPU / single node (from CATEGORY_PROFILE)
    layer.runtime -- validates the HIP compiler's -fgpu-rdc (RDC) toolchain path,
                     not a math library (from CATEGORY_PROFILE)
    ci.nightly    -- third-party build + full ctest suite (from CATEGORY_PROFILE)
    e2e.stack     -- full-stack end-to-end scenario (from CATEGORY_PROFILE)
    os.linux      -- bash/cmake/hipcc build path is Linux-only (from CATEGORY_PROFILE)
    runtime.soak  -- ~50 min total (~6 min build as a session fixture + ~43 min
                     ctest; the ctest run alone exceeds 30 min -> soak)
"""

import logging

import pytest

from tests.e2e.hpc.kokkos._workload import CTEST_RUN_TIMEOUT, CTEST_TIMEOUT, ctest_parallel_arg

logger = logging.getLogger(__name__)

# ctest prints this banner when any test in the suite fails; its presence (or a
# non-zero ctest exit) fails the strict gate below.
_CTEST_FAIL_BANNER = "The following tests FAILED"


@pytest.mark.runtime.soak
def test_gpu_rdc_kokkos(
    target_executor,
    rock_dir: str,
    ld_path: dict,
    kokkos_rdc_build,
):
    """Assert HIP RDC compilation occurred, then run the Kokkos ctest suite.

    The ``-fgpu-rdc`` trace in the VERBOSE build output is the first gate (it
    proves RDC compilation actually happened). The ctest run is then strict:
    every test in the suite must pass, so any ctest failure fails this test.
    """
    build = kokkos_rdc_build
    ld = ld_path["LD_LIBRARY_PATH"]

    # --- AUTHORITATIVE RDC gate -----------------------------------------------
    # Faithful to the original: the sole hard pass/fail criterion is the presence
    # of -fgpu-rdc in the VERBOSE compile output. If it is absent, RDC compilation
    # did not happen and the test fails here (before ctest is even run).
    assert build.rdc_flag_present, (
        "HIP RDC not exercised: '-fgpu-rdc' was not found in the Kokkos VERBOSE "
        f"build output (Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON). "
        f"Inspect the build log: {build.build_log_path}"
    )

    # --- secondary validation: run the full Kokkos ctest suite (strict) -------
    # ROCR/HIP visible-device vars are intentionally omitted (target_executor
    # injects them). ctest runs in parallel (`-j`, mirroring the original launch)
    # with the original's long per-test --timeout so a slow-but-valid test (e.g.
    # the atomic performance benchmark) completes instead of being killed.
    # Parallelism is KOKKOS_CTEST_JOBS (default $(nproc), node-resolved); set it
    # to 1 for serial.
    ctest_log = f"{build.build_dir}/results.log"
    cmd = (
        f"env ROCM_PATH={rock_dir} "
        f"PATH={rock_dir}/bin:$PATH "
        f"LD_LIBRARY_PATH={ld}:$LD_LIBRARY_PATH "
        f"ctest --test-dir {build.build_dir} {ctest_parallel_arg()} --output-on-failure "
        f"--timeout {CTEST_TIMEOUT} --output-log {ctest_log}"
    )
    logger.info("Kokkos ctest starting; full log -> %s (use `pytest -s` for live output)", ctest_log)

    result = target_executor.run(cmd, timeout=CTEST_RUN_TIMEOUT)

    # Strict ctest gate (matches the original, which parsed ctest results into the
    # test verdict): every test in the suite must pass. First confirm the suite
    # actually ran, then require a clean exit with no reported failures.
    no_tests = f"Kokkos ctest ran no tests — build produced no test targets:\n{result.stdout[-2000:]}"
    assert "No tests were found" not in result.stdout, no_tests
    assert result.stdout.strip(), no_tests
    assert _CTEST_FAIL_BANNER not in result.stdout, (
        f"Kokkos ctest reported failing tests:\n{result.stdout[-4000:]}"
    )
    assert result.ok, (
        f"Kokkos ctest suite failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
    )
