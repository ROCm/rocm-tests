# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_quda.py -- QUDA lattice-QCD library end-to-end HPC validation.

Validates:
    1. QUDA (https://github.com/lattice/quda) clones, configures with the HIP
       target, builds (incl. downloaded USQCD/QMP/Eigen), installs, and the full
       ``ctest`` suite passes under a 2-rank MPI launch (grid ``1 1 1 2``).

The clone + custom CMake build + install are handled by the session-scoped
``quda_build`` fixture in ``conftest.py``.  This test drives the resulting
``ctest`` suite on the GPU node via ``target_executor`` with the QUDA runtime
environment injected as an ``env VAR=... cmd`` prefix (never via ``os.environ``).

The GPU count is configurable on a single node via ``QUDA_NUM_GPUS`` (default 2):
``1`` runs single-GPU mode (``hw.gpu``, 1 MPI rank), ``>1`` runs multi-GPU mode
(``hw.multi_gpu``, one rank per GPU). See ``_workload.py`` for all env knobs
(``QUDA_NUM_GPUS`` / ``QUDA_TEST_GRID_SIZE`` / ``QUDA_CTEST_TIMEOUT``).

Markers (declared explicitly; also registered as a CATEGORY_PROFILE for
tests/e2e/quda/):
    hw.gpu / hw.multi_gpu -- chosen from QUDA_NUM_GPUS (single vs multi GPU)
    gpu_count(N)  -- acquire N=QUDA_NUM_GPUS GPUs from one node
    layer.math_lib -- QUDA is a GPU compute (lattice-QCD solver) library
    ci.nightly    -- third-party build + full ctest suite (from CATEGORY_PROFILE)
    e2e.stack     -- full-stack end-to-end scenario
    os.linux      -- bash/cmake/clang++ build path is Linux-only
    runtime.medium -- ctest suite runs ~15 min (build is a separate session fixture)
"""

import logging
import os

import pytest

from tests.e2e.quda._workload import CTEST_TIMEOUT, GRID, IS_SINGLE_GPU, NUM_GPUS, NUM_PROCS

logger = logging.getLogger(__name__)

# hw dimension follows the configured GPU count (QUDA_NUM_GPUS). Declared via the
# module-level ``pytestmark`` list rather than a bare MarkDecorator global: a
# standalone MarkDecorator is callable, so pytest's collector would introspect it
# and the repo's dotted-mark __getattr__ patch would emit spurious "unknown mark"
# warnings. This hw.* overrides the hw.multi_gpu default from the CATEGORY_PROFILE.
pytestmark = [pytest.mark.hw.gpu if IS_SINGLE_GPU else pytest.mark.hw.multi_gpu]

# ctest emits this on success: "100% tests passed, 0 tests failed out of N".
# The banner below is printed for any failing test in the suite.
_CTEST_FAIL_BANNER = "The following tests FAILED"


@pytest.mark.gpu_count(NUM_GPUS)
@pytest.mark.layer.math_lib
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.medium
def test_quda_ctest_suite(
    target_executor,
    rock_dir: str,
    ld_path: dict,
    quda_build: str,
    quda_mpi_runtime,
):
    """Run the QUDA ctest suite and assert every test passes.

    ``target_executor`` acquires ``QUDA_NUM_GPUS`` GPUs and injects
    ``ROCR_VISIBLE_DEVICES``; QUDA's ctest harness spawns ``QUDA_TEST_NUM_PROCS``
    ranks over the configured grid, one rank per visible GPU.
    """
    build_dir = quda_build
    ld = ld_path["LD_LIBRARY_PATH"]

    mpi_home = quda_mpi_runtime.env.get("MPI_HOME", "")
    mpi_bin = os.path.dirname(quda_mpi_runtime.launcher)
    mpi_lib = quda_mpi_runtime.env.get("LD_LIBRARY_PATH", "")

    # Writable QUDA tuning cache dir. Placed inside the arch-namespaced build_dir
    # (tune params are hardware-specific) and made absolute — QUDA resolves
    # QUDA_RESOURCE_PATH relative to each test's cwd (the build dir), so a relative
    # path produced the "path ... does not exist" warning. Disabling tuning below
    # keeps the suite fast/deterministic; the resource path caches tuning if it is
    # ever re-enabled. mkdir runs in-command so it works on a remote node too
    # (target_executor may be an SshExecutor).
    tunecache = os.path.abspath(os.path.join(build_dir, "tunecache"))

    # QUDA runtime env; ROCR/HIP visible-device vars are intentionally omitted
    # (the executor injects them).
    # NOTE: the tuning variable is QUDA_ENABLE_TUNING (not QUDA_TUNING_ENABLED,
    # which QUDA silently ignores — leaving autotuning ON and causing ~1500s
    # per-test ctest timeouts). 0 disables autotuning.
    # ctest runs serially: each test uses NUM_GPUS GPUs, so parallel ctest
    # would oversubscribe the GPUs. --timeout (QUDA_CTEST_TIMEOUT) bounds any single
    # hung test and stays below the outer target_executor.run() cap below. NUM_PROCS/
    # GRID come from _workload and MUST match what the build was configured with
    # (QUDA bakes them into the CTest launch at configure time).
    cmd = (
        f"mkdir -p {tunecache} && "
        f"env MPI_HOME={mpi_home} ROCM_PATH={rock_dir} "
        f"PATH={mpi_bin}:{rock_dir}/bin:$PATH "
        f"LD_LIBRARY_PATH={mpi_lib}:{ld}:$LD_LIBRARY_PATH "
        f"QUDA_RESOURCE_PATH={tunecache} "
        f"QUDA_ENABLE_TUNING=0 QUDA_TEST_NUM_PROCS={NUM_PROCS} QUDA_ENABLE_P2P=0 "
        f"QUDA_TEST_GRID_SIZE='{GRID}' "
        f"ctest --output-on-failure --timeout {CTEST_TIMEOUT} --test-dir {build_dir}"
    )

    # ctest streams to the console live (run pytest with -s to see it) and always
    # writes a full per-run log here — tail -f it to watch a run started without -s.
    ctest_log = os.path.join(build_dir, "Testing", "Temporary", "LastTest.log")
    logger.info("QUDA ctest starting; full log -> %s (use `pytest -s` for live console output)", ctest_log)

    # runtime.medium cap: 30 min for the full ctest suite (~15 min measured, so
    # ~2x headroom for slower GPUs/variance). The one-time source build is a
    # separate session fixture with its own therock.build_timeout_secs.
    result = target_executor.run(cmd, timeout=1800.0)

    assert result.ok, (
        f"QUDA ctest suite failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
    )
    assert _CTEST_FAIL_BANNER not in result.stdout, f"QUDA ctest reported failing tests:\n{result.stdout[-4000:]}"
    assert "No tests were found" not in result.stdout and result.stdout.strip(), (
        f"QUDA ctest ran no tests — build likely produced no test targets:\n{result.stdout[-2000:]}"
    )
    assert "tests passed" in result.stdout, f"QUDA ctest did not report a pass summary:\n{result.stdout[-4000:]}"
