# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rochpl.py -- rocHPL (High-Performance Linpack) end-to-end HPC benchmark.

Validates:
    1. rocHPL (https://github.com/ROCm/rocHPL) clones, builds via install.sh, and
       runs a full DGEMM-heavy LU factorization (Linpack) across P*Q MPI ranks
       (one rank per GPU) using the generated ``mpirun_rochpl`` launcher.
    2. The HPL residual check passes (stdout ends the solve with "PASSED").
    3. rocHPL reports a positive total GFLOPS figure (recorded as a metric); when
       ROCHPL_MIN_GFLOPS is set, GFLOPS must meet that floor.

The clone + install.sh build are handled by the session-scoped ``rochpl_build``
fixture in ``conftest.py``.  This test drives the resulting ``mpirun_rochpl`` on
the GPU node via ``target_executor`` with the MPI + ROCm runtime environment
injected as an ``env VAR=... cmd`` prefix (never via ``os.environ``).

The GPU count is configurable on a single node via ``ROCHPL_NUM_GPUS`` (default
2): ``1`` runs single-GPU mode (``hw.gpu``, 1 rank, grid 1x1), ``>1`` runs
multi-GPU mode (``hw.multi_gpu``, one rank per GPU). See ``_workload.py`` for all
env knobs (``ROCHPL_P``/``ROCHPL_Q``/``ROCHPL_N``/``ROCHPL_NB``/
``ROCHPL_ITERATIONS``/``ROCHPL_MIN_GFLOPS``).

Markers (declared explicitly; also registered as a CATEGORY_PROFILE for
tests/e2e/hpc/rochpl/):
    hw.gpu / hw.multi_gpu -- chosen from ROCHPL_NUM_GPUS (single vs multi GPU)
    gpu_count(N)   -- acquire N=ROCHPL_NUM_GPUS GPUs from one node
    layer.math_lib -- rocHPL is a GPU compute (rocBLAS/DGEMM) benchmark
    ci.weekly      -- long-running Linpack performance benchmark (from CATEGORY_PROFILE)
    e2e.stack      -- full-stack end-to-end scenario
    os.linux       -- bash/cmake/install.sh build path is Linux-only
    runtime.soak   -- a tuned HPL solve is a long, GPU-saturating run
"""

import logging
import os
import re

import pytest

from framework.reporting.allure_reporter import report_metric
from tests.e2e.hpc.rochpl._workload import (
    IS_SINGLE_GPU,
    ITERATIONS,
    MIN_GFLOPS,
    MPI_EXTRA_ENV,
    NB,
    NUM_GPUS,
    N,
    P,
    Q,
)

logger = logging.getLogger(__name__)

# hw dimension follows the configured GPU count (ROCHPL_NUM_GPUS). Declared via a
# module-level ``pytestmark`` list rather than a bare MarkDecorator global: a
# standalone MarkDecorator is callable, so pytest's collector would introspect it
# and the repo's dotted-mark __getattr__ patch would emit spurious "unknown mark"
# warnings. This hw.* overrides the hw.multi_gpu default from the CATEGORY_PROFILE.
pytestmark = [pytest.mark.hw.gpu if IS_SINGLE_GPU else pytest.mark.hw.multi_gpu]

# rocHPL prints a result row after the "T/V ... Gflops" header, e.g.:
#   WC00C2R4       45312   512     2     1        12.34      4567.8 (  2283.9)
# Columns: T/V  N  NB  P  Q  Time  Gflops ( per-GPU ). Capture total Gflops.
_FLOAT = r"[\d.]+(?:[eE][+-]?\d+)?"
_RESULT_RE = re.compile(
    rf"^W[A-Za-z0-9]+\s+\d+\s+\d+\s+\d+\s+\d+\s+{_FLOAT}\s+({_FLOAT})",
    re.MULTILINE,
)
# The solve ends with an HPL residual check line ending in PASSED or FAILED.
_FAILED_RE = re.compile(r"\bFAILED\b")
_PASSED_RE = re.compile(r"\bPASSED\b")


@pytest.mark.gpu_count(NUM_GPUS)
@pytest.mark.runtime.soak
def test_rochpl_benchmark(
    target_executor,
    rock_dir: str,
    ld_path: dict,
    rochpl_build: str,
    rochpl_mpi_runtime,
):
    """Run rocHPL across P*Q GPUs and assert the Linpack residual PASSED.

    ``target_executor`` acquires ``ROCHPL_NUM_GPUS`` GPUs and injects
    ``ROCR_VISIBLE_DEVICES``; ``mpirun_rochpl`` launches ``P*Q`` MPI ranks, one
    per visible GPU, over the ``P x Q`` process grid.
    """
    build_dir = rochpl_build
    ld = ld_path["LD_LIBRARY_PATH"]

    mpi_bin = os.path.dirname(rochpl_mpi_runtime.launcher)
    mpi_lib = rochpl_mpi_runtime.env.get("LD_LIBRARY_PATH", "")

    it_arg = f" --it {ITERATIONS}" if ITERATIONS else ""

    mpi_extra_env = f"{MPI_EXTRA_ENV} " if MPI_EXTRA_ENV else ""

    cmd = (
        f"cd {build_dir} && "
        f"env {mpi_extra_env}ROCM_PATH={rock_dir} "
        f"PATH={mpi_bin}:{rock_dir}/bin:$PATH "
        f"LD_LIBRARY_PATH={mpi_lib}:{ld}:$LD_LIBRARY_PATH "
        f"./mpirun_rochpl -P {P} -Q {Q} -N {N} --NB {NB}{it_arg} && "
        f"cat HPL.out"
    )

    logger.info("rocHPL launch: P=%d Q=%d N=%d NB=%d ranks=%d", P, Q, N, NB, NUM_GPUS)

    # runtime.soak cap: 2h for a tuned Linpack solve. The one-time source build is
    # a separate session fixture with its own therock.build_timeout_secs.
    result = target_executor.run(cmd, timeout=7200.0)

    assert result.ok, (
        f"rocHPL run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
    )
    assert not _FAILED_RE.search(result.stdout), f"rocHPL reported a FAILED residual check:\n{result.stdout[-4000:]}"
    assert _PASSED_RE.search(result.stdout), f"rocHPL did not report a PASSED residual check:\n{result.stdout[-4000:]}"

    match = _RESULT_RE.search(result.stdout)
    assert match, f"rocHPL produced no parseable GFLOPS result row:\n{result.stdout[-4000:]}"
    gflops = float(match.group(1))
    report_metric("ROCHPL_GFLOPS", gflops, "GFLOPS")
    logger.info("rocHPL total performance: %.1f GFLOPS", gflops)

    assert gflops > 0.0, f"rocHPL reported non-positive GFLOPS ({gflops}):\n{result.stdout[-2000:]}"
    if MIN_GFLOPS is not None:
        assert gflops >= MIN_GFLOPS, f"rocHPL GFLOPS {gflops:.1f} below floor ROCHPL_MIN_GFLOPS={MIN_GFLOPS:.1f}"
