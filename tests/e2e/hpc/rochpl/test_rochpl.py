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
fixture in ``conftest.py`` (one build per GPU arch -- P/Q/N/NB are runtime args
to ``mpirun_rochpl``, so the binary is shared across every variant). This test
drives that launcher on the GPU node via ``target_executor`` with the MPI + ROCm
runtime environment injected as an ``env VAR=... cmd`` prefix (never via
``os.environ``).

The test is *parametrized* over an ASIC-specific run matrix: one variant per GPU
count (typically ``1:2:4:8``), each with the tuned ``P``/``Q``/``N``/``NB`` for
the target ``--gpu-arch``. ``pytest_generate_tests`` builds the parametrization at
collection time from ``_workload.variants_for(arch)`` and attaches the matching
``gpu_count`` and ``hw.gpu``/``hw.multi_gpu`` markers to each variant. See
``_workload.py`` for the matrix and all env knobs (``ROCHPL_GPU_COUNTS`` /
``ROCHPL_MATRIX_JSON`` / ``ROCHPL_NUM_GPUS`` / ``ROCHPL_ITERATIONS`` /
``ROCHPL_MIN_GFLOPS``).

Markers (hw.* + gpu_count are applied per-variant by ``pytest_generate_tests``;
the rest come from the CATEGORY_PROFILE for tests/e2e/hpc/rochpl/):
    hw.gpu / hw.multi_gpu -- per variant (single- vs multi-GPU count)
    gpu_count(N)   -- per variant: acquire N GPUs from one node
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
    ITERATIONS,
    MIN_GFLOPS,
    MPI_EXTRA_ENV,
    Variant,
    variants_for,
)

logger = logging.getLogger(__name__)

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


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize ``variant`` over the ASIC-specific 1:2:4:8 GPU-count matrix.

    Runs at collection time so the target arch is read from ``--gpu-arch`` and
    each variant carries its own ``gpu_count`` and ``hw.gpu``/``hw.multi_gpu``
    markers (the per-variant hw.* overrides the hw.multi_gpu CATEGORY_PROFILE
    default; the marker linter still sees hw.* satisfied via that profile).
    """
    if "variant" not in metafunc.fixturenames:
        return
    arch = metafunc.config.getoption("--gpu-arch", default=None)
    params = []
    for variant in variants_for(arch):
        hw_marker = pytest.mark.hw.gpu if variant.is_single_gpu else pytest.mark.hw.multi_gpu
        params.append(
            pytest.param(
                variant,
                id=variant.label,
                marks=[pytest.mark.gpu_count(variant.gpus), hw_marker],
            )
        )
    metafunc.parametrize("variant", params)


@pytest.mark.runtime.soak
def test_rochpl_benchmark(
    variant: Variant,
    target_executor,
    rock_dir: str,
    ld_path: dict,
    rochpl_build: str,
    rochpl_mpi_runtime,
):
    """Run rocHPL across the variant's P*Q GPUs and assert the residual PASSED.

    ``target_executor`` acquires ``variant.gpus`` GPUs and injects
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
        f"./mpirun_rochpl -P {variant.p} -Q {variant.q} -N {variant.n} --NB {variant.nb}{it_arg} && "
        f"cat HPL.out"
    )

    logger.info(
        "rocHPL launch: P=%d Q=%d N=%d NB=%d ranks=%d",
        variant.p,
        variant.q,
        variant.n,
        variant.nb,
        variant.gpus,
    )

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
    report_metric(f"ROCHPL_GFLOPS_{variant.label}", gflops, "GFLOPS")
    logger.info("rocHPL total performance: %.1f GFLOPS", gflops)

    assert gflops > 0.0, f"rocHPL reported non-positive GFLOPS ({gflops}):\n{result.stdout[-2000:]}"
    if MIN_GFLOPS is not None:
        assert gflops >= MIN_GFLOPS, f"rocHPL GFLOPS {gflops:.1f} below floor ROCHPL_MIN_GFLOPS={MIN_GFLOPS:.1f}"
