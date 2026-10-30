# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_lammps_kokkos.py -- LAMMPS Kokkos/HIP molecular-dynamics benchmark.

Validates:
    1. LAMMPS (https://github.com/lammps/lammps) clones, configures with the
       Kokkos/HIP backend, and builds (see ``conftest.py``).
    2. Each benchmark in the run matrix completes a full simulation across
       ``N`` MPI ranks, one per GPU, and reports a run summary.
    3. The achieved throughput is recorded as a metric in atom-steps/s; when
       ``LAMMPS_MIN_ATOM_STEPS`` is set it must also meet that floor.

The clone + Kokkos/HIP build are handled by the session-scoped ``lammps_build``
fixture in ``conftest.py``.  This test drives the resulting ``lmp`` binary on the
GPU node via ``target_executor`` with the MPI + ROCm runtime environment injected
as an ``env VAR=... cmd`` prefix (never via ``os.environ``).

The test is *parametrized* over the cross product of benchmarks and GPU counts
(four benchmarks x ``1:2:4:8`` = 16 variants by default).
``pytest_generate_tests`` builds the
parametrization at collection time from ``_workload.variants_for()`` and attaches
the matching ``gpu_count`` and ``hw.gpu``/``hw.multi_gpu`` markers to each
variant.  See ``_workload.py`` for the matrix and all env knobs
(``LAMMPS_GPU_COUNTS`` / ``LAMMPS_BENCHMARKS`` / ``LAMMPS_MATRIX_JSON`` /
``LAMMPS_STEPS`` / ``LAMMPS_MIN_ATOM_STEPS``).

Markers (hw.* + gpu_count are applied per-variant by ``pytest_generate_tests``;
the rest come from the CATEGORY_PROFILE for tests/e2e/hpc/lammps_kokkos/):
    hw.gpu / hw.multi_gpu -- per variant (single- vs multi-GPU count)
    gpu_count(N)   -- per variant: acquire N GPUs from one node
    layer.math_lib -- LAMMPS/Kokkos is a GPU compute workload
    ci.nightly     -- third-party build + benchmark sweep (from CATEGORY_PROFILE)
    e2e.stack      -- full-stack end-to-end scenario
    os.linux       -- cmake/hipcc build path is Linux-only
    runtime.medium -- each benchmark is a few hundred steps, minutes per variant
"""

import logging
import os
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric
from tests.e2e.hpc.lammps_kokkos._workload import MIN_ATOM_STEPS, MPI_EXTRA_ENV, Variant, variants_for

logger = logging.getLogger(__name__)

# LAMMPS closes a run with a performance line, e.g.
#   Performance: 12.345 ns/day, 1.944 hours/ns, 142.857 timesteps/s, 4.571 Gatom-step/s
# The "hours/ns" column and the trailing atom-step figure are both optional: unit-
# less styles (in.lj) report "tau/day", and older releases omit atom-step/s
# entirely, which is why the atom count is parsed separately as a fallback.
_PERF_RE = re.compile(
    r"^\s*Performance:\s+[\d.]+\s+(?:ns|tau)/day,"
    r"(?:\s+[\d.]+\s+hours/ns,)?"
    r"\s+(?P<timesteps>[\d.]+)\s+timesteps/s"
    r"(?:,\s+(?P<atom_steps>[\d.]+)\s+(?P<scale>[GMk])atom-step/s)?",
    re.MULTILINE,
)
# Loop time of 7.2 on 8 procs for 300 steps with 32000 atoms
_ATOMS_RE = re.compile(
    r"^\s*Loop time of\s+[\d.]+\s+on\s+\d+\s+procs for\s+\d+\s+steps with\s+(?P<atoms>[\d.]+)\s+atoms",
    re.MULTILINE,
)
_SCALE = {"G": 1e9, "M": 1e6, "k": 1e3}

# Unit for the recorded metric: atoms advanced one timestep, per second. This is
# the only figure comparable across benchmarks -- timesteps/s alone is
# meaningless between differently sized boxes.
_METRIC_UNIT = "atom-timesteps/s"


def pytest_generate_tests(metafunc):
    """Parametrize ``variant`` over the benchmark x GPU-count matrix.

    Runs at collection time so each variant carries its own ``gpu_count`` and
    ``hw.gpu``/``hw.multi_gpu`` markers (the per-variant hw.* overrides the
    hw.multi_gpu CATEGORY_PROFILE default; the marker linter still sees hw.*
    satisfied via that profile).
    """
    if "variant" not in metafunc.fixturenames:
        return
    params = []
    for variant in variants_for():
        hw_marker = pytest.mark.hw.gpu if variant.is_single_gpu else pytest.mark.hw.multi_gpu
        params.append(
            pytest.param(
                variant,
                id=variant.label,
                marks=[pytest.mark.gpu_count(variant.gpus), hw_marker],
            )
        )
    metafunc.parametrize("variant", params)


def _score(stdout: str) -> float:
    """Return the run's throughput in atom-steps/s, or 0.0 when unparseable.

    LAMMPS's own ``atom-step/s`` figure is preferred when present; otherwise the
    score is reconstructed as ``timesteps/s * atoms``, which is what that column
    reports. Both paths are kept because the column is absent on the older
    releases this suite still has to read.
    """
    perf = _PERF_RE.search(stdout)
    if perf is None:
        return 0.0
    if perf.group("atom_steps") is not None:
        return float(perf.group("atom_steps")) * _SCALE[perf.group("scale")]
    atoms = _ATOMS_RE.search(stdout)
    if atoms is None:
        return 0.0
    return float(perf.group("timesteps")) * float(atoms.group("atoms"))


@pytest.mark.runtime.medium
def test_lammps_kokkos_benchmark(
    variant: Variant,
    target_executor,
    rock_dir: str,
    ld_path: dict,
    lammps_build: tuple[str, str],
    lammps_mpi_runtime,
    lammps_xnack_env: str,
):
    """Run one LAMMPS benchmark across the variant's GPUs and record its score.

    ``target_executor`` acquires ``variant.gpus`` GPUs and injects
    ``ROCR_VISIBLE_DEVICES``; ``mpirun`` launches one rank per visible GPU and
    Kokkos binds each rank to its own device via ``-k on g <N>``.
    """
    lammps_dir, lmp_binary = lammps_build
    ld = ld_path["LD_LIBRARY_PATH"]

    mpi_bin = os.path.dirname(lammps_mpi_runtime.launcher)
    mpi_lib = lammps_mpi_runtime.env.get("LD_LIBRARY_PATH", "")
    mpi_extra_env = f"{MPI_EXTRA_ENV} " if MPI_EXTRA_ENV else ""

    # Run from the deck's own directory: the input scripts reference their
    # potential files (Cu_u3.eam, Si.tersoff, ffield.reax.hns) by relative path.
    run_dir = os.path.join(lammps_dir, variant.path)

    # -log none keeps LAMMPS from writing into the shared checkout; the same
    # summary goes to stdout, which is what the score is parsed from.
    lammps_args = (
        f"-k on g {variant.gpus} "
        f"-sf kk "
        f"-pk kokkos gpu/aware {variant.gpu_aware} {variant.optimizations} "
        f"-v x {variant.x} -v y {variant.y} -v z {variant.z} "
        f"-v steps {variant.steps} "
        f"-in {shlex.quote(variant.input_script)} "
        f"-nocite -log none"
    )

    # Kokkos/HIP shares device buffers between ranks over IPC; this workload
    # requires the legacy IPC mode. lammps_xnack_env is non-empty only on APU
    # parts, where unified memory needs XNACK.
    cmd = (
        f"cd {shlex.quote(run_dir)} && "
        f"env {mpi_extra_env}ROCM_PATH={rock_dir} "
        f"HSA_ENABLE_IPC_MODE_LEGACY=1 {lammps_xnack_env}"
        f"PATH={mpi_bin}:{rock_dir}/bin:$PATH "
        f"LD_LIBRARY_PATH={mpi_lib}:{ld}:$LD_LIBRARY_PATH "
        f"{mpi_bin}/mpirun -np {variant.gpus} {shlex.quote(lmp_binary)} {lammps_args}"
    )

    logger.info(
        "LAMMPS launch: benchmark=%s ranks=%d box=%dx%dx%d steps=%d gpu_aware=%s",
        variant.benchmark,
        variant.gpus,
        variant.x,
        variant.y,
        variant.z,
        variant.steps,
        variant.gpu_aware,
    )

    # runtime.medium cap: 30 min per variant. The one-time source build is a
    # separate session fixture with its own therock.build_timeout_secs.
    result = target_executor.run(cmd, timeout=1800.0)

    assert result.ok, (
        f"LAMMPS {variant.label} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
    )

    score = _score(result.stdout)
    assert score > 0.0, (
        f"LAMMPS {variant.label} produced no parseable performance summary; "
        f"the run may have stopped before the final timestep:\n{result.stdout[-4000:]}"
    )

    report_metric(f"LAMMPS_ATOM_STEPS_{variant.label}", score, _METRIC_UNIT)
    logger.info("LAMMPS %s throughput: %.4g %s", variant.label, score, _METRIC_UNIT)

    if MIN_ATOM_STEPS is not None:
        assert score >= MIN_ATOM_STEPS, (
            f"LAMMPS {variant.label} scored {score:.4g} {_METRIC_UNIT}, "
            f"below floor LAMMPS_MIN_ATOM_STEPS={MIN_ATOM_STEPS:.4g}"
        )
