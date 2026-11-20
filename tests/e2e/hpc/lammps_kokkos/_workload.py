# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable run matrix for the LAMMPS-Kokkos test.

Imported by both ``conftest.py`` (clone/build) and ``test_lammps_kokkos.py``
(run + markers). Every knob LAMMPS takes here -- box dimensions, step count,
neighbor-list settings, rank count -- is a *runtime* argument to ``lmp``, so one
per-arch build serves the whole matrix and the test is parametrized over it.

The matrix is a small, JSON-serializable table (``BENCHMARK_MATRIX``) keyed by
benchmark name. Each entry holds the simulation geometry and the input deck for
that benchmark; the GPU counts to sweep are separate, so the full run set is the
cross product of the two (four benchmarks x four GPU counts = 16 variants by
default).

Environment overrides:
    LAMMPS_GPU_COUNTS    Comma-separated GPU counts to run (default "1,2,4,8").
    LAMMPS_BENCHMARKS    Comma-separated subset of benchmark names to run
                         (default: every key in the matrix).
    LAMMPS_MATRIX_JSON   Path to a JSON file, or an inline JSON string, that
                         *replaces* ``BENCHMARK_MATRIX`` wholesale. Same shape.
    LAMMPS_STEPS         Override the simulation step count for every benchmark.
                         Useful for a quick smoke pass; leaves the matrix alone.
    LAMMPS_MIN_ATOM_STEPS  Optional lower bound (float, atom-steps/s). When set,
                         a variant fails if its score falls below it. Unset ->
                         the test only checks that the run produced a score.
    LAMMPS_MPI_EXTRA_ENV  Extra ``VAR=val`` pairs injected before the launcher at
                         run time (default enables the UCX-on-any-transport
                         fallback; see MPI_EXTRA_ENV below). Set empty to disable.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

# --- Benchmark matrix ----------------------------------------------------------
# One entry per LAMMPS benchmark:
#   x, y, z      -- simulation box dimensions passed as ``-v x/-v y/-v z``.
#   steps        -- simulation steps passed as ``-v steps``.
#   optimizations-- neighbor-list / Newton's-third-law settings. These are LAMMPS
#                   ``-pk kokkos`` sub-options appended to the package command.
#   input_script -- the input deck (``-in``).
#   path         -- directory, relative to the LAMMPS checkout, the deck and its
#                   potential files live in; ``lmp`` is run from there because the
#                   decks reference their potentials by relative path.
BENCHMARK_MATRIX: dict[str, dict[str, object]] = {
    "eam": {
        "x": 8,
        "y": 8,
        "z": 8,
        "steps": 300,
        "optimizations": "neigh full neigh/qeq full newton off",
        "input_script": "in.eam",
        "path": "bench",
    },
    "lj": {
        "x": 8,
        "y": 8,
        "z": 8,
        "steps": 600,
        "optimizations": "neigh full neigh/qeq full newton off",
        "input_script": "in.lj",
        "path": "bench",
    },
    "reaxff": {
        "x": 16,
        "y": 16,
        "z": 16,
        "steps": 300,
        "optimizations": "neigh half neigh/qeq full newton on",
        "input_script": "in.reaxff.hns",
        "path": "examples/reaxff/HNS",
    },
    "tersoff": {
        "x": 10,
        "y": 10,
        "z": 10,
        "steps": 300,
        "optimizations": "neigh half neigh/qeq full newton on binsize 5.6",
        "input_script": "in.tersoff",
        "path": "bench/POTENTIALS",
    },
}

# Default GPU counts to sweep; one MPI rank is launched per GPU.
_DEFAULT_GPU_COUNTS = "1,2,4,8"

# ReaxFF's QEq solver is unstable on Kokkos/HIP with GPU-aware MPI -- upstream
# reports segfaults and QEq breakdown and recommends disabling it
# (https://github.com/lammps/lammps/issues/3895). Every other benchmark keeps
# GPU-aware MPI on, which is where the multi-GPU scaling comes from.
_NON_GPU_AWARE = frozenset({"reaxff"})


class Variant(NamedTuple):
    """One parametrized LAMMPS run: a benchmark at a given GPU/rank count."""

    benchmark: str
    gpus: int
    x: int
    y: int
    z: int
    steps: int
    optimizations: str
    input_script: str
    path: str

    @property
    def is_single_gpu(self) -> bool:
        """True for the single-GPU variant (drives hw.gpu vs hw.multi_gpu)."""
        return self.gpus <= 1

    @property
    def gpu_aware(self) -> str:
        """``on``/``off`` for the Kokkos ``gpu/aware`` package option."""
        return "off" if self.benchmark in _NON_GPU_AWARE else "on"

    @property
    def label(self) -> str:
        """Stable, filesystem-safe id used for the pytest param and metrics."""
        return f"{self.benchmark}_g{self.gpus}"


def _load_matrix() -> dict[str, dict[str, object]]:
    """Return the active benchmark matrix, honoring LAMMPS_MATRIX_JSON.

    The override may be a path to a ``.json`` file or an inline JSON string; when
    unset the in-file ``BENCHMARK_MATRIX`` is used.
    """
    raw = os.environ.get("LAMMPS_MATRIX_JSON", "").strip()
    if not raw:
        return BENCHMARK_MATRIX
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(raw)


def _gpu_counts() -> list[int]:
    """Return the GPU counts to sweep, from LAMMPS_GPU_COUNTS (default 1,2,4,8)."""
    raw = os.environ.get("LAMMPS_GPU_COUNTS", _DEFAULT_GPU_COUNTS)
    return [int(tok) for tok in raw.split(",") if tok.strip()]


def _selected_benchmarks(matrix: dict[str, dict[str, object]]) -> list[str]:
    """Return the benchmark names to run, from LAMMPS_BENCHMARKS (default: all).

    Names are matched case-insensitively, so ``Eam`` and ``eam`` both select the
    same benchmark.
    """
    raw = os.environ.get("LAMMPS_BENCHMARKS", "").strip()
    if not raw:
        return list(matrix)
    by_lowercase = {name.lower(): name for name in matrix}
    selected = []
    for token in (tok.strip() for tok in raw.split(",")):
        if not token:
            continue
        name = by_lowercase.get(token.lower())
        if name is None:
            raise ValueError(f"LAMMPS_BENCHMARKS names {token!r}, which is not in the matrix: {sorted(matrix)}")
        selected.append(name)
    return selected


def _make_variant(benchmark: str, gpus: int, params: dict[str, object]) -> Variant:
    """Build a :class:`Variant` from one matrix row at *gpus* ranks."""
    steps_override = os.environ.get("LAMMPS_STEPS", "").strip()
    return Variant(
        benchmark=benchmark,
        gpus=gpus,
        x=int(params["x"]),
        y=int(params["y"]),
        z=int(params["z"]),
        steps=int(steps_override) if steps_override else int(params["steps"]),
        optimizations=str(params["optimizations"]),
        input_script=str(params["input_script"]),
        path=str(params["path"]),
    )


def variants_for() -> list[Variant]:
    """Return the ordered run matrix: selected benchmarks x selected GPU counts.

    Unlike rocHPL the geometry does not vary with the GPU count -- LAMMPS keeps
    the box fixed and decomposes it across ranks -- so this is a plain cross
    product rather than a per-count lookup.
    """
    matrix = _load_matrix()
    counts = _gpu_counts()
    variants = [
        _make_variant(benchmark, gpus, matrix[benchmark])
        for benchmark in _selected_benchmarks(matrix)
        for gpus in counts
    ]
    if not variants:
        raise ValueError(
            f"LAMMPS-Kokkos: no run variants at GPU counts {counts}; "
            "check LAMMPS_MATRIX_JSON / LAMMPS_BENCHMARKS / LAMMPS_GPU_COUNTS."
        )
    return variants


# --- Run-time knobs shared by every variant -----------------------------------

# Optional performance floor in atom-steps/s. Empty -> score is recorded only.
_min_atom_steps_raw = os.environ.get("LAMMPS_MIN_ATOM_STEPS", "").strip()
MIN_ATOM_STEPS = float(_min_atom_steps_raw) if _min_atom_steps_raw else None

# OpenMPI + UCX single-node fallback, matching the rocHPL suite.
# Override via LAMMPS_MPI_EXTRA_ENV; set it empty to disable.
_DEFAULT_MPI_EXTRA_ENV = "OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any"
MPI_EXTRA_ENV = os.environ.get("LAMMPS_MPI_EXTRA_ENV", _DEFAULT_MPI_EXTRA_ENV).strip()
