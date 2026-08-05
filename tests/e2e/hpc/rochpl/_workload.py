# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable run matrix for the rocHPL test.

Imported by both ``conftest.py`` (clone/build) and ``test_rochpl.py`` (run +
markers). rocHPL's process grid ``P x Q``, matrix order ``N``, and panel size
``NB`` are all *runtime* arguments to the generated ``mpirun_rochpl`` launcher --
the compiled binary is identical for every combination -- so a single per-arch
build serves the whole matrix and the test is parametrized over the variants.

The matrix is expressed as a small, JSON-serializable table (``ASIC_MATRIX``)
keyed by GPU architecture (ASIC) and then by GPU count. Each cell holds the
tuned ``P``/``Q``/``N``/``NB`` for that ASIC at that GPU count. ``test_rochpl.py``
selects the row for ``--gpu-arch`` at collection time and parametrizes one test
per GPU count (typically ``1:2:4:8``), attaching the matching ``gpu_count`` and
``hw.gpu``/``hw.multi_gpu`` markers to each variant.

Environment overrides:
    ROCHPL_GPU_COUNTS   Comma-separated GPU counts to run (default "1,2,4,8").
                        Each must have a row in the selected ASIC profile.
    ROCHPL_MATRIX_JSON  Path to a JSON file, or an inline JSON string, that
                        *replaces* ``ASIC_MATRIX`` wholesale. Same shape:
                        ``{"<arch>": {"<gpus>": {"p":P,"q":Q,"n":N,"nb":NB}}}``.
    ROCHPL_NUM_GPUS     Backward-compat single-variant pin. When set, exactly one
                        variant is run for this GPU count and the matrix is
                        ignored; ``ROCHPL_P``/``ROCHPL_Q``/``ROCHPL_N``/
                        ``ROCHPL_NB`` override the per-field defaults (which fall
                        back to the selected ASIC profile, then to a 1x1-style
                        derived grid). ``P * Q`` must equal ``ROCHPL_NUM_GPUS``.
    ROCHPL_P/ROCHPL_Q   Process-grid rows/cols (single-variant pin only).
    ROCHPL_N            Matrix order N (single-variant pin only). Larger N ->
                        higher GFLOPS and longer runtime; size to fit GPU VRAM.
    ROCHPL_NB           Panel/block size NB (single-variant pin only).
    ROCHPL_ITERATIONS   Optional rocHPL ``--it`` repeat count; empty -> rocHPL
                        default (a single solve).
    ROCHPL_MIN_GFLOPS   Optional lower bound (float). When set, the test fails if
                        the reported total GFLOPS is below it. Unset -> the test
                        only checks the HPL residual PASSED and that a positive
                        GFLOPS value was produced.
    ROCHPL_MPI_EXTRA_ENV  Extra ``VAR=val`` pairs injected before the launcher at
                        run time (default enables the UCX-on-any-transport
                        fallback; see MPI_EXTRA_ENV below). Set empty to disable.
"""

from __future__ import annotations

import json
import os
from typing import NamedTuple

# --- ASIC-specific run matrix --------------------------------------------------
# Keyed by GPU arch (ASIC) then by GPU count (as a string, so the table is valid
# JSON and can be dumped/loaded or replaced via ROCHPL_MATRIX_JSON). Each cell is
# the tuned rocHPL run for that ASIC at that GPU count:
#   p, q  -- MPI process grid; P * Q must equal the GPU count (one rank per GPU).
#   n     -- matrix order N; scaled to the ASIC's aggregate HBM so the panel fits.
#   nb    -- panel/block size NB; 512 is the rocBLAS-DGEMM-friendly tile on CDNA.
# N values are conservative starting points sized to fit VRAM with headroom; tune
# per fleet via ROCHPL_MATRIX_JSON without touching code. Unlisted arches fall
# back to the "default" profile.
ASIC_MATRIX: dict[str, dict[str, dict[str, int]]] = {
    "gfx942": {  # MI300X-class, large HBM3
        "1": {"p": 1, "q": 1, "n": 64512, "nb": 512},
        "2": {"p": 2, "q": 1, "n": 90112, "nb": 512},
        "4": {"p": 2, "q": 2, "n": 129024, "nb": 512},
        "8": {"p": 2, "q": 4, "n": 181248, "nb": 512},
    },
    "gfx90a": {  # MI200-class
        "1": {"p": 1, "q": 1, "n": 45312, "nb": 512},
        "2": {"p": 2, "q": 1, "n": 64512, "nb": 512},
        "4": {"p": 2, "q": 2, "n": 90112, "nb": 512},
        "8": {"p": 2, "q": 4, "n": 129024, "nb": 512},
    },
    "default": {  # any other/unknown arch: modest, VRAM-safe sizes
        "1": {"p": 1, "q": 1, "n": 45312, "nb": 512},
        "2": {"p": 2, "q": 1, "n": 45312, "nb": 512},
        "4": {"p": 2, "q": 2, "n": 64512, "nb": 512},
        "8": {"p": 2, "q": 4, "n": 90112, "nb": 512},
    },
}

# Default GPU counts to sweep. P * Q of each selected row must equal the count.
_DEFAULT_GPU_COUNTS = "1,2,4,8"

# Fallback process grids when a single-variant pin (ROCHPL_NUM_GPUS) names a GPU
# count with no ASIC-profile row and no explicit ROCHPL_P/ROCHPL_Q.
_DEFAULT_GRIDS: dict[int, tuple[int, int]] = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (2, 4)}

# Fallback problem/panel size when neither the profile nor env supplies one.
_FALLBACK_N = 45312
_FALLBACK_NB = 512


class Variant(NamedTuple):
    """One parametrized rocHPL run: a GPU count and its tuned P/Q/N/NB."""

    gpus: int
    p: int
    q: int
    n: int
    nb: int

    @property
    def is_single_gpu(self) -> bool:
        """True for the single-GPU variant (drives hw.gpu vs hw.multi_gpu)."""
        return self.gpus <= 1

    @property
    def label(self) -> str:
        """Stable, filesystem-safe id used for the pytest param and metrics."""
        return f"g{self.gpus}_p{self.p}q{self.q}_n{self.n}_nb{self.nb}"


def _load_matrix() -> dict[str, dict[str, dict[str, int]]]:
    """Return the active ASIC matrix, honoring the ROCHPL_MATRIX_JSON override.

    The override may be a path to a ``.json`` file or an inline JSON string; when
    unset the in-file ``ASIC_MATRIX`` is used.
    """
    raw = os.environ.get("ROCHPL_MATRIX_JSON", "").strip()
    if not raw:
        return ASIC_MATRIX
    if os.path.isfile(raw):
        with open(raw, encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(raw)


def _gpu_counts() -> list[int]:
    """Return the GPU counts to sweep, from ROCHPL_GPU_COUNTS (default 1,2,4,8)."""
    raw = os.environ.get("ROCHPL_GPU_COUNTS", _DEFAULT_GPU_COUNTS)
    return [int(tok) for tok in raw.split(",") if tok.strip()]


def _profile_for(arch: str | None) -> dict[str, dict[str, int]]:
    """Return the per-GPU-count rows for *arch*, falling back to ``default``."""
    matrix = _load_matrix()
    return matrix.get(arch or "", matrix.get("default", {}))


def _make_variant(gpus: int, params: dict[str, int]) -> Variant:
    """Build and validate a :class:`Variant` (enforces P * Q == gpus)."""
    variant = Variant(
        gpus=gpus,
        p=int(params["p"]),
        q=int(params["q"]),
        n=int(params["n"]),
        nb=int(params["nb"]),
    )
    if variant.p * variant.q != variant.gpus:
        raise ValueError(
            f"rocHPL variant grid P*Q ({variant.p}*{variant.q}={variant.p * variant.q}) "
            f"must equal the GPU count ({variant.gpus}); fix ASIC_MATRIX/ROCHPL_MATRIX_JSON."
        )
    return variant


def _env_pinned_variant(arch: str | None) -> Variant:
    """Return the single variant pinned by ROCHPL_NUM_GPUS + ROCHPL_P/Q/N/NB.

    Per-field defaults come from the selected ASIC profile row for this GPU count
    when present, then from a derived grid / fallback sizes.
    """
    gpus = int(os.environ["ROCHPL_NUM_GPUS"])
    grid_p, grid_q = _DEFAULT_GRIDS.get(gpus, (gpus, 1))
    tuned = _profile_for(arch).get(str(gpus), {})
    params = {
        "p": int(os.environ.get("ROCHPL_P", str(tuned.get("p", grid_p)))),
        "q": int(os.environ.get("ROCHPL_Q", str(tuned.get("q", grid_q)))),
        "n": int(os.environ.get("ROCHPL_N", str(tuned.get("n", _FALLBACK_N)))),
        "nb": int(os.environ.get("ROCHPL_NB", str(tuned.get("nb", _FALLBACK_NB)))),
    }
    return _make_variant(gpus, params)


def variants_for(arch: str | None) -> list[Variant]:
    """Return the ordered list of run variants for *arch*.

    When ``ROCHPL_NUM_GPUS`` is set, a single pinned variant is returned
    (backward-compatible with the original scalar env knobs). Otherwise the
    selected ASIC profile is swept across ``ROCHPL_GPU_COUNTS`` (default 1,2,4,8),
    skipping counts the profile does not define.
    """
    if os.environ.get("ROCHPL_NUM_GPUS", "").strip():
        return [_env_pinned_variant(arch)]

    profile = _profile_for(arch)
    variants: list[Variant] = []
    for gpus in _gpu_counts():
        row = profile.get(str(gpus))
        if row is None:
            continue
        variants.append(_make_variant(gpus, row))

    if not variants:
        raise ValueError(
            f"rocHPL: no run variants for arch={arch!r} at GPU counts {_gpu_counts()}; "
            "check ASIC_MATRIX / ROCHPL_MATRIX_JSON / ROCHPL_GPU_COUNTS."
        )
    return variants


# --- Run-time knobs shared by every variant -----------------------------------

# Optional fixed iteration count (rocHPL --it). Empty -> rocHPL default.
ITERATIONS = os.environ.get("ROCHPL_ITERATIONS", "").strip()

# Optional performance floor in GFLOPS. Empty -> correctness-only gate.
_min_gflops_raw = os.environ.get("ROCHPL_MIN_GFLOPS", "").strip()
MIN_GFLOPS = float(_min_gflops_raw) if _min_gflops_raw else None

# OpenMPI + UCX single-node fallback.
# Override via ROCHPL_MPI_EXTRA_ENV; set it empty to disable.
_DEFAULT_MPI_EXTRA_ENV = "OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any"
MPI_EXTRA_ENV = os.environ.get("ROCHPL_MPI_EXTRA_ENV", _DEFAULT_MPI_EXTRA_ENV).strip()
