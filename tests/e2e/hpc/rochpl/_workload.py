# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable run parameters for the rocHPL test.

Imported by both ``conftest.py`` (clone/build) and ``test_rochpl.py`` (run +
markers) so the build-time and run-time settings can never drift: the MPI rank
count is the process grid ``P x Q``, which must equal the number of GPUs the
executor acquires, so the ``gpu_count`` marker and the ``mpirun_rochpl -P -Q``
launch must agree.

Environment overrides:
    ROCHPL_NUM_GPUS    GPUs / MPI ranks on one node (default 2). ``1`` selects
                       single-GPU mode; ``>1`` selects multi-GPU mode. Must equal
                       ``ROCHPL_P * ROCHPL_Q``.
    ROCHPL_P           Process-grid rows P (default derived from NUM_GPUS).
    ROCHPL_Q           Process-grid cols Q (default derived from NUM_GPUS).
    ROCHPL_N           Matrix order N (default 45312). Larger N -> higher GFLOPS
                       and longer runtime; size to fit total GPU VRAM.
    ROCHPL_NB          Panel/block size NB (default 512).
    ROCHPL_ITERATIONS  Optional rocHPL ``--it`` repeat count; empty -> rocHPL
                       default (a single solve).
    ROCHPL_REF         Git branch/tag/commit of rocHPL to build (default "main").
    ROCHPL_MIN_GFLOPS  Optional lower bound (float). When set, the test fails if
                       the reported total GFLOPS is below it. Unset -> the test
                       only checks the HPL residual PASSED and that a positive
                       GFLOPS value was produced.
    ROCHPL_MPI_EXTRA_ENV  Extra ``VAR=val`` pairs injected before the launcher at
                       run time (default enables the UCX-on-any-transport fallback;
                       see MPI_EXTRA_ENV below). Set empty to disable.
"""

import os

# GPUs on a single node; one MPI rank is launched per GPU (P * Q ranks total).
NUM_GPUS = int(os.environ.get("ROCHPL_NUM_GPUS", "2"))

# Default process grids per single-node GPU count. P * Q must equal NUM_GPUS.
_DEFAULT_GRIDS = {1: (1, 1), 2: (2, 1), 4: (2, 2), 8: (2, 4)}
_P_default, _Q_default = _DEFAULT_GRIDS.get(NUM_GPUS, (NUM_GPUS, 1))
P = int(os.environ.get("ROCHPL_P", str(_P_default)))
Q = int(os.environ.get("ROCHPL_Q", str(_Q_default)))

# Problem size (matrix order) and panel/block size.
N = int(os.environ.get("ROCHPL_N", "45312"))
NB = int(os.environ.get("ROCHPL_NB", "512"))

# Optional fixed iteration count (rocHPL --it). Empty -> rocHPL default.
ITERATIONS = os.environ.get("ROCHPL_ITERATIONS", "").strip()

# Optional performance floor in GFLOPS. Empty -> correctness-only gate.
_min_gflops_raw = os.environ.get("ROCHPL_MIN_GFLOPS", "").strip()
MIN_GFLOPS = float(_min_gflops_raw) if _min_gflops_raw else None

# OpenMPI + UCX single-node fallback.
# Override via ROCHPL_MPI_EXTRA_ENV; set it empty to disable.
_DEFAULT_MPI_EXTRA_ENV = "OMPI_MCA_pml_ucx_tls=any OMPI_MCA_pml_ucx_devices=any"
MPI_EXTRA_ENV = os.environ.get("ROCHPL_MPI_EXTRA_ENV", _DEFAULT_MPI_EXTRA_ENV).strip()

# Per-config build label: a 2x1 and a 1x1 build target distinct GPU counts, so
# they must not share a clone/build tree.
CONFIG_LABEL = f"p{P}q{Q}"

# True for single-GPU mode (drives the hw.gpu vs hw.multi_gpu marker choice).
IS_SINGLE_GPU = NUM_GPUS <= 1
