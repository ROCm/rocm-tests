# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable run parameters for the QUDA test.

Imported by both ``conftest.py`` (build) and ``test_quda.py`` (run + markers) so
the build-time and run-time QUDA settings can never drift: QUDA bakes the MPI
rank count and grid into the CTest test definitions at *configure* time, so the
build and the ctest run must agree on them.

Environment overrides:
    QUDA_NUM_GPUS        GPUs / MPI ranks on one node (default 2). ``1`` selects
                         single-GPU mode; ``>1`` selects multi-GPU mode.
    QUDA_TEST_GRID_SIZE  QUDA lattice grid ``"X Y Z T"``; a default is derived
                         from QUDA_NUM_GPUS. Set explicitly for GPU counts without
                         a built-in default.
    QUDA_CTEST_TIMEOUT   per-test ctest timeout in seconds (``ctest --timeout N``);
                         default 1200. Keep it below the outer run cap in
                         test_quda.py.
"""
import os

# GPUs on a single node; one MPI rank is launched per GPU.
NUM_GPUS = int(os.environ.get("QUDA_NUM_GPUS", "2"))
NUM_PROCS = NUM_GPUS

# The grid must decompose the lattice into exactly NUM_PROCS subvolumes. Defaults
# cover the common single-node layouts; other counts require QUDA_TEST_GRID_SIZE.
_DEFAULT_GRIDS = {1: "1 1 1 1", 2: "1 1 1 2", 4: "1 1 1 4", 8: "1 1 2 4"}
GRID = " ".join((os.environ.get("QUDA_TEST_GRID_SIZE") or _DEFAULT_GRIDS.get(NUM_GPUS, "")).split())

# Per-test ctest timeout (seconds). Bounds any single hung test; ctest's own
# default is 1500s. Kept below the outer target_executor.run() cap in test_quda.py
# so one slow test can't consume the whole budget.
CTEST_TIMEOUT = os.environ.get("QUDA_CTEST_TIMEOUT", "1200")

# Per-config build-tree label: a 1-rank and a 2-rank build are distinct artifacts
# (the rank count is baked into ctest at configure time), so they must not share
# a build directory.
CONFIG_LABEL = f"np{NUM_PROCS}"

# True for single-GPU mode (drives the hw.gpu vs hw.multi_gpu marker choice).
IS_SINGLE_GPU = NUM_GPUS <= 1
