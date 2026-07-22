# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable run parameters for the Apex L0 test.

The source checkout, kernel build, and L0 run all happen inside a single command
executed by ``target_executor`` (see ``test_apex.py``).  Running everything
through one executor keeps the test correct across all environment profiles --
local bare-metal, remote SSH, and Docker container -- because the source tree is
always created wherever the GPU command runs, with no host<->container path
mismatch and nothing to volume-mount.

The Apex commit to check out (the "related commit") is required input:

    * On bare-metal there is no prebuilt PyTorch tree, so the commit must be
      supplied by the user via ``APEX_COMMIT``.
    * In a prebuilt-PyTorch container it is looked up from the image's
      ``related_commits`` manifest (or overridden with ``APEX_COMMIT``).

Environment overrides:
    APEX_URL             Git URL of the Apex source tree (default: the upstream
                         ROCm Apex repository).
    APEX_COMMIT          Apex "related commit" id to check out. Required on
                         bare-metal; optional in container mode where it overrides
                         the value read from the ``related_commits`` manifest.
    APEX_RELATED_COMMITS Explicit path to the ``related_commits`` manifest inside
                         a prebuilt-PyTorch container. When empty, well-known
                         locations are probed with a bounded ``find`` fallback.
    APEX_NUM_GPUS        GPUs to use on one node. Unset (default) uses EVERY GPU
                         the node/container exposes, so the tensor/pipeline-parallel
                         and multi-device L0 tests run instead of skipping. Set an
                         integer to cap it (e.g. ``APEX_NUM_GPUS=1`` for single-GPU).
    APEX_WORK_DIR        Writable scratch dir for the checkout + build (default
                         ``/tmp/rocm-tests/apex``). Must be writable both on the
                         bare-metal node and inside the container image.
    APEX_RUN_TIMEOUT     Wall-clock cap for clone + build + the whole L0 suite in
                         seconds. The suite compiles the fused kernels on first run
                         and then executes hundreds of correctness checks, so the
                         default is generous.
"""

from __future__ import annotations

import os

# Upstream public Apex source tree (PyTorch fused-kernel extension for ROCm).
APEX_URL = os.environ.get("APEX_URL", "https://github.com/ROCmSoftwarePlatform/apex")

# User-supplied Apex "related commit" id. Required on bare-metal; optional in
# container mode, where it overrides the value read from the image's
# related_commits manifest.
APEX_COMMIT = os.environ.get("APEX_COMMIT", "").strip()

# Optional explicit path to the related_commits manifest inside a prebuilt PyTorch
# container. When empty, well-known locations are probed with a bounded find(1).
RELATED_COMMITS_PATH = os.environ.get("APEX_RELATED_COMMITS", "").strip()

# GPUs to use on one node. ``None`` means "every GPU the node/container exposes"
# (resolved at run time); an integer caps the count.
_APEX_NUM_GPUS_RAW = os.environ.get("APEX_NUM_GPUS", "").strip()
APEX_NUM_GPUS = int(_APEX_NUM_GPUS_RAW) if _APEX_NUM_GPUS_RAW else None

# Argument for ``@pytest.mark.gpu_count(...)``: an explicit int when the user pins
# APEX_NUM_GPUS, else the framework "all" sentinel so the bare-metal acquisition
# path reserves every GPU on the node. (Container mode ignores gpu_count and
# controls visibility via ROCR_VISIBLE_DEVICES in the run command -- see
# test_apex.py.)
GPU_COUNT_ARG = APEX_NUM_GPUS if APEX_NUM_GPUS is not None else "all"

# Writable scratch dir for the checkout + kernel build. Must be writable on the
# bare-metal node and inside the container image alike.
WORK_DIR = os.environ.get("APEX_WORK_DIR", "/tmp/rocm-tests/apex")

# Location of the L0 unit-test suite and its runner inside the checkout.
L0_SUBDIR = "tests/L0"
RUN_SCRIPT = "run_rocm.sh"

# Whole-workflow wall-clock cap (seconds): clone + first-run kernel build + suite.
RUN_TIMEOUT = float(os.environ.get("APEX_RUN_TIMEOUT", "14400"))
