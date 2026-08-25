# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run-time constants for the Apex test area."""

from __future__ import annotations

import os
import pathlib
import shlex

from framework.common.workspace_layout import REMOTE_WORKSPACE_DIR

# Host output/ tree bind-mounted into the container at a fixed absolute target.
_OUTPUT_HOST = pathlib.Path("output").resolve()
_CONTAINER_WORKSPACE = f"/mnt/{REMOTE_WORKSPACE_DIR}"
CONTAINER_MOUNT_FLAGS = f"-v {shlex.quote(str(_OUTPUT_HOST))}:{_CONTAINER_WORKSPACE}"

# GPUs to use on one node: None = every GPU target_executor acquires, else the cap.
_APEX_NUM_GPUS_RAW = os.environ.get("APEX_NUM_GPUS", "").strip()
APEX_NUM_GPUS = int(_APEX_NUM_GPUS_RAW) if _APEX_NUM_GPUS_RAW else None

# Argument for @pytest.mark.gpu_count(...): explicit int when APEX_NUM_GPUS is set,
# else the "all" sentinel so target_executor reserves every GPU on the node.
GPU_COUNT_ARG = APEX_NUM_GPUS if APEX_NUM_GPUS is not None else "all"

# L0 unit-test suite location and runner inside the checkout.
L0_SUBDIR = "tests/L0"
RUN_SCRIPT = "run_rocm.sh"

# Whole-suite wall-clock cap (seconds): first-run kernel build + the L0 suite.
RUN_TIMEOUT = float(os.environ.get("APEX_RUN_TIMEOUT", "14400"))

# Upstream ROCm Apex source tree (PyTorch fused-kernel extension).
APEX_URL = os.environ.get("APEX_URL", "https://github.com/ROCmSoftwarePlatform/apex")

# Apex commit checked out by the apex_repo fixture; empty = default branch.
APEX_COMMIT = os.environ.get("APEX_COMMIT", "").strip()
