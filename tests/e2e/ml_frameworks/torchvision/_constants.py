# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run-time constants for the TorchVision test area."""

from __future__ import annotations

import os as _os
import pathlib
import shlex

from framework.common.workspace_layout import REMOTE_WORKSPACE_DIR

# Host output/ tree bind-mounted into the container at a fixed absolute target.
_OUTPUT_HOST = pathlib.Path("output").resolve()
_CONTAINER_WORKSPACE = f"/mnt/{REMOTE_WORKSPACE_DIR}"
CONTAINER_MOUNT_FLAGS = f"-v {shlex.quote(str(_OUTPUT_HOST))}:{_CONTAINER_WORKSPACE}"

# GPU UT suite files, restricted to cuda-tagged cases; each runs as its own test.
TEST_FILES = (
    "test/test_functional_tensor.py",
    "test/test_transforms_tensor.py",
)
PYTEST_SELECTOR = "cuda"

# All GPUs on the node (hw.multi_gpu); target_executor owns the allocation.
GPU_COUNT_ARG = "all"

# Whole-suite wall-clock cap (seconds): first-run ops build + one UT run.
RUN_TIMEOUT = 14400.0

# Seconds for the in-container related_commits lookup and ops build baseline.
_RESOLVE_TIMEOUT = 120.0

# Optional overrides: set these env vars to skip the related_commits lookup.
# Example: export TORCHVISION_URL=https://github.com/pytorch/vision
#          export TORCHVISION_COMMIT=<sha>
# To find values: docker run --rm <image> cat /workspace/pytorch/related_commits
TORCHVISION_URL = _os.environ.get("TORCHVISION_URL", "").strip()
TORCHVISION_COMMIT = _os.environ.get("TORCHVISION_COMMIT", "").strip()
