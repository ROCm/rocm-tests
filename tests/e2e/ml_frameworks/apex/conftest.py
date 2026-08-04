# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apex area config and fixtures.

Env-configurable run parameters and the ``apex_repo`` fixture, which clones Apex
and exposes the checkout inside the container via a bind mount.
"""

from __future__ import annotations

import os
import pathlib
import shlex

import pytest

from framework.common.workspace_layout import REMOTE_WORKSPACE_DIR

# Host output/ tree bind-mounted into the container at a fixed absolute target.
# Not $HOME: the host shell would expand it to the host user's home when building
# the docker command, which need not match the container's home.
_OUTPUT_HOST = pathlib.Path("output").resolve()
_CONTAINER_WORKSPACE = f"/mnt/{REMOTE_WORKSPACE_DIR}"
CONTAINER_MOUNT_FLAGS = f"-v {shlex.quote(str(_OUTPUT_HOST))}:{_CONTAINER_WORKSPACE}"

# Upstream ROCm Apex source tree (PyTorch fused-kernel extension).
APEX_URL = os.environ.get("APEX_URL", "https://github.com/ROCmSoftwarePlatform/apex")

# Apex "related commit" checked out by the apex_repo fixture; empty = default branch.
APEX_COMMIT = os.environ.get("APEX_COMMIT", "").strip()

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


@pytest.fixture(scope="session")
def apex_repo(external_build, compiler_build_dir: str) -> str:
    """Clone Apex once per session; return the checkout path inside the container.

    Clones at ref ``APEX_COMMIT`` (else the default branch) into the bind-mounted
    output tree, so the returned path resolves inside the container.
    """
    dest = pathlib.Path(compiler_build_dir) / "apex"
    repo = external_build.clone_repo(APEX_URL, dest, ref=APEX_COMMIT or None)
    external_build.assert_license_present(repo)  # provenance guard
    return f"{_CONTAINER_WORKSPACE}/external/{pathlib.Path(repo).name}"
