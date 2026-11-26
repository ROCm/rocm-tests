# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Constants for tests/e2e/hipfile/."""

from __future__ import annotations

import os

# hipFile is built from the ROCm/rocm-systems monorepo (projects/hipfile).
HIPFILE_MONOREPO_URL = "https://github.com/ROCm/rocm-systems.git"
HIPFILE_SUBPATH = "projects/hipfile"
HIPFILE_REF = os.environ.get("ROCM_TEST_HIPFILE_REF", "release/therock-10.2")

# Environment variable naming an NVMe-backed ext4 mount point on the target node.
NVME_PATH_ENV = "ROCM_TEST_HIPFILE_NVME_PATH"

SRC_DIR = "tests/e2e/hipfile/src"
BINARY_NAME = "hipFile_system_test"
