# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os

UCX_GIT_URL = os.environ.get("UCX_GIT_URL", "https://github.com/openucx/ucx")
UCX_GIT_REF = os.environ.get("UCX_GIT_REF", "v1.21.x")
GTEST_FILTER = os.environ.get("UCX_GTEST_FILTER", "*rocm*")
NUM_SHARDS = max(1, int(os.environ.get("UCX_GTEST_SHARDS", "1")))
SHARD_IDS = tuple(range(NUM_SHARDS))
_OMP = os.environ.get("UCX_OMP_NUM_THREADS", "").strip()
OMP_NUM_THREADS = int(_OMP) if _OMP.isdigit() else None
GTEST_TIMEOUT = float(os.environ.get("UCX_GTEST_TIMEOUT", "1800"))
