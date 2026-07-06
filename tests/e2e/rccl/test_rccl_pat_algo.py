# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rccl_pat_algo.py -- RCCL PAT (Parallel-Aggregated-Trees) algorithm test.

Builds the public ``rccl-tests`` perf clients, forces the PAT algorithm via
``NCCL_ALGO=PAT``, and runs ``all_gather_perf`` plus ``reduce_scatter_perf``.
Each run validates correctness under PAT.

Multi-GPU (hw.multi_gpu) via the tests/e2e/rccl profile.
"""

import os

import pytest

from framework.rocm.libs.rccl import correctness_ok, run_perf

_PAT_COLLECTIVES = (
    ("all_gather", "all_gather_perf"),
    ("reduce_scatter", "reduce_scatter_perf"),
)


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
@pytest.mark.parametrize(
    ("collective", "binary_name"),
    _PAT_COLLECTIVES,
    ids=[name for name, _ in _PAT_COLLECTIVES],
)
def test_rccl_pat_algo(
    collective: str,
    binary_name: str,
    target_executor,
    ld_path: dict,
    require_rccl,
    requested_gpu_count: int,
    rccl_tests_build: str,
):
    """PAT collective with NCCL_ALGO=PAT must pass data validation."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = os.path.join(rccl_tests_build, binary_name)
    result = run_perf(
        target_executor,
        binary,
        n_gpus=requested_gpu_count,
        extra_args="-b 1M -e 256M -f 2 -c 1",
        env={"LD_LIBRARY_PATH": ld, "NCCL_ALGO": "PAT"},
        operation=f"pat_{collective}",
    )
    assert correctness_ok(result), f"RCCL PAT {collective} failed validation:\n{result.raw_output[:3000]}"
