# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging

import pytest

from framework.rocm.libs.gtest import run_gtest
from tests.e2e.hpc.ucx._workload import (
    GTEST_FILTER,
    GTEST_TIMEOUT,
    NUM_SHARDS,
    OMP_NUM_THREADS,
    SHARD_IDS,
)

logger = logging.getLogger(__name__)


@pytest.mark.runtime.medium
@pytest.mark.parametrize("shard_index", SHARD_IDS, ids=lambda i: f"shard{i}of{NUM_SHARDS}")
def test_ucx_rocm_gtest_suite(shard_index, target_executor, ld_path, ucx_build, ucx_gtest_binary):
    ucx_ld = f"{ucx_build}/ucx/lib:{ld_path['LD_LIBRARY_PATH']}"
    logger.info("UCX gtest shard %d/%d starting", shard_index, NUM_SHARDS)
    result = run_gtest(
        target_executor,
        ucx_gtest_binary,
        gtest_filter=GTEST_FILTER,
        shard_index=shard_index,
        total_shards=NUM_SHARDS,
        env={"LD_LIBRARY_PATH": ucx_ld},
        cwd=ucx_build,
        omp_num_threads=OMP_NUM_THREADS,
        timeout=GTEST_TIMEOUT,
    )
    logger.info(
        "UCX gtest shard %d/%d done: total=%d passed=%d failed=%d exit=%d",
        shard_index,
        NUM_SHARDS,
        result.total,
        result.passed,
        result.failed,
        result.exit_code,
    )
    label = f"shard {shard_index}/{NUM_SHARDS}"
    assert (
        result.ok
    ), f"UCX gtest {label} failed (exit={result.exit_code}, failed={result.failed})\n{result.raw_output[-4000:]}"
    assert result.failed == 0, f"UCX gtest {label}: {result.failed} failing test(s)\n{result.raw_output[-4000:]}"
    assert result.total > 0, f"UCX gtest {label} ran no tests (filter={GTEST_FILTER!r})\n{result.raw_output[-2000:]}"
