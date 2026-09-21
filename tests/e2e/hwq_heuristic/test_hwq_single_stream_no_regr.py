# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hwq_single_stream_no_regr.py -- HIP single-stream queue regression check.

Runs a single-stream SAXPY workload under dynamic queue modes 0, 1, and 2.
The objective is to ensure queue-management modes do not regress traditional
single-stream throughput.

Binary compiled via CMake from:
    tests/e2e/hwq_heuristic/src/hwq_single_stream_no_regr.cpp

Coverage arguments:
    --n=16777216 --passes=32 --warmup=4
"""

from __future__ import annotations

import re

import pytest

_THROUGHPUT_RE = re.compile(r"elements_per_sec=([0-9.+\-eE]+)")


def _run_single_stream_mode(target_executor, ld_path: dict, binary: str, dq_mode: int) -> float:
    """Run one dynamic queue mode and return reported elements/sec."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" DEBUG_HIP_DYNAMIC_QUEUES={dq_mode}"
        f" {binary}"
        f" --n=16777216 --passes=32 --warmup=4",
        timeout=300.0,
    )
    assert result.ok, (
        f"hwq_single_stream_no_regr dq_mode={dq_mode} failed"
        f" (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASS" in result.stdout, (
        f"hwq_single_stream_no_regr dq_mode={dq_mode}: expected PASS in stdout:\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    match = _THROUGHPUT_RE.search(result.stdout)
    assert match, (
        f"hwq_single_stream_no_regr dq_mode={dq_mode}: missing elements_per_sec metric:\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    return float(match.group(1))


@pytest.mark.runtime.medium
@pytest.mark.retry(count=1)
def test_hwq_single_stream_no_regression(
    target_executor,
    ld_path: dict,
    hwq_single_stream_no_regr_binary: str,
):
    """Verify dynamic queue modes do not regress single-stream throughput by more than 5%."""
    throughputs = {
        dq_mode: _run_single_stream_mode(target_executor, ld_path, hwq_single_stream_no_regr_binary, dq_mode)
        for dq_mode in (0, 1, 2)
    }
    baseline = throughputs[0]
    assert baseline > 0.0, f"mode 0 baseline throughput must be positive: {throughputs}"

    min_allowed = baseline * 0.95
    for dq_mode in (1, 2):
        assert throughputs[dq_mode] >= min_allowed, (
            f"dynamic queue mode {dq_mode} single-stream throughput regressed by more than 5% "
            f"vs mode 0: throughputs={throughputs}"
        )
