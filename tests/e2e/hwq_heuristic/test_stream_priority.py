# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_stream_priority.py -- HIP stream-priority semantics + ignore-override flag.

Builds ``hwq_stream_priority`` and validates the two behaviours the queue
heuristic must guarantee (TMS hwq_stream_priority_interaction):

- ``test_stream_priority_semantics``: with no override, the heuristic PRESERVES
  requested stream priorities -- ``hipStreamGetPriority`` echoes the range-clamped
  request, the high stream is numerically <= the low stream, and both workloads
  compute correct results.
- ``test_stream_priority_ignore_override``: with ``DEBUG_HIP_IGNORE_STREAM_PRIORITY=1``
  the override is honored gracefully -- priority streams still create and both
  workloads execute correctly.

runtime.fast is declared explicitly.
"""

import pytest


@pytest.mark.runtime.fast
def test_stream_priority_semantics(target_executor, ld_path: dict, hwq_stream_priority_binary: str):
    """Queue heuristic preserves requested HIP stream priorities."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {hwq_stream_priority_binary}")
    assert result.ok, (
        f"hwq_stream_priority (semantics) failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "priority semantics preserved" in result.stdout, f"semantics not preserved:\n{result.stdout[:2000]}"
    assert "hwq_stream_priority: PASSED" in result.stdout, f"missing PASSED sentinel:\n{result.stdout[:2000]}"


@pytest.mark.runtime.fast
def test_stream_priority_ignore_override(target_executor, ld_path: dict, hwq_stream_priority_binary: str):
    """DEBUG_HIP_IGNORE_STREAM_PRIORITY override is honored without breaking execution."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} DEBUG_HIP_IGNORE_STREAM_PRIORITY=1 {hwq_stream_priority_binary} --ignore"
    )
    assert result.ok, (
        f"hwq_stream_priority (ignore) failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "ignore-override accepted" in result.stdout, f"override not accepted cleanly:\n{result.stdout[:2000]}"
    assert "hwq_stream_priority: PASSED" in result.stdout, f"missing PASSED sentinel:\n{result.stdout[:2000]}"
