# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hwq_robustness.py -- HIP hardware queue robustness stress coverage.

Validates that HIP runtime queue selection remains stable while stream counts
cycle through 1, 4, 8, and 16 streams. The binary reports resident-set-size
drift so soak variants can catch host-side growth across long runs.

Binary compiled via CMake from:
    tests/e2e/hwq_heuristic/src/hwq_robustness.cpp

Coverage variants:
    * 60 seconds with 15-second phases
    * 1 hour with 600-second phases
    * 24 hours with 600-second phases
"""

import pytest


def _run_hwq_robustness(
    target_executor,
    ld_path: dict,
    hwq_robustness_binary: str,
    *,
    duration: int,
    phase_sec: int,
    timeout: float,
) -> None:
    """Run the HWQ robustness binary with dynamic queue mode enabled."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" DEBUG_HIP_DYNAMIC_QUEUES=2"
        f" {hwq_robustness_binary}"
        f" --duration={duration} --phase-sec={phase_sec}",
        timeout=timeout,
        stream=True,
    )
    assert result.ok, (
        f"hwq_robustness duration={duration} failed"
        f" (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASS" in result.stdout, (
        f"hwq_robustness duration={duration}: expected PASS in stdout:\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "starting stress loop" in result.stdout, (
        f"hwq_robustness duration={duration}: stress loop did not start:\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )


@pytest.mark.runtime.fast
def test_hwq_robustness_smoke(target_executor, ld_path: dict, hwq_robustness_binary: str):
    """Run the 60-second robustness smoke variant."""
    _run_hwq_robustness(
        target_executor,
        ld_path,
        hwq_robustness_binary,
        duration=60,
        phase_sec=15,
        timeout=180.0,
    )


@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
def test_hwq_robustness_one_hour(target_executor, ld_path: dict, hwq_robustness_binary: str):
    """Run the 1-hour robustness soak variant."""
    _run_hwq_robustness(
        target_executor,
        ld_path,
        hwq_robustness_binary,
        duration=3600,
        phase_sec=600,
        timeout=4500.0,
    )


@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
def test_hwq_robustness_full_day(target_executor, ld_path: dict, hwq_robustness_binary: str):
    """Run the 24-hour robustness soak variant."""
    _run_hwq_robustness(
        target_executor,
        ld_path,
        hwq_robustness_binary,
        duration=86400,
        phase_sec=600,
        timeout=87000.0,
    )
