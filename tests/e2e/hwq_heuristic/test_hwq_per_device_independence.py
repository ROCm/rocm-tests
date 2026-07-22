# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hwq_per_device_independence.py -- HIP per-device queue independence.

Validates that HIP runtime queue management operates independently across
visible GPUs while each device runs multiple streams. Multi-GPU variants also
exercise a peer-to-peer copy before running the per-device stream workload.

Binary compiled via CMake from:
    tests/e2e/hwq_heuristic/src/hwq_per_device_independence_test.cpp

Coverage variants:
    --gpus=1 --streams=8
    --gpus=2 --streams=8 --p2p
    --gpus=8 --streams=8 --p2p
"""

import pytest


def _run_per_device_independence(
    target_executor,
    ld_path: dict,
    binary: str,
    *,
    gpu_count: int,
    p2p: bool,
) -> None:
    """Run the per-device independence binary for a specific visible GPU count."""
    ld = ld_path["LD_LIBRARY_PATH"]
    p2p_arg = " --p2p" if p2p else ""
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" DEBUG_HIP_DYNAMIC_QUEUES=2"
        f" {binary}"
        f" --gpus={gpu_count} --streams=8"
        f"{p2p_arg}",
        timeout=300.0,
    )
    assert result.ok, (
        f"hwq_per_device_independence gpu_count={gpu_count} failed"
        f" (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASS" in result.stdout, (
        f"hwq_per_device_independence gpu_count={gpu_count}: expected PASS in stdout:\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    for device_id in range(gpu_count):
        marker = f"device={device_id} streams=8"
        assert marker in result.stdout, (
            f"hwq_per_device_independence gpu_count={gpu_count}: expected '{marker}' in stdout:\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
    if p2p:
        for src in range(gpu_count - 1):
            marker = f"p2p_copy src={src} dst={src + 1}"
            assert marker in result.stdout, (
                f"hwq_per_device_independence gpu_count={gpu_count}: expected '{marker}' in stdout:\n"
                f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
            )


@pytest.mark.runtime.fast
def test_hwq_per_device_independence_single_gpu(
    target_executor,
    ld_path: dict,
    hwq_per_device_independence_binary: str,
):
    """Run the single-GPU per-device queue independence variant."""
    _run_per_device_independence(
        target_executor,
        ld_path,
        hwq_per_device_independence_binary,
        gpu_count=1,
        p2p=False,
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
def test_hwq_per_device_independence_two_gpu_p2p(
    target_executor,
    ld_path: dict,
    requested_gpu_count: int,
    hwq_per_device_independence_binary: str,
):
    """Run the 2-GPU queue independence variant with peer copy exercised."""
    _run_per_device_independence(
        target_executor,
        ld_path,
        hwq_per_device_independence_binary,
        gpu_count=requested_gpu_count,
        p2p=True,
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(8)
@pytest.mark.ci.weekly
@pytest.mark.runtime.medium
def test_hwq_per_device_independence_eight_gpu_p2p(
    target_executor,
    ld_path: dict,
    requested_gpu_count: int,
    hwq_per_device_independence_binary: str,
):
    """Run the 8-GPU queue independence variant with peer copy exercised."""
    _run_per_device_independence(
        target_executor,
        ld_path,
        hwq_per_device_independence_binary,
        gpu_count=requested_gpu_count,
        p2p=True,
    )
