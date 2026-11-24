# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deployment sizing derived from the GPUs the test actually acquired."""

from __future__ import annotations

from dataclasses import dataclass

from tests.e2e.ml_frameworks.vllm_mooncake._constants import (
    MODEL_WEIGHTS_GB,
    TENSOR_PARALLEL_OVERRIDE,
    WEIGHT_BUDGET_FRAC,
)

ROLES = 2


@dataclass(frozen=True)
class DeploymentPlan:
    """Prefill/decode layout for one node.

    Attributes:
        arch: GPU architecture name of the first acquired GPU.
        gpu_count: GPUs visible to the test.
        vram_gb: Smallest per-GPU VRAM in GiB.
        tp: Tensor-parallel size of each vLLM instance.
        instances_per_role: Prefill instances, equal to the number of decode instances.
    """

    arch: str
    gpu_count: int
    vram_gb: float
    tp: int
    instances_per_role: int


def min_tensor_parallel(vram_gb: float) -> int:
    """Return the smallest power-of-two tensor-parallel size whose weight shard fits the budget."""
    tp = 1
    while MODEL_WEIGHTS_GB / tp > vram_gb * WEIGHT_BUDGET_FRAC:
        tp *= 2
    return tp


def plan_deployment(arch: str, gpu_count: int, vram_gb: float) -> DeploymentPlan:
    """Size the deployment for the acquired GPUs.

    Raises:
        ValueError: When the GPUs cannot host one prefill and one decode instance.
    """
    tp = TENSOR_PARALLEL_OVERRIDE or min_tensor_parallel(vram_gb)
    instances = gpu_count // (ROLES * tp)
    if instances < 1:
        raise ValueError(
            f"{MODEL_WEIGHTS_GB:.0f} GB of weights on {gpu_count} x {vram_gb:.0f} GiB {arch} needs "
            f"tensor-parallel {tp}, i.e. at least {ROLES * tp} GPUs"
        )
    return DeploymentPlan(arch, gpu_count, vram_gb, tp, instances)
