# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Test registry -- workload profiles for monitoring validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TestSpec:
    name: str
    goal: str
    workload_profile: Optional[dict] = None


ALL_TESTS = [
    TestSpec(
        name="gpu_rvs_tst_monitored",
        goal="RVS TST thermal stress with monitoring",
        workload_profile={"min_util": 10, "min_vram_pct": 0.5, "serial": True},
    ),
    TestSpec(
        name="gpu_rvs_iet_stress_monitored",
        goal="RVS IET power delivery stress with monitoring",
        workload_profile={"min_util": 70, "min_vram_pct": 0.5},
    ),
    TestSpec(
        name="gpu_hipblaslt_bench_monitored",
        goal="hipBLASLt GEMM perf sweep with monitoring",
        workload_profile={"min_util": 30, "min_vram_pct": 0, "serial": True},
    ),
]


def get_test(name: str) -> Optional[TestSpec]:
    """Find test by name (case-insensitive)."""
    name = name.lower()
    for t in ALL_TESTS:
        if t.name.lower() == name:
            return t
    return None
