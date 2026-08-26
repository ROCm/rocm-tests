# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS IET stress test (power delivery)."""

from __future__ import annotations

from tests.common.gpu_monitored.workloads._rvs_based import _RvsBased
from tests.common.gpu_monitored.workloads.base import TestSpec


class RvsIetStress(_RvsBased):
    spec = TestSpec(
        name="rvs_iet_stress",
        goal="RVS IET power delivery stress with monitoring",
        workload_profile={"min_util": 70, "min_vram_pct": 0.5},
    )
    _conf_name = "iet_stress.conf"
    _human_label = "IET"
