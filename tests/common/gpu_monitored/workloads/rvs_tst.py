# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS TST (thermal stress test)."""

from __future__ import annotations

from tests.common.gpu_monitored.workloads._rvs_based import _RvsBased
from tests.common.gpu_monitored.workloads.base import TestSpec


class RvsTst(_RvsBased):
    spec = TestSpec(
        name="rvs_tst",
        goal="RVS TST thermal stress with monitoring",
        workload_profile={"min_util": 50, "min_vram_pct": 0.5},
    )
    _conf_name = "tst_single.conf"
    _human_label = "TST"
    # Relies on the base-class generic fallback (``_gpu_only=False``):
    # upstream RVS ships a generic ``tst_single.conf`` but a per-silicon
    # copy only for a few parts (e.g. MI210), so without the fallback this
    # reported UNSUPPORTED on MI300X/MI300A/MI325X despite a usable config
    # being available.
