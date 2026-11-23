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
    # Upstream RVS ships a generic ``tst_single.conf`` but a per-silicon copy
    # only for a few parts (e.g. MI210). The qualification matrix names which
    # of the two each GPU was qualified with, so parts like MI300X and MI325X
    # resolve to the generic config rather than reporting UNSUPPORTED.
