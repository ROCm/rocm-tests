# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test registry -- each module self-registers by being imported here."""

from tests.common.gpu_monitored.workloads.cudamemtest import CudaMemtest
from tests.common.gpu_monitored.workloads.hipblaslt_bench import HipblasltBench
from tests.common.gpu_monitored.workloads.rvs_iet_stress import RvsIetStress
from tests.common.gpu_monitored.workloads.rvs_tst import RvsTst
from tests.common.gpu_monitored.workloads.transferbench import TransferBench

ALL_TESTS = [
    CudaMemtest(),
    TransferBench(),
    RvsIetStress(),
    RvsTst(),
    HipblasltBench(),
]


def get_test(name: str):
    """Find test by name (case-insensitive)."""
    name = name.lower()
    for t in ALL_TESTS:
        if t.spec.name.lower() == name:
            return t
    return None
