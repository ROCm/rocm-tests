# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmi_memory_partition.py -- amd-smi memory/compute partition switching.

Switches GPU memory and compute (accelerator) partitions via ``amd-smi``, reloads
the amdgpu driver on memory changes, verifies the readback against an independent
query, drives GPU workloads under each partition mode, and gates on dmesg faults.
Focused on MI350X POR values with scalable hooks for future ASICs; skips on any
other ASIC.

Runtime knobs (``key=value`` pairs, ``,``-separated) come from the
``ROCM_TEST_AMDSMI_TEST_FILTER`` environment variable, e.g.::

    ROCM_TEST_AMDSMI_TEST_FILTER="loop_count=3,workload_timeout=1200"
    ROCM_TEST_AMDSMI_TEST_FILTER="workload_filter=hipbone:transferbench"
    ROCM_TEST_AMDSMI_TEST_FILTER="workload_cmd=/opt/rocm/bin/rocm-bandwidth-test,loop_count=1"

Markers:
    hw.gpu, ci.weekly, layer.runtime, runtime.soak, os.linux, e2e.stack
"""

import os

import pytest

from tests.e2e.amd_smi.partition_utils import (
    AmdSmiMemoryPartition,
    AmdSmiMemoryPartitionChangePostWorkload,
    AmdSmiMemoryPartitionChangeThreeTimes,
    AmdSmiMemoryPartitionMultipleAmgSolve,
    AmdSmiMemoryPartitionMultipleHipbone,
    AmdSmiMemoryPartitionSingleWorkloadOnly,
    parse_test_filter,
)

_TEST_FILTER_ENV = "ROCM_TEST_AMDSMI_TEST_FILTER"

# All flows reload the amdgpu driver node-wide, so no two may run concurrently on a node.
_SERIAL_GROUP = "amdsmi_partition_serial"


def _drive(orchestrator_cls, target_executor, rock_dir, platform_name):
    """Instantiate the orchestrator, run it, and translate its outcome to pytest."""
    test_filter = parse_test_filter(os.environ.get(_TEST_FILTER_ENV, ""))
    orchestrator = orchestrator_cls(target_executor, rock_dir, platform_name, test_filter=test_filter)
    outcome = orchestrator.execute()
    if outcome.status == "SKIP":
        pytest.skip(outcome.message)
    assert outcome.status == "PASS", outcome.message


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.e2e.stack
@pytest.mark.xdist_group(_SERIAL_GROUP)
def test_amdsmi_memory_partition(target_executor, rock_dir, platform_name):
    """Toggle memory partition to POR (DPX/NPS2) and back, verifying each transition."""
    _drive(AmdSmiMemoryPartition, target_executor, rock_dir, platform_name)


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.e2e.stack
@pytest.mark.xdist_group(_SERIAL_GROUP)
def test_amdsmi_mem_partition_change_post_workload(target_executor, rock_dir, platform_name):
    """Run all workloads under baseline (SPX/NPS1) and POR (DPX/NPS2) across the toggle loop."""
    _drive(AmdSmiMemoryPartitionChangePostWorkload, target_executor, rock_dir, platform_name)


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.e2e.stack
@pytest.mark.xdist_group(_SERIAL_GROUP)
def test_amdsmi_mem_partition_change_3x(target_executor, rock_dir, platform_name):
    """Stress the partition-set path: each set+verify pair executes three times per iteration."""
    _drive(AmdSmiMemoryPartitionChangeThreeTimes, target_executor, rock_dir, platform_name)


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.e2e.stack
@pytest.mark.xdist_group(_SERIAL_GROUP)
def test_amdsmi_mem_partition_change_multiple_hipbone(target_executor, rock_dir, platform_name):
    """Partition toggle exercising the hipbone workload under each partition mode."""
    _drive(AmdSmiMemoryPartitionMultipleHipbone, target_executor, rock_dir, platform_name)


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.e2e.stack
@pytest.mark.xdist_group(_SERIAL_GROUP)
def test_amdsmi_mem_partition_change_multiple_amgsolve(target_executor, rock_dir, platform_name):
    """Partition toggle exercising the amgsolve workload under each partition mode."""
    _drive(AmdSmiMemoryPartitionMultipleAmgSolve, target_executor, rock_dir, platform_name)


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.e2e.stack
@pytest.mark.xdist_group(_SERIAL_GROUP)
def test_amdsmi_mem_partition_single_workload(target_executor, rock_dir, platform_name):
    """Partition toggle for a single workload resolved from test_filter or the class default."""
    _drive(AmdSmiMemoryPartitionSingleWorkloadOnly, target_executor, rock_dir, platform_name)
