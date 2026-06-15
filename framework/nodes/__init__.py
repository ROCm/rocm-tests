# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.nodes -- Fleet management and GPU slot allocation.

Provides the ``NodePool`` resource manager and supporting types for
location-transparent GPU test execution across local and remote nodes.

Public API (import from here, not from sub-modules)::

    from framework.nodes import NodePool, NodeSlot, MultiGpuSlots
    from framework.nodes import NodeSpec, HostConfigLoader
    from framework.nodes import GpuFileLock

Test scheduling (xdist_group assignment + policy sort) is handled by
``framework.scheduling.dynamic_scheduler``.
"""

from framework.nodes.gpu_file_lock import GpuFileLock
from framework.nodes.node_pool import MultiGpuSlots, NodePool, NodeSlot
from framework.nodes.node_spec import HostConfigLoader, NodeSpec
from framework.nodes.pending_tracker import PendingAcquisitionTracker

__all__ = [
    "GpuFileLock",
    "HostConfigLoader",
    "MultiGpuSlots",
    "NodePool",
    "NodeSlot",
    "NodeSpec",
    "PendingAcquisitionTracker",
]
