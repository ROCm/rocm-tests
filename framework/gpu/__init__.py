# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.gpu -- GPU detection, NUMA-aware allocation, and per-test monitoring.

Public API (import from here, not from sub-modules)::

    from framework.gpu import GpuAllocator
    from framework.gpu import AbstractGpuDetector, GpuDetector, GpuInfo, MockGpuDetector
    from framework.gpu import GpuMetrics, GpuMonitor
    from framework.gpu import GpuDrainChecker
"""

from framework.gpu.allocator import GpuAllocator
from framework.gpu.detector import AbstractGpuDetector, GpuDetector, GpuInfo, MockGpuDetector
from framework.gpu.drain import GpuDrainChecker
from framework.gpu.monitor import GpuMetrics, GpuMonitor

__all__ = [
    "AbstractGpuDetector",
    "GpuAllocator",
    "GpuDetector",
    "GpuDrainChecker",
    "GpuInfo",
    "GpuMetrics",
    "GpuMonitor",
    "MockGpuDetector",
]
