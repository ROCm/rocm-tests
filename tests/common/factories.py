# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
factories.py -- Test data factories for framework unit tests.

Provides factory functions that build lightweight fake objects (GpuInfo,
ExecutionResult) without touching real hardware or running subprocesses.
Import these in test files instead of constructing objects manually.

Usage:
    from tests.common.factories import fake_gpu_info, fake_execution_result

    gpu = fake_gpu_info(arch="gfx1100", vram_mb=16384)
    result = fake_execution_result(exit_code=0, stdout="RESULT_OK\\nMETRIC=42.0\\n")
"""

from __future__ import annotations

from framework.common.helpers import ExecutionResult
from framework.gpu.detector import GpuInfo


def fake_gpu_info(
    index: int = 0,
    arch: str = "gfx942",
    vram_mb: int = 32768,
    numa_node: int = 0,
) -> GpuInfo:
    """Create a synthetic GpuInfo for tests that don't need real hardware.

    Args:
        index:     GPU ordinal (default: 0).
        arch:      GFX architecture string (default: ``"gfx942"``).
        vram_mb:   VRAM in MB (default: 32768 = 32 GB).
        numa_node: NUMA node (default: 0).

    Returns:
        Immutable GpuInfo with the specified fields.
    """
    return GpuInfo(index=index, arch=arch, vram_mb=vram_mb, numa_node=numa_node)


def fake_execution_result(
    exit_code: int = 0,
    stdout: str = "RESULT_OK\n",
    stderr: str = "",
    duration: float = 0.1,
) -> ExecutionResult:
    """Create a synthetic ExecutionResult for fixture and framework unit tests.

    Args:
        exit_code: Shell exit code (default: 0 = success).
        stdout:    Captured stdout (default: ``"RESULT_OK\\n"``).
        stderr:    Captured stderr (default: empty).
        duration:  Wall-clock seconds (default: 0.1).

    Returns:
        Immutable ExecutionResult with the specified fields.
    """
    return ExecutionResult(exit_code=exit_code, stdout=stdout, stderr=stderr, duration=duration)
