# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.os_adapter -- Cross-platform OS abstraction layer.

Unifies Linux and Windows GPU enumeration and platform-specific operations
behind a single interface so test code never contains ``sys.platform`` checks.

Modules:
    abstract_adapter  -- AbstractOsAdapter base class.
    linux_adapter     -- Linux implementation (/dev/kfd, modprobe, etc.).
    windows_adapter   -- Windows implementation (amd-smi, runas, etc.).

Usage:
    from framework.os_adapter import os_adapter_factory

    adapter = os_adapter_factory()
    gpu_paths = adapter.list_gpu_device_paths()
    adapter.load_kernel_module("amdgpu")
"""

from __future__ import annotations

import sys

from framework.os_adapter.abstract_adapter import AbstractOsAdapter


def os_adapter_factory() -> AbstractOsAdapter:
    """Return the platform-appropriate OS adapter.

    Returns:
        LinuxOsAdapter on Linux/WSL, WindowsOsAdapter on Windows.

    Raises:
        RuntimeError: If the platform is not supported.
    """
    if sys.platform.startswith("linux"):
        from framework.os_adapter.linux_adapter import LinuxOsAdapter  # pylint: disable=import-outside-toplevel

        return LinuxOsAdapter()
    if sys.platform == "win32":
        from framework.os_adapter.windows_adapter import WindowsOsAdapter  # pylint: disable=import-outside-toplevel

        return WindowsOsAdapter()
    raise RuntimeError(f"Unsupported platform: {sys.platform}")
