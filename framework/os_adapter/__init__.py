# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.os_adapter -- OS-level GPU enumeration and kernel module interface.

AbstractOsAdapter: unified interface for Linux and Windows.
LinuxOsAdapter: uses lspci for GPU paths, modprobe for kernel modules.
Factory: get_os_adapter() returns the correct adapter for the current platform.
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
