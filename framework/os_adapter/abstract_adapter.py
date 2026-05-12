# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
abstract_adapter.py -- Abstract OS adapter interface.

All platform-specific operations are routed through this interface so test code
and framework modules are OS-agnostic. Concrete adapters (Linux, Windows) implement
the methods below; the factory returns the correct one at runtime.
"""

from __future__ import annotations

import abc


class AbstractOsAdapter(abc.ABC):
    """OS-agnostic interface for platform-specific GPU and system operations."""

    @abc.abstractmethod
    def list_gpu_device_paths(self) -> list[str]:
        """Return device paths for all detected AMD GPU render nodes.

        Linux: ``/dev/dri/renderD*`` paths.
        Windows: Device instance IDs from ``amd-smi static --json``.

        Returns:
            List of device path strings, empty if no GPUs found.
        """

    @abc.abstractmethod
    def load_kernel_module(self, module: str) -> bool:
        """Load a kernel module (Linux: modprobe, Windows: no-op).

        Args:
            module: Module name, e.g. ``"amdgpu"``.

        Returns:
            True if the module was loaded (or already loaded).
        """

    @abc.abstractmethod
    def unload_kernel_module(self, module: str) -> bool:
        """Unload a kernel module (Linux: modprobe -r, Windows: no-op).

        Args:
            module: Module name, e.g. ``"amdgpu"``.

        Returns:
            True if unloaded successfully.
        """

    @abc.abstractmethod
    def is_wsl(self) -> bool:
        """Return True if the process is running inside Windows Subsystem for Linux."""

    @abc.abstractmethod
    def is_module_loaded(self, module: str) -> bool:
        """Return True if the kernel module *module* is currently loaded.

        Linux/WSL: reads ``lsmod`` — no elevated privileges required.
        Windows: returns ``True`` (driver presence is implied by amd-smi operating).

        Args:
            module: Kernel module name, e.g. ``"amdgpu"``.

        Returns:
            True if the module is loaded (or assumed present on Windows).
        """

    @abc.abstractmethod
    def get_platform_name(self) -> str:
        """Return a human-readable platform identifier: ``"linux"``, ``"windows"``, ``"wsl"``."""
