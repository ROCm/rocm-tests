# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
linux_adapter.py -- Linux OS adapter implementation.

Implements AbstractOsAdapter for Linux systems using /dev/kfd, /dev/dri,
and standard Linux utilities (modprobe, uname, etc.).

WSL environments are auto-detected via /proc/version and treated as a
distinct platform with partial support (no kernel module operations).
"""

from __future__ import annotations

import logging
import pathlib
import subprocess

from framework.os_adapter.abstract_adapter import AbstractOsAdapter

logger = logging.getLogger(__name__)


class LinuxOsAdapter(AbstractOsAdapter):
    """OS adapter for Linux and WSL environments."""

    def list_gpu_device_paths(self) -> list[str]:
        """Return /dev/dri/renderD* device paths for AMD GPU nodes."""
        return sorted(str(p) for p in pathlib.Path("/dev/dri").glob("renderD*"))

    def load_kernel_module(self, module: str) -> bool:
        """Load a kernel module via modprobe.

        Args:
            module: Kernel module name, e.g. ``"amdgpu"``.

        Returns:
            True if the module loaded (or was already loaded), False on error.
        """
        result = subprocess.run(
            ["modprobe", module],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "modprobe %s failed (rc=%d): %s",
                module,
                result.returncode,
                result.stderr.strip(),
            )
            return False
        logger.debug("Loaded kernel module: %s", module)
        return True

    def unload_kernel_module(self, module: str) -> bool:
        """Unload a kernel module via modprobe -r.

        Args:
            module: Kernel module name, e.g. ``"amdgpu"``.

        Returns:
            True if the module was unloaded, False on error.
        """
        result = subprocess.run(
            ["modprobe", "-r", module],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            logger.warning(
                "modprobe -r %s failed (rc=%d): %s",
                module,
                result.returncode,
                result.stderr.strip(),
            )
            return False
        logger.debug("Unloaded kernel module: %s", module)
        return True

    def is_module_loaded(self, module: str) -> bool:
        """Return True if *module* is currently loaded via lsmod (no sudo required).

        Args:
            module: Kernel module name, e.g. ``"amdgpu"``.

        Returns:
            True if the module appears in ``lsmod`` output.
        """
        result = subprocess.run(
            ["lsmod"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0:
            logger.warning("lsmod failed (rc=%d): %s", result.returncode, result.stderr.strip())
            return False
        return any(line.split()[0] == module for line in result.stdout.splitlines() if line.split())

    def is_wsl(self) -> bool:
        """Detect WSL via /proc/version."""
        try:
            version = pathlib.Path("/proc/version").read_text(encoding="utf-8").lower()
            return "microsoft" in version or "wsl" in version
        except OSError:
            return False

    def get_platform_name(self) -> str:
        """Return 'wsl' if running in WSL, otherwise 'linux'."""
        return "wsl" if self.is_wsl() else "linux"
