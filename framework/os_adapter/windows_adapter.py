# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
windows_adapter.py -- Windows OS adapter implementation.

Implements AbstractOsAdapter for Windows systems using amd-smi for GPU
enumeration and the Windows API for privilege elevation (runas).

GPU device paths are returned as BDF (Bus:Device.Function) identifiers
from ``amd-smi static --json``, the closest Windows analogue to Linux
``/dev/dri/renderD*`` paths.
"""

from __future__ import annotations

import json
import logging
import subprocess

from framework.os_adapter.abstract_adapter import AbstractOsAdapter

logger = logging.getLogger(__name__)


class WindowsOsAdapter(AbstractOsAdapter):
    """OS adapter for Windows environments."""

    def list_gpu_device_paths(self) -> list[str]:
        """Return AMD GPU device identifiers via ``amd-smi static --json``.

        On Windows there are no ``/dev/dri/renderD*`` paths. BDF
        (Bus:Device.Function) strings from amd-smi are the closest analogue
        and are used by other amd-smi calls to target specific GPUs.

        Returns:
            List of BDF strings (e.g. ``["0000:03:00.0", "0000:04:00.0"]``),
            or ``["GPU:0", "GPU:1", ...]`` if BDF is unavailable.
            Empty list if amd-smi is not found or returns an error.
        """
        try:
            result = subprocess.run(
                ["amd-smi", "static", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError:
            logger.warning("amd-smi not found — cannot list GPU device paths on Windows")
            return []

        if result.returncode != 0:
            logger.warning(
                "amd-smi static failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return []

        try:
            devices = json.loads(result.stdout)
            if not isinstance(devices, list):
                devices = [devices]
            paths: list[str] = []
            for dev in devices:
                bdf = dev.get("bdf") or dev.get("pcie_id") or (dev.get("location") or {}).get("bdf")
                if bdf:
                    paths.append(str(bdf))
                elif "gpu" in dev:
                    paths.append(f"GPU:{dev['gpu']}")
            return paths
        except (json.JSONDecodeError, AttributeError) as exc:
            logger.warning("Failed to parse amd-smi static output: %s", exc)
            return []

    def is_module_loaded(self, module: str) -> bool:
        """Return True on Windows — driver presence is implied by amd-smi operating.

        Args:
            module: Kernel module name (ignored on Windows).

        Returns:
            Always True on Windows.
        """
        logger.debug("is_module_loaded('%s'): always True on Windows", module)
        return True

    def load_kernel_module(self, module: str) -> bool:
        """No-op on Windows (kernel modules not applicable)."""
        logger.debug("load_kernel_module('%s'): no-op on Windows", module)
        return True

    def unload_kernel_module(self, module: str) -> bool:
        """No-op on Windows."""
        logger.debug("unload_kernel_module('%s'): no-op on Windows", module)
        return True

    def is_wsl(self) -> bool:
        """Always False on native Windows."""
        return False

    def get_platform_name(self) -> str:
        """Return 'windows'."""
        return "windows"
