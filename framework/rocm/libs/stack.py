# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
stack.py -- ROCm stack summary and cross-cutting version helpers.

Combines ROCm version, HIP runtime, driver state, and GPU count into a single
``RocmStackSummary`` for test assertions or Allure attachments.  Version gating
helpers (``require_rocm_version``) live in ``framework.rocm.libs.hip`` — import them
from there for the canonical implementation.

Usage::

    from framework.rocm.libs.stack import stack_summary, get_rocm_version, is_driver_loaded

    def test_stack_ready(cpu_executor):
        summary = stack_summary(cpu_executor)
        assert summary.driver_loaded, "amdgpu driver not loaded"
        assert summary.device_count >= 1
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)


@dataclass
class RocmStackSummary:
    """High-level snapshot of the ROCm software stack on the target node.

    Attributes:
        rocm_version:   ROCm release version string (e.g. ``"6.3.0"``).
        hip_version:    HIP runtime version string.
        driver_version: amdgpu kernel driver version.
        driver_loaded:  True if the amdgpu kernel module is loaded.
        device_count:   Number of AMD GPU devices visible.
    """

    rocm_version: str | None = None
    hip_version: str | None = None
    driver_version: str | None = None
    driver_loaded: bool = False
    device_count: int = 0


def get_rocm_version(executor: AbstractExecutor) -> str | None:
    """Detect the installed ROCm release version string.

    Tries ``/opt/rocm/.info/version``, then ``rocminfo``, then ``hipconfig``.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        Version string (e.g. ``"6.3.0"``), or None if not detectable.
    """
    # Strategy 1: version file (most reliable, exists in ROCm >= 5.0)
    result = executor.run("cat /opt/rocm/.info/version 2>/dev/null")
    if result.ok and result.stdout.strip():
        return result.stdout.strip()

    # Strategy 2: rocminfo header
    result = executor.run("rocminfo 2>/dev/null | grep -i 'ROCm Runtime Version'")
    if result.ok and result.stdout.strip():
        m = re.search(r"(\d+\.\d+[\.\d]*)", result.stdout)
        if m:
            return m.group(1)

    # Strategy 3: hipconfig
    result = executor.run("hipconfig --version 2>/dev/null")
    if result.ok and result.stdout.strip():
        m = re.search(r"(\d+\.\d+[\.\d]*)", result.stdout)
        if m:
            return m.group(1)

    return None


def is_driver_loaded(executor: AbstractExecutor) -> bool:
    """Return True if the amdgpu kernel module is loaded on the executor host.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        True if ``lsmod | grep amdgpu`` reports a match.
    """
    result = executor.run("lsmod 2>/dev/null | grep -c amdgpu")
    try:
        return result.ok and int(result.stdout.strip()) > 0
    except (ValueError, AttributeError):
        return False


def get_driver_version(executor: AbstractExecutor) -> str | None:
    """Return the amdgpu kernel driver version string.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        Driver version string (e.g. ``"6.7.8"``), or None if unavailable.
    """
    result = executor.run("cat /sys/module/amdgpu/version 2>/dev/null || modinfo amdgpu 2>/dev/null | grep '^version:'")
    if result.ok and result.stdout.strip():
        m = re.search(r"(\d+[\.\d]+)", result.stdout)
        return m.group(1) if m else result.stdout.strip()
    return None


def stack_summary(executor: AbstractExecutor) -> RocmStackSummary:
    """Return a high-level snapshot of the ROCm stack on the executor host.

    Combines ROCm version, HIP version, driver state, and GPU count into a
    single object for use in test assertions or Allure attachments.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        RocmStackSummary dataclass with all available fields populated.

    Example::

        def test_stack_ready(cpu_executor, allure_reporter):
            summary = stack_summary(cpu_executor)
            allure_reporter.attach(str(summary), name="rocm_stack_summary")
            assert summary.driver_loaded, "amdgpu not loaded"
            assert summary.device_count >= 1
    """
    from framework.rocm.libs.hip import get_device_count, hip_version  # pylint: disable=import-outside-toplevel

    return RocmStackSummary(
        rocm_version=get_rocm_version(executor),
        hip_version=hip_version(executor),
        driver_version=get_driver_version(executor),
        driver_loaded=is_driver_loaded(executor),
        device_count=get_device_count(executor),
    )
