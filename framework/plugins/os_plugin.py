# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
os_plugin.py -- Pytest plugin for OS-aware test gating and platform fixtures.

Responsibilities:
    - Detect the current host platform (``"linux"`` or ``"windows"``) via the
      OS adapter factory.
    - Provide ``os_adapter`` and ``platform_name`` session fixtures so tests can
      perform platform-specific operations without duplicating ``sys.platform``
      checks.
    - Hook ``pytest_runtest_setup`` to automatically skip tests whose
      ``os.*`` marker does not match the running platform.

SUPPORTED PLATFORMS:
    - ``linux``   — bare-metal or VM Linux.
    - ``windows`` — native Windows with ROCm for Windows.

Platform detection uses ``framework.os_adapter.os_adapter_factory()``,
which returns ``LinuxOsAdapter`` on ``sys.platform.startswith("linux")`` and
``WindowsOsAdapter`` on ``sys.platform == "win32"``.

MARKER SEMANTICS:
    @pytest.mark.os.linux   — run only on Linux; skip on Windows.
    @pytest.mark.os.windows — run only on Windows; skip on Linux.
    @pytest.mark.os.both    — always runs (explicit cross-platform).
    (no os.* marker)        — treated as platform-agnostic; always runs.

Loaded automatically via ``pytest_plugins`` in ``conftest.py``.
"""

from __future__ import annotations

import logging

import pytest

from framework.os_adapter import os_adapter_factory

logger = logging.getLogger(__name__)

# Evaluated once per session at module import time — cheapest possible check.
_PLATFORM: str = os_adapter_factory().get_platform_name()


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip tests whose ``os.*`` marker does not match the current platform.

    Only ``os.linux``, ``os.windows``, and ``os.both`` are evaluated.
    Tests with no ``os.*`` marker run on all platforms.

    Args:
        item: The collected test item about to be set up.
    """
    for marker in item.iter_markers():
        name = marker.name  # e.g. "os.linux", "os.windows", "os.both"
        if not name.startswith("os."):
            continue
        os_val = name.split(".", 1)[1]

        if os_val == "both":
            return  # explicitly cross-platform — never skip

        if os_val == "linux" and _PLATFORM != "linux":
            pytest.skip(
                f"os.linux test skipped on platform={_PLATFORM!r}. " "Run on a Linux host to execute this test."
            )

        if os_val == "windows" and _PLATFORM != "windows":
            pytest.skip(
                f"os.windows test skipped on platform={_PLATFORM!r}. "
                "Run on a Windows host with ROCm for Windows to execute this test."
            )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def os_adapter():
    """Return the platform-appropriate OS adapter for this host.

    Returns ``LinuxOsAdapter`` on Linux or ``WindowsOsAdapter`` on Windows.
    The adapter exposes:
        - ``list_gpu_device_paths()``  — ``/dev/dri/renderD*`` (Linux) or BDF
                                          strings (Windows).
        - ``load_kernel_module(mod)``  — ``modprobe`` (Linux) / no-op (Windows).
        - ``unload_kernel_module(mod)``— ``modprobe -r`` (Linux) / no-op (Windows).
        - ``get_platform_name()``      — ``"linux"`` or ``"windows"``.

    Use this fixture when test logic needs platform-specific paths or operations
    that differ between Linux and Windows.  Prefer ``@pytest.mark.os.linux`` /
    ``@pytest.mark.os.windows`` markers for simple skip-on-wrong-OS cases.

    The adapter exposes:
        - ``list_gpu_device_paths()``   — ``/dev/dri/renderD*`` (Linux) or BDF strings (Windows).
        - ``is_module_loaded(mod)``     — ``lsmod`` check (Linux, no sudo) / True (Windows).
        - ``load_kernel_module(mod)``   — ``modprobe`` (Linux) / no-op (Windows).
        - ``unload_kernel_module(mod)`` — ``modprobe -r`` (Linux) / no-op (Windows).
        - ``get_platform_name()``       — ``"linux"``, ``"windows"``, or ``"wsl"``.

    Returns:
        ``AbstractOsAdapter`` concrete instance for the running host.

    Example::

        @pytest.mark.hw.cpu_only
        @pytest.mark.ci.pr
        @pytest.mark.layer.driver
        @pytest.mark.runtime.fast
        @pytest.mark.os.linux
        def test_amdgpu_driver_loaded(os_adapter):
            assert os_adapter.is_module_loaded("amdgpu"), \
                "amdgpu not loaded — run: sudo modprobe amdgpu"

        @pytest.mark.hw.cpu_only
        @pytest.mark.ci.pr
        @pytest.mark.layer.driver
        @pytest.mark.runtime.fast
        def test_gpu_device_paths(os_adapter, platform_name):
            paths = os_adapter.list_gpu_device_paths()
            assert paths, f"No GPU device paths found on {platform_name}"
    """
    return os_adapter_factory()


@pytest.fixture(scope="session")
def platform_name(os_adapter) -> str:
    """Return a string identifying the current host platform.

    Returns:
        ``"linux"`` or ``"windows"``.

    Example::

        def test_platform_detected(platform_name):
            assert platform_name in ("linux", "windows")
    """
    return os_adapter.get_platform_name()  # type: ignore[no-any-return]
