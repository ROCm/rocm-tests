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

        if os_val == "linux" and _PLATFORM not in ("linux", "wsl"):
            pytest.skip(
                f"os.linux test skipped on platform={_PLATFORM!r}. " "Run on a Linux host to execute this test."
            )

        if os_val == "windows" and _PLATFORM != "windows":
            pytest.skip(
                f"os.windows test skipped on platform={_PLATFORM!r}. "
                "Run on a Windows host with ROCm for Windows to execute this test."
            )


@pytest.fixture(scope="session")
def os_adapter():
    """Return the platform-appropriate OS adapter (``LinuxOsAdapter`` or ``WindowsOsAdapter``).

    Use for platform-specific GPU device paths or kernel module operations.
    For simple OS-skip logic prefer ``@pytest.mark.os.linux`` markers instead.

    Returns:
        ``AbstractOsAdapter`` concrete instance for the running host.
    """
    return os_adapter_factory()


@pytest.fixture(scope="session")
def platform_name(os_adapter) -> str:
    """Return a string identifying the current host platform (``"linux"`` or ``"windows"``)."""
    return os_adapter.get_platform_name()  # type: ignore[no-any-return]
