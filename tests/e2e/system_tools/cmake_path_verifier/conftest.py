# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Session fixtures for cmake_path_verifier tests."""

from __future__ import annotations

import pytest

from framework.plugins.os_deps_plugin import OS_DEPS
from tests.e2e.system_tools.cmake_path_verifier._constants import PACKAGES_TO_INSTALL


@pytest.fixture(autouse=True)
def cmake_packages_installed(target_executor) -> None:
    """Install ROCm devel packages required by cmake_path_verifier tests.

    Probe-before-install makes this idempotent: re-runs for already-present
    packages return immediately.  Uses per-distro dict specs so dnf/zypper
    targets the -devel variant of each package (matching the original behaviour).
    """
    # Ubuntu ships the base name; RHEL/SLES require the -devel variant.
    specs = [{"apt": p, "dnf": f"{p}-devel", "zypper": f"{p}-devel", "*": p} for p in PACKAGES_TO_INSTALL]
    OS_DEPS(*specs).install()
