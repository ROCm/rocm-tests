# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Session fixtures for cmake_path_verifier tests."""

from __future__ import annotations

import pytest

from tests.common.prereqs import install_packages
from tests.e2e.system_tools.cmake_path_verifier._constants import PACKAGES_TO_INSTALL


@pytest.fixture(scope="session", autouse=True)
def cmake_packages_installed(target_executor) -> None:
    """Install ROCm devel packages required by cmake_path_verifier tests.

    Runs once per session before any test in this area. Requires passwordless
    sudo on the test node for the OS package manager.
    """
    install_packages(target_executor.primary, PACKAGES_TO_INSTALL)
