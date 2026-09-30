# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Session fixtures for cmake_path_verifier tests."""

from __future__ import annotations

import pytest

from tests.common.os_packages import install_packages
from tests.e2e.system_tools.cmake_path_verifier._constants import PACKAGES_TO_INSTALL

# Guard so package installation runs only once per pytest session even though
# the fixture is function-scoped (target_executor cannot be session-scoped).
_packages_installed = False


@pytest.fixture(autouse=True)
def cmake_packages_installed(target_executor) -> None:
    """Install ROCm devel packages required by cmake_path_verifier tests.

    Runs once per session (guarded by module-level flag). target_executor is
    function-scoped so this fixture must be too; the flag prevents repeated installs.
    """
    global _packages_installed
    if not _packages_installed:
        install_packages(target_executor, PACKAGES_TO_INSTALL)
        _packages_installed = True
