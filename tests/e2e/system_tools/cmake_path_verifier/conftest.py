# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Session fixtures for cmake_path_verifier tests."""

from __future__ import annotations

import os
import pathlib

import pytest

from tests.common.os_packages import install_packages
from tests.e2e.system_tools.cmake_path_verifier._constants import PACKAGES_TO_INSTALL

# Cross-process guard: a module-level flag does not work under pytest-xdist, where
# each worker is a separate Python process starting with its own flag == False, so
# all workers would try to install concurrently. Serialise with an OS file lock and
# a per-run sentinel file so the install runs exactly once per session.
_LOCK_DIR = pathlib.Path("output") / ".pkg-locks"
# PYTEST_XDIST_TESTRUNUID is shared by all workers of one run; fall back to the pid
# for a non-xdist run so the sentinel is still unique per session.
_RUN_ID = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getpid())


@pytest.fixture(autouse=True)
def cmake_packages_installed(target_executor) -> None:
    """Install ROCm devel packages required by cmake_path_verifier tests.

    Runs exactly once per session across all xdist workers: a ``filelock.FileLock``
    serialises workers and a per-run sentinel file records that the install has
    completed. target_executor is function-scoped so this fixture must be too.
    """
    from filelock import FileLock  # local import: optional dependency, matches GpuFileLock

    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = _LOCK_DIR / f"cmake_path_verifier_{_RUN_ID}.done"
    lock_path = _LOCK_DIR / f"cmake_path_verifier_{_RUN_ID}.lock"

    with FileLock(str(lock_path)):
        if sentinel.exists():
            return
        install_packages(target_executor, PACKAGES_TO_INSTALL)
        sentinel.touch()
