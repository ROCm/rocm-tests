# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build fixture for rocm-bandwidth-test.

Uses the pre-installed binary at ``{rock_dir}/bin/rocm-bandwidth-test`` when available;
falls back to cloning and building from source via CMake.
"""

from __future__ import annotations

import os
import pathlib

import pytest

_RBT_REPO_URL = "https://github.com/ROCm/rocm_bandwidth_test"
_RBT_REPO_REF = os.environ.get("ROCM_TEST_RBT_REF", None)  # None → repo default branch

# Minimum GCC major version required to build from source on RHEL/CentOS.
_MIN_GCC_MAJOR = 7


def _check_gcc_preflight(executor) -> None:
    """Skip build-from-source on RHEL/CentOS if GCC < 7 is installed.

    GCC 7.5 is required to compile rocm-bandwidth-test from source on RHEL/CentOS 7.
    Not needed on other OSes (Ubuntu, SLES) which ship a sufficiently modern toolchain.
    """
    os_release = (executor.run("cat /etc/os-release 2>/dev/null").stdout or "").lower()
    is_rhel_centos = any(token in os_release for token in ("rhel", "centos"))
    if not is_rhel_centos:
        return
    gcc_out = executor.run("gcc -dumpversion 2>/dev/null").stdout or "0"
    major = int(gcc_out.strip().split(".")[0])
    if major < _MIN_GCC_MAJOR:
        pytest.skip(
            f"Building rocm-bandwidth-test from source on RHEL/CentOS requires GCC >= {_MIN_GCC_MAJOR} "
            f"(detected GCC {major}). Install GCC 7.5 on the host before running this test."
        )


@pytest.fixture(scope="session")
def rbt_binary(rock_dir: str, compiler_build_dir: str, external_build, cmake_build_dir, cmake_executor) -> str:
    """Resolve the rocm-bandwidth-test binary path.

    Returns the pre-installed binary if present; otherwise runs a GCC preflight
    check and clones + builds from source via CMake.
    """
    installed = os.path.join(rock_dir, "bin", "rocm-bandwidth-test")
    if os.path.isfile(installed) and os.access(installed, os.X_OK):
        return installed

    _check_gcc_preflight(cmake_executor)

    dest = pathlib.Path(compiler_build_dir) / "system_tools" / "rocm_bandwidth_test"
    with external_build.build_lock("rbt"):
        repo = external_build.clone_repo(_RBT_REPO_URL, dest, ref=_RBT_REPO_REF)
        external_build.assert_license_present(repo)
        build_dir = cmake_build_dir(
            src=str(repo),
            subdir="system_tools/rocm_bandwidth_test",
            extra_cmake_args=[
                "-DAMD_APP_STANDALONE_BUILD_PACKAGE=OFF",
                "-DAMD_APP_ROCM_BUILD_PACKAGE=ON",
                f"-DCLANG_COMPILER_CXX={rock_dir}/bin/hipcc",
            ],
            artifact="rocm-bandwidth-test",
            label="rbt",
        )

    binary = os.path.join(build_dir, "rocm-bandwidth-test")
    if not os.path.isfile(binary):
        pytest.fail(f"rocm-bandwidth-test build succeeded but binary not found at {binary}")
    return binary
