# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build fixture for the rocm-bandwidth-test suite."""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

_RBT_REPO_URL = "https://github.com/ROCm/rocm_bandwidth_test"
_RBT_REPO_REF = os.environ.get("ROCM_TEST_RBT_REF", None)

_MIN_GCC_MAJOR = 7


def _read_os_release(cmake_executor) -> str:
    """Read /etc/os-release from the build host; uses subprocess when cmake_executor is None (local mode)."""
    if cmake_executor is not None:
        return (cmake_executor.run("cat /etc/os-release 2>/dev/null").stdout or "").lower()
    proc = subprocess.run(["cat", "/etc/os-release"], capture_output=True, text=True)
    return proc.stdout.lower()


def _read_gcc_version(cmake_executor) -> str:
    """Read GCC major version from the build host; uses subprocess when cmake_executor is None (local mode)."""
    if cmake_executor is not None:
        return cmake_executor.run("gcc -dumpversion 2>/dev/null").stdout or "0"
    proc = subprocess.run(["gcc", "-dumpversion"], capture_output=True, text=True)
    return proc.stdout or "0"


def _check_gcc_preflight(cmake_executor) -> None:
    """Skip build-from-source on RHEL/CentOS when GCC major version is below the minimum required."""
    os_release = _read_os_release(cmake_executor)
    if not any(token in os_release for token in ("rhel", "centos")):
        return
    gcc_out = _read_gcc_version(cmake_executor)
    major = int(gcc_out.strip().split(".")[0])
    if major < _MIN_GCC_MAJOR:
        pytest.skip(
            f"Building rocm-bandwidth-test from source on RHEL/CentOS requires GCC >= {_MIN_GCC_MAJOR} "
            f"(detected GCC {major}). Install GCC 7.5 on the host before running this test."
        )


@pytest.fixture(scope="session")
def rbt_binary(rock_dir: str, compiler_build_dir: str, external_build, cmake_build_dir, cmake_executor) -> str:
    """Return the absolute path to the rocm-bandwidth-test binary.

    Returns the pre-installed binary when present under rock_dir/bin.
    Otherwise runs a GCC preflight check then clones and builds from source.
    """
    installed = os.path.join(rock_dir, "bin", "rocm-bandwidth-test")
    if os.path.isfile(installed) and os.access(installed, os.X_OK):
        return installed

    _check_gcc_preflight(cmake_executor)

    dest = pathlib.Path(compiler_build_dir) / "system_tools" / "rocm_bandwidth_test"
    with external_build.build_lock("rbt"):
        repo = external_build.clone_repo(_RBT_REPO_URL, dest, ref=_RBT_REPO_REF)
        external_build.assert_license_present(repo)

        # rocm_bandwidth_test vendors fmt as a git submodule; initialize it after clone.
        if cmake_executor is not None:
            sub = cmake_executor.run(f"git -C {repo} submodule update --init --recursive")
            if not sub.ok:
                pytest.fail(f"git submodule init failed on remote:\n{sub.stderr}")
        else:
            sub = subprocess.run(
                ["git", "-C", str(repo), "submodule", "update", "--init", "--recursive"],
                capture_output=True,
                text=True,
            )
            if sub.returncode != 0:
                pytest.fail(f"git submodule init failed:\n{sub.stderr}")

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
