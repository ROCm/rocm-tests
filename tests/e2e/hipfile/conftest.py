# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build and environment fixtures for tests/e2e/hipfile/.

hipFile is built from ROCm/rocm-systems (``projects/hipfile``) into a private
install prefix, then the system test binary is built against it. The suite is
skipped when no NVMe mount is supplied via ``ROCM_TEST_HIPFILE_NVME_PATH`` or
when the kernel UAPI headers lack ``STATX_DIOALIGN`` (Linux 6.1+).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
import shlex
import subprocess

import pytest

from framework.builder.binary_builder import find_rocm_clangpp
from tests.e2e.hipfile._constants import (
    BINARY_NAME,
    HIPFILE_MONOREPO_URL,
    HIPFILE_REF,
    HIPFILE_SUBPATH,
    NVME_PATH_ENV,
    SRC_DIR,
)

_SUBDIR = "hipfile"
_STATX_HEADER = "/usr/include/linux/stat.h"


@dataclass(frozen=True)
class HipfileEnv:
    """Resolved paths for one hipFile system-test run."""

    binary: str
    ld_library_path: str
    nvme_path: str


def _node_path(cmake_executor, compiler_build_dir: str, *parts: str) -> str:
    """Return a path under the build workspace on the node that builds."""
    path = pathlib.Path(compiler_build_dir).joinpath(*parts)
    if cmake_executor is not None:
        return str(cmake_executor.workspace_path_for(path))
    return str(path.resolve())


def _sh(cmake_executor, cmd: str, timeout: float = 60.0) -> bool:
    """Run a shell command on the build node; return True when it exits 0."""
    if cmake_executor is not None:
        return cmake_executor.run(cmd, timeout=timeout).ok
    return subprocess.run(["bash", "-c", cmd], capture_output=True, timeout=timeout, check=False).returncode == 0


@pytest.fixture(scope="session")
def hipfile_nvme_path() -> str:
    """Return the NVMe mount point from the environment; skip when unset."""
    path = os.environ.get(NVME_PATH_ENV, "").strip()
    if not path:
        pytest.skip(f"{NVME_PATH_ENV} not set — hipFile system tests need an NVMe ext4 mount")
    return path


@pytest.fixture(scope="session")
def hipfile_install_prefix(
    hipfile_nvme_path: str,  # requested first so a missing mount skips before the build
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    cmake_build_dir,
) -> str:
    """Build hipFile from source into a private prefix; return the prefix path."""
    rocm_path = os.path.realpath(rock_dir)
    clangpp = find_rocm_clangpp(rocm_path)
    if clangpp is None:
        pytest.fail(f"ROCm clang++ not found under {rocm_path} — required to build hipFile")
    clangpp_str = str(clangpp)
    clang_str = clangpp_str[:-2] if clangpp_str.endswith("++") else clangpp_str

    prefix = _node_path(cmake_executor, compiler_build_dir, _SUBDIR, "install")
    statx_hdr = _node_path(cmake_executor, compiler_build_dir, _SUBDIR, "statx-hdr")

    # hipFile reads direct-I/O alignment through statx(STATX_DIOALIGN), which needs
    # Linux 6.1+ UAPI headers. Fall back to the running kernel's headers when the
    # system ones are older.
    q_hdr = shlex.quote(statx_hdr)
    stage_cmd = (
        f"grep -q STATX_DIOALIGN {_STATX_HEADER} || "
        f"{{ mkdir -p {q_hdr}/linux && "
        f"cp /usr/src/linux-headers-$(uname -r)/include/uapi/linux/stat.h {q_hdr}/linux/stat.h; }}"
    )
    if not _sh(cmake_executor, stage_cmd):
        pytest.skip("Kernel UAPI headers with STATX_DIOALIGN (Linux 6.1+) are not available on the build node")

    src = external_build.clone_repo(
        url=HIPFILE_MONOREPO_URL,
        dest=pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-systems",
        ref=HIPFILE_REF,
        timeout=float(framework_config.therock.build_timeout_secs),
        sparse_subtree=HIPFILE_SUBPATH,
    )
    external_build.assert_license_present(src)

    cmake_build_dir(
        src=str(src),
        subdir=f"{_SUBDIR}/hipfile-build",
        gpu_arch_var="CMAKE_HIP_ARCHITECTURES",
        compiler_mode="cxx_hip",
        target="install",
        label="hipfile",
        extra_cmake_args=[
            f"-DCMAKE_C_COMPILER={clang_str}",
            "-DCMAKE_HIP_PLATFORM=amd",
            f"-DCMAKE_CXX_FLAGS=-isystem {statx_hdr}",
            "-DBUILD_TESTING=OFF",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
        ],
    )
    return prefix


@pytest.fixture(scope="session")
def hipfile_system_test_binary(
    hipfile_install_prefix: str,
    compiler_build_dir: str,
    cmake_executor,
    cmake_build_dir,
    built_binary,
) -> str:
    """Build the hipFile system test against the private hipFile; return the binary path."""
    statx_hdr = _node_path(cmake_executor, compiler_build_dir, _SUBDIR, "statx-hdr")
    build_dir = cmake_build_dir(
        src=SRC_DIR,
        subdir=f"{_SUBDIR}/system_test",
        compiler_mode="auto",
        label="hipfile_system_test",
        sync_dirs=[SRC_DIR],
        artifact=BINARY_NAME,
        extra_cmake_args=[
            f"-DHIPFILE_PREFIX={hipfile_install_prefix}",
            f"-DCMAKE_CXX_FLAGS=-isystem {statx_hdr}",
        ],
    )
    return built_binary(os.path.join(build_dir, BINARY_NAME), BINARY_NAME)


@pytest.fixture
def hipfile_env(
    target_executor,
    ld_path: dict,
    hipfile_install_prefix: str,
    hipfile_system_test_binary: str,
    hipfile_nvme_path: str,
) -> HipfileEnv:
    """Resolve the run environment; fail fast when the NVMe mount is missing on the node."""
    check = target_executor.run(f"test -d {shlex.quote(hipfile_nvme_path)}", timeout=30.0)
    if not check.ok:
        pytest.fail(f"{NVME_PATH_ENV}={hipfile_nvme_path} is not a directory on the target node")
    ld = ":".join(
        [
            f"{hipfile_install_prefix}/lib",
            f"{hipfile_install_prefix}/lib64",
            ld_path["LD_LIBRARY_PATH"],
        ]
    )
    return HipfileEnv(binary=hipfile_system_test_binary, ld_library_path=ld, nvme_path=hipfile_nvme_path)
