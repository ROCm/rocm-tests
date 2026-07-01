# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/rocm_libs/.

Each binary has its own cmake_build_dir call with ``target=`` so that running
a single test file compiles only the binary that test needs.

Build output layout::

    output/test-binaries/rocm_libs/small_sliding_contact/build/small_sliding_contact
    output/test-binaries/rocm_libs/jacobian_svd_multistream/build/jacobian_svd_multistream
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

logger = logging.getLogger(__name__)

_CORE_SRC = "tests/e2e/rocm_libs/src"


def check_rocblas_library(rock_dir: str, remote: bool = False, cmake_executor=None) -> None:
    """Fail with an actionable message if ``librocblas.so`` is absent from the ROCm install.

    Args:
        rock_dir:       Path to the ROCm/TheRock install root.
        remote:         When ``True``, delegate the filesystem check to ``cmake_executor`` via SSH.
        cmake_executor: Session-scoped ``SshExecutor``; required when ``remote=True``.
    """
    fail_msg = (
        f"rocBLAS library not found under {rock_dir}/lib — "
        "ensure the rocblas artifact was downloaded and extracted correctly."
    )
    if remote:
        if cmake_executor is not None:
            result = cmake_executor.run(f"ls {rock_dir}/lib/librocblas.so* 2>/dev/null")
            if not result.ok or not result.stdout.strip():
                pytest.fail(fail_msg)
        return
    lib_dir = pathlib.Path(rock_dir) / "lib"
    if not list(lib_dir.glob("librocblas.so*")):
        pytest.fail(fail_msg)


@pytest.fixture(scope="session")
def rocblas_library_guard(rock_dir: str, cmake_executor) -> None:
    """Session-scoped guard: fail early if rocBLAS is absent from the ROCm install.

    Tests declare this fixture to avoid threading ``rock_dir`` and ``cmake_executor``
    through their own signatures.
    """
    check_rocblas_library(rock_dir, remote=cmake_executor is not None, cmake_executor=cmake_executor)


_COMMON_BUILD_KWARGS = dict(
    src=_CORE_SRC,
    compiler_mode="optional_cxx_hip",
    sync_dirs=[_CORE_SRC],
)


@pytest.fixture(scope="session")
def small_sliding_contact_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the small sliding-contact sparse solve workload."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/small_sliding_contact",
        gpu_arch=gpu_arch,
        label="rocm_libs/small_sliding_contact",
        artifact="small_sliding_contact",
        target="small_sliding_contact",
    )
    return built_binary(os.path.join(build_dir, "small_sliding_contact"), "small_sliding_contact")


@pytest.fixture(scope="session")
def jacobian_svd_multistream_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the multi-stream Jacobian/SVD workload binary."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/jacobian_svd_multistream",
        gpu_arch=gpu_arch,
        label="rocm_libs/jacobian_svd_multistream",
        artifact="jacobian_svd_multistream",
        target="jacobian_svd_multistream",
    )
    return built_binary(os.path.join(build_dir, "jacobian_svd_multistream"), "jacobian_svd_multistream")


# requested_gpu_count is provided by the shared suite-level conftest (tests/conftest.py).
