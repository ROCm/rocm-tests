# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/rocm_libs/.

Build output layout::

    output/test-binaries/rocm_libs/core_build/precond_conjugate_gradient
    output/test-binaries/rocm_libs/core_build/small_sliding_contact
    output/test-binaries/rocm_libs/core_build/jacobian_svd_multistream
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from tests.common._cmake_build import cmake_build, find_rocm_clangpp

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


@pytest.fixture(scope="session")
def _rocm_libs_core_build_dir(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str, cmake_executor) -> str:
    rocm_path = os.path.realpath(rock_dir)
    clangpp = find_rocm_clangpp(rocm_path)
    # Core build: compiler is optional (CMakeLists.txt falls back to hipcc).
    compiler_args = [f"-DCMAKE_CXX_COMPILER={clangpp}", f"-DCMAKE_HIP_COMPILER={clangpp}"] if clangpp else []
    cmake_build(
        os.path.abspath(_CORE_SRC),
        os.path.join(compiler_build_dir, "rocm_libs", "core_build"),
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=compiler_args,
        label="rocm_libs_core",
        remote_executor=cmake_executor,
        sync_dirs=[os.path.abspath(_CORE_SRC), os.path.abspath("tests/common/cmake")],
    )
    return os.path.abspath(os.path.join(compiler_build_dir, "rocm_libs", "core_build"))


@pytest.fixture(scope="session")
def precond_conjugate_gradient_binary(_rocm_libs_core_build_dir: str, cmake_executor) -> str:
    binary = os.path.join(_rocm_libs_core_build_dir, "precond_conjugate_gradient")
    if cmake_executor is None:
        assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary


@pytest.fixture(scope="session")
def small_sliding_contact_binary(_rocm_libs_core_build_dir: str, cmake_executor) -> str:
    binary = os.path.join(_rocm_libs_core_build_dir, "small_sliding_contact")
    if cmake_executor is None:
        assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary


@pytest.fixture(scope="session")
def jacobian_svd_multistream_binary(_rocm_libs_core_build_dir: str, cmake_executor) -> str:
    binary = os.path.join(_rocm_libs_core_build_dir, "jacobian_svd_multistream")
    if cmake_executor is None:
        assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary
