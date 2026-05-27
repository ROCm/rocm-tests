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


def check_rocblas_library(rock_dir: str) -> None:
    """Skip the test if the rocBLAS shared library is absent from the ROCm install.

    Both ``small_sliding_contact`` and ``jacobian_svd_multistream`` link against
    ``librocblas.so`` at runtime.  When the library is missing the binary will fail
    with a dynamic-linker error rather than a meaningful assertion — this guard
    surfaces a clear skip message instead.

    Args:
        rock_dir: Path to the ROCm/TheRock install root (from the ``rock_dir`` fixture).
    """
    lib_dir = pathlib.Path(rock_dir) / "lib"
    candidates = list(lib_dir.glob("librocblas.so*"))
    if not candidates:
        pytest.fail(
            f"rocBLAS library not found under {lib_dir} — "
            "ensure the rocblas artifact was downloaded and extracted correctly."
        )


@pytest.fixture(scope="session")
def _rocm_libs_core_build_dir(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str) -> str:
    rocm_path = os.path.realpath(rock_dir)
    clangpp = find_rocm_clangpp(rocm_path)
    # Core build: compiler is optional (CMakeLists.txt falls back to hipcc).
    compiler_args = [f"-DCMAKE_CXX_COMPILER={clangpp}", f"-DCMAKE_HIP_COMPILER={clangpp}"] if clangpp else []
    cmake_build(
        _CORE_SRC,
        os.path.join(compiler_build_dir, "rocm_libs", "core_build"),
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=compiler_args,
        label="rocm_libs_core",
    )
    return os.path.join(compiler_build_dir, "rocm_libs", "core_build")


@pytest.fixture(scope="session")
def precond_conjugate_gradient_binary(_rocm_libs_core_build_dir: str) -> str:
    binary = os.path.join(_rocm_libs_core_build_dir, "precond_conjugate_gradient")
    assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary


@pytest.fixture(scope="session")
def small_sliding_contact_binary(_rocm_libs_core_build_dir: str) -> str:
    binary = os.path.join(_rocm_libs_core_build_dir, "small_sliding_contact")
    assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary


@pytest.fixture(scope="session")
def jacobian_svd_multistream_binary(_rocm_libs_core_build_dir: str) -> str:
    binary = os.path.join(_rocm_libs_core_build_dir, "jacobian_svd_multistream")
    assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary
