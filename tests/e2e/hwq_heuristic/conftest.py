# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/hwq_heuristic/.

Builds all hwq_heuristic test binaries via a single CMake configure+build
invocation. All targets share the same CMakeLists.txt and build directory.

Build output layout::

    output/test-binaries/hwq_heuristic/build/hwq_heuristic_test
    output/test-binaries/hwq_heuristic/build/hwq_null_stream_protection_regr
    output/test-binaries/hwq_heuristic/build/hwq_compute_copy_overlap_test

GPU architecture is forwarded from ``--gpu-arch`` when provided; the
CMakeLists.txt raises FATAL_ERROR when GPU_ARCH is absent.
"""

from __future__ import annotations

import logging
import os

import pytest

from tests.common._cmake_build import cmake_build, find_rocm_clangpp

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/hwq_heuristic/src"


def _binary_path(build_dir: str, name: str, remote: bool = False) -> str:
    binary = os.path.join(build_dir, name)
    if not remote:
        assert os.path.isfile(binary), f"hwq_heuristic: binary '{name}' not found at {binary} after successful build"
    return binary


@pytest.fixture(scope="session")
def _hwq_cmake_build_dir(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str, cmake_executor) -> str:
    """Build all hwq_heuristic CMake targets; return absolute build directory path.

    Runs ``cmake -S <src> -B <build> -DROCM_PATH=<rock_dir> [-DGPU_ARCH=<arch>]``
    followed by ``cmake --build <build> --parallel``.  Builds all targets in one
    invocation so individual binary fixtures are cheap path lookups.

    Args:
        gpu_arch:            Target GPU architecture from the ``gpu_arch`` fixture (``--gpu-arch``).
        rock_dir:            Path to the ROCm/TheRock install (``--rock-dir``
                             / ``ROCK_DIR``).  Passed as ``ROCM_PATH`` to cmake.
        compiler_build_dir:  Session-scoped output root
                             (``output/test-binaries/`` by default).
        cmake_executor:      Session-scoped ``SshExecutor`` for remote cmake; ``None`` for local builds.

    Returns:
        Absolute path to the cmake build directory containing all binaries.
    """
    rocm_path = os.path.realpath(rock_dir)
    build_dir = os.path.abspath(os.path.join(compiler_build_dir, "hwq_heuristic", "build"))

    # Compiler is optional: CMakeLists.txt falls back to hipcc when clang++ is absent.
    clangpp = find_rocm_clangpp(rocm_path)
    compiler_args = [f"-DCMAKE_CXX_COMPILER={clangpp}"] if clangpp else []

    cmake_build(
        os.path.abspath(_SRC_DIR),
        build_dir,
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=compiler_args,
        label="hwq_heuristic",
        remote_executor=cmake_executor,
        sync_dirs=[os.path.abspath(_SRC_DIR), os.path.abspath("tests/common/cmake")],
    )
    return build_dir


@pytest.fixture(scope="session")
def hwq_heuristic_binary(_hwq_cmake_build_dir: str, cmake_executor) -> str:
    """Compiled ``hwq_heuristic_test`` binary path."""
    return _binary_path(_hwq_cmake_build_dir, "hwq_heuristic_test", remote=cmake_executor is not None)


@pytest.fixture(scope="session")
def hwq_null_stream_protection_binary(_hwq_cmake_build_dir: str, cmake_executor) -> str:
    """Compiled ``hwq_null_stream_protection_regr`` binary path."""
    return _binary_path(_hwq_cmake_build_dir, "hwq_null_stream_protection_regr", remote=cmake_executor is not None)


@pytest.fixture(scope="session")
def hwq_compute_copy_overlap_binary(_hwq_cmake_build_dir: str, cmake_executor) -> str:
    """Compiled ``hwq_compute_copy_overlap_test`` binary path."""
    return _binary_path(_hwq_cmake_build_dir, "hwq_compute_copy_overlap_test", remote=cmake_executor is not None)
