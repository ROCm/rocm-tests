# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
conftest.py -- CMake build fixtures for tests/e2e/hip_runtime/.

Two separate CMake build steps keep the host-only binary independent of GPU_ARCH:

- ``_hip_host_cmake_build_dir``   — builds ``hip_invalid_codeobject_load_test``
  (pure HIP driver API, no GPU kernels; does not require ``--gpu-arch``).
- ``_hip_stream_cmake_build_dir`` — builds ``multi_stream_serialization``
  (HIP kernel code; requires ``--gpu-arch`` so ``-DGPU_ARCH`` can be forwarded).

Build output layout::

    output/test-binaries/hip_runtime/host_build/hip_invalid_codeobject_load_test
    output/test-binaries/hip_runtime/stream_build/multi_stream_serialization
"""

from __future__ import annotations

import logging
import os

import pytest

from tests.common._cmake_build import cmake_build, find_rocm_clangpp

logger = logging.getLogger(__name__)
_SRC_DIR = "tests/e2e/hip_runtime/src"


def _hip_cmake_build(src_dir: str, build_dir: str, rocm_path: str, extra_cmake_args: list[str], label: str) -> None:
    """Build a hip_runtime cmake target using the shared cmake_build helper.

    Args:
        src_dir:          Absolute path to the CMakeLists.txt directory.
        build_dir:        Absolute path to the cmake binary output directory.
        rocm_path:        Resolved TheRock / ROCm install root.
        extra_cmake_args: Additional ``-D`` flags (e.g. ``-DBUILD_HIP_KERNEL_TESTS=OFF``).
        label:            Short name used in assertion messages.
    """
    clangpp = find_rocm_clangpp(rocm_path)
    compiler_args = [f"-DCMAKE_CXX_COMPILER={clangpp}"] if clangpp else []
    cmake_build(
        src_dir, build_dir, rocm_path, compiler_args=compiler_args, extra_cmake_args=extra_cmake_args, label=label
    )


@pytest.fixture(scope="session")
def _hip_host_cmake_build_dir(rock_dir: str, compiler_build_dir: str) -> str:
    """Build the host-only hip_runtime binary (no GPU_ARCH required).

    Compiles ``hip_invalid_codeobject_load_test`` which uses the HIP driver API
    only — no GPU kernels, no ``--offload-arch`` flag needed.

    Args:
        rock_dir:            Path to the ROCm/TheRock install (``--rock-dir`` / ``ROCK_DIR``).
        compiler_build_dir:  Session-scoped output root (``output/test-binaries/`` by default).

    Returns:
        Absolute path to the CMake build directory.
    """
    rocm_path = os.path.realpath(rock_dir)
    build_dir = os.path.join(compiler_build_dir, "hip_runtime", "host_build")
    _hip_cmake_build(
        os.path.abspath(_SRC_DIR),
        build_dir,
        rocm_path,
        extra_cmake_args=["-DBUILD_HIP_KERNEL_TESTS=OFF"],
        label="hip_runtime/host",
    )
    return build_dir


@pytest.fixture(scope="session")
def _hip_stream_cmake_build_dir(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str) -> str:
    """Build the HIP-kernel hip_runtime binary (requires ``--gpu-arch``).

    Compiles ``multi_stream_serialization`` which contains HIP kernels and must
    be compiled for a specific GPU architecture.  ``--gpu-arch`` should be
    supplied on the pytest command line; the CMakeLists.txt fallback is
    ``gfx950``.

    Args:
        request:             Pytest fixture request (provides config access).
        rock_dir:            Path to the ROCm/TheRock install (``--rock-dir`` / ``ROCK_DIR``).
        compiler_build_dir:  Session-scoped output root (``output/test-binaries/`` by default).

    Returns:
        Absolute path to the CMake build directory.
    """
    rocm_path = os.path.realpath(rock_dir)
    build_dir = os.path.join(compiler_build_dir, "hip_runtime", "stream_build")
    clangpp = find_rocm_clangpp(rocm_path)
    # Stream build uses LANGUAGES CXX HIP — pass both CXX and HIP compiler when available.
    compiler_args = [f"-DCMAKE_CXX_COMPILER={clangpp}", f"-DCMAKE_HIP_COMPILER={clangpp}"] if clangpp else []
    cmake_build(
        os.path.abspath(_SRC_DIR),
        build_dir,
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=compiler_args,
        extra_cmake_args=["-DBUILD_HOST_ONLY_TESTS=OFF"],
        label="hip_runtime/stream",
    )
    return build_dir


@pytest.fixture(scope="session")
def hip_invalid_codeobject_load_binary(_hip_host_cmake_build_dir: str) -> str:
    """Return absolute path to the compiled hip_invalid_codeobject_load_test binary.

    Args:
        _hip_host_cmake_build_dir: Build directory from the host-only CMake fixture.

    Returns:
        Absolute path to the ``hip_invalid_codeobject_load_test`` binary.
    """
    binary = os.path.join(_hip_host_cmake_build_dir, "hip_invalid_codeobject_load_test")
    assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary


@pytest.fixture(scope="session")
def multi_stream_serialization_binary(_hip_stream_cmake_build_dir: str) -> str:
    """Return absolute path to the compiled multi_stream_serialization binary.

    Args:
        _hip_stream_cmake_build_dir: Build directory from the HIP-kernel CMake fixture.

    Returns:
        Absolute path to the ``multi_stream_serialization`` binary.
    """
    binary = os.path.join(_hip_stream_cmake_build_dir, "multi_stream_serialization")
    assert os.path.isfile(binary), f"binary not found: {binary}"
    return binary
