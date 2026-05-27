# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixture for tests/e2e/rocprim/.

Builds the rocPRIM standalone test suite (rocprim_tests GTest binary) via CMake.

Build output layout::

    output/test-binaries/rocprim/build/rocprim_tests

The binary is invoked by each test with the appropriate ``--gtest_filter``:

- ``SystemMultiStreamTests.*`` — single-GPU concurrency (test_rocprim_concurrent.py)
- ``MultiGPUHMMTests.*``       — 2-GPU HMM managed-memory (test_rocprim_concurrent.py)
- ``SystemStressTests.*``      — longevity/stress weekly gate (test_rocprim_stress.py)

Markers auto-injected by CATEGORY_PROFILES:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from tests.common._cmake_build import cmake_build, find_rocm_clangpp

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/rocprim/src"


def _binary_path(build_dir: str, name: str) -> str:
    binary = os.path.join(build_dir, name)
    assert os.path.isfile(binary), f"rocprim: binary '{name}' not found at {binary} after successful build"
    return binary


@pytest.fixture(scope="session")
def rocprim_tests_binary(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str) -> str:
    """Build rocprim_tests via CMake; return absolute binary path.

    Configures with ``ROCM_PATH`` pointing to the TheRock install and optionally
    ``AMDGPU_TARGETS`` from ``--gpu-arch``.  Both cmake configure and cmake build raise
    ``AssertionError`` on failure, causing pytest to report ``ERROR`` on every
    test that depends on this fixture.

    Args:
        gpu_arch:            Target GPU architecture from the ``gpu_arch`` fixture (``--gpu-arch``).
        rock_dir:            Path to the ROCm/TheRock install.
        compiler_build_dir:  Session-scoped output root (``output/test-binaries/``).

    Returns:
        Absolute path to the compiled ``rocprim_tests`` binary.
    """
    rocm_path = os.path.realpath(rock_dir)
    build_dir = os.path.join(compiler_build_dir, "rocprim", "build")

    # rocPRIM samples apply --offload-arch directly to CXX compile options, so the
    # system C++ compiler (gcc/g++) cannot be used — it doesn't understand HIP flags.
    # Require ROCm clang++ explicitly.
    clangpp = find_rocm_clangpp(rocm_path)
    if clangpp is None:
        pytest.fail(
            f"ROCm clang++ not found under {rocm_path}/lib/llvm/bin (or llvm/bin or bin/amdclang++). "
            "rocprim requires a HIP-capable compiler. "
            "Verify ROCK_DIR / --rock-dir points to a complete ROCm install."
        )

    rocprim_cmake_paths = [
        pathlib.Path(rocm_path) / "lib" / "cmake" / "rocprim",
        pathlib.Path(rocm_path) / "lib64" / "cmake" / "rocprim",
    ]
    if not any(p.is_dir() for p in rocprim_cmake_paths):
        pytest.fail(
            "rocprim CMake config not found. Searched:\n"
            + "\n".join(f"  {p}" for p in rocprim_cmake_paths)
            + "\nrocPRIM cmake configs are in prim_lib — add --prim to the "
            "install_rocm_from_artifacts.py invocation in e2e-nightly.yml."
        )

    cmake_build(
        _SRC_DIR,
        build_dir,
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=[f"-DCMAKE_CXX_COMPILER={clangpp}"],
        extra_cmake_args=["-DCMAKE_BUILD_TYPE=Release"],
        label="rocprim",
    )
    return _binary_path(build_dir, "rocprim_tests")
