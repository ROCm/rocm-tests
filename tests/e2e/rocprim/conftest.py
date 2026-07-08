# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixture for tests/e2e/rocprim/.

Builds the ``rocprim_tests`` GTest binary via CMake.

Build output layout::

    output/test-binaries/rocprim/build/rocprim_tests

The binary is invoked per test with ``--gtest_filter`` (SystemMultiStreamTests,
MultiGPUHMMTests, SystemStressTests).
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/rocprim/src"


@pytest.fixture(scope="session")
def rocprim_tests_binary(
    gpu_arch: str | None,
    rock_dir: str,
    cmake_executor,
    cmake_build_dir,
    built_binary,
    require_gpu_arch_for,
) -> str:
    """Build ``rocprim_tests`` via CMake; return absolute binary path."""
    require_gpu_arch_for("rocprim")
    rocm_path = os.path.realpath(rock_dir)

    if cmake_executor is None:
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

    build_dir = cmake_build_dir(
        src=_SRC_DIR,
        subdir="rocprim",
        gpu_arch=gpu_arch,
        extra_cmake_args=["-DCMAKE_BUILD_TYPE=Release"],
        compiler_mode="auto",
        label="rocprim",
        sync_dirs=[_SRC_DIR],
        artifact="rocprim_tests",
    )
    return built_binary(os.path.join(build_dir, "rocprim_tests"), "rocprim_tests")
