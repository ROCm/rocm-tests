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

from tests.common._cmake_build import cmake_build, find_rocm_clangpp

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/rocprim/src"


def _binary_path(build_dir: str, name: str, remote: bool = False) -> str:
    binary = os.path.join(build_dir, name)
    if not remote:
        assert os.path.isfile(binary), f"rocprim: binary '{name}' not found at {binary} after successful build"
    return binary


@pytest.fixture(scope="session")
def rocprim_tests_binary(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str, cmake_executor) -> str:
    """Build ``rocprim_tests`` via CMake; return absolute binary path."""
    rocm_path = os.path.realpath(rock_dir)
    build_dir = os.path.abspath(os.path.join(compiler_build_dir, "rocprim", "build"))

    # rocPRIM samples apply --offload-arch directly to CXX compile options, so the
    # system C++ compiler (gcc/g++) cannot be used — it doesn't understand HIP flags.
    # Require ROCm clang++ explicitly.
    # In remote mode the clangpp path is validated on the remote host by cmake;
    # skip the local existence check and pass the expected path unconditionally.
    clangpp = find_rocm_clangpp(rocm_path)
    if cmake_executor is None:
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

    # In remote mode clangpp may be None locally (rock_dir may not exist here),
    # but the same path will be valid on the remote host. Build the expected path
    # from rocm_path so cmake receives it when available.
    compiler_path = clangpp or (pathlib.Path(rocm_path) / "lib" / "llvm" / "bin" / "clang++")
    compiler_args = [f"-DCMAKE_CXX_COMPILER={compiler_path}"]

    cmake_build(
        os.path.abspath(_SRC_DIR),
        build_dir,
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=compiler_args,
        extra_cmake_args=["-DCMAKE_BUILD_TYPE=Release"],
        label="rocprim",
        remote_executor=cmake_executor,
        sync_dirs=[os.path.abspath(_SRC_DIR), os.path.abspath("tests/common/cmake")],
    )
    return _binary_path(build_dir, "rocprim_tests", remote=cmake_executor is not None)
