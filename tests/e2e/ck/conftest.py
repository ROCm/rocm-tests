# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone and build fixtures for tests/e2e/ck/."""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

logger = logging.getLogger(__name__)

_ROCM_LIBRARIES_URL = "https://github.com/ROCm/rocm-libraries.git"
_ROCM_LIBRARIES_REF = os.environ.get("ROCM_TEST_CK_REF", "develop")
_CK_SPARSE_SUBTREE = "projects/composablekernel"
_SUBDIR = "ck"

_STREAMK_TARGET = "tile_example_streamk_gemm_basic"
_STREAMK_BINARY = f"bin/{_STREAMK_TARGET}"

_SUPPORTED_ARCHS = frozenset({"gfx942", "gfx950"})


@pytest.fixture(scope="session")
def ck_repo(external_build, compiler_build_dir: str) -> pathlib.Path:
    """Sparse-clone rocm-libraries (projects/composablekernel); return checkout path."""
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-libraries"
    return external_build.clone_repo(
        _ROCM_LIBRARIES_URL,
        dest,
        ref=_ROCM_LIBRARIES_REF,
        sparse_subtree=_CK_SPARSE_SUBTREE,
    )


@pytest.fixture(scope="session")
def ck_streamk_build(
    ck_repo: pathlib.Path,
    cmake_build_dir,
    gpu_arch: str | None,
) -> str:
    """Configure and build the CK tile stream-k GEMM example; return build directory."""
    if gpu_arch is None or gpu_arch not in _SUPPORTED_ARCHS:
        pytest.skip(f"CK stream-k GEMM requires gfx942 or gfx950; detected arch: {gpu_arch}")

    return cmake_build_dir(
        src=str(ck_repo),
        subdir=_SUBDIR,
        extra_cmake_args=[
            "-DBUILD_DEV=ON",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_FLAGS=-O3 -ftemplate-backtrace-limit=0",
            "-DCMAKE_VERBOSE_MAKEFILE=ON",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        ],
        gpu_arch_var="GPU_TARGETS",
        target=_STREAMK_TARGET,
        artifact=_STREAMK_BINARY,
        label="ck_streamk",
    )
