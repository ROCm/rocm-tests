# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone and build fixtures for tests/e2e/ck/."""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from framework.gpu.detector import GpuDetector

logger = logging.getLogger(__name__)

_ROCM_LIBRARIES_URL = "https://github.com/ROCm/rocm-libraries.git"
_ROCM_LIBRARIES_REF = os.environ.get("ROCM_TEST_CK_REF", "develop")
_CK_SPARSE_SUBTREE = "projects/composablekernel"
_SUBDIR = "ck"

_FMHA_FWD_TARGET = "tile_example_fmha_fwd"
_FMHA_BWD_TARGET = "tile_example_fmha_bwd"
_FMHA_FWD_BINARY = f"bin/{_FMHA_FWD_TARGET}"
_FMHA_BWD_BINARY = f"bin/{_FMHA_BWD_TARGET}"

_SUPPORTED_ARCHS = frozenset({"gfx942", "gfx950"})

_FMHA_CMAKE_ARGS = [
    "-DBUILD_DEV=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_CXX_FLAGS=-O3 -ftemplate-backtrace-limit=0",
    "-DCMAKE_VERBOSE_MAKEFILE=ON",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
]


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
def ck_fmha_build(
    ck_repo: pathlib.Path,
    cmake_build_dir,
    gpu_arch: str | None,
) -> str:
    """Build CK FMHA forward and backward targets; return build directory."""
    arch = gpu_arch
    if arch is None:
        gpus = GpuDetector().detect()
        arch = gpus[0].arch if gpus else None
    if arch is None or arch not in _SUPPORTED_ARCHS:
        pytest.skip(f"CK FMHA dropout requires gfx942 or gfx950; detected arch: {arch}")

    # Build fwd target first (also runs cmake configure).
    build_dir = cmake_build_dir(
        src=str(ck_repo),
        subdir=_SUBDIR,
        gpu_arch=arch,
        extra_cmake_args=_FMHA_CMAKE_ARGS,
        gpu_arch_var="GPU_TARGETS",
        target=_FMHA_FWD_TARGET,
        artifact=_FMHA_FWD_BINARY,
        label="ck_fmha_fwd",
    )

    # Build bwd target in the same already-configured build directory.
    cmake_build_dir(
        src=str(ck_repo),
        subdir=_SUBDIR,
        gpu_arch=arch,
        extra_cmake_args=_FMHA_CMAKE_ARGS,
        gpu_arch_var="GPU_TARGETS",
        target=_FMHA_BWD_TARGET,
        artifact=_FMHA_BWD_BINARY,
        label="ck_fmha_bwd",
    )

    return build_dir
