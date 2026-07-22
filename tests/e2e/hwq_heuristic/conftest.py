# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/hwq_heuristic/.

Each binary has its own cmake_build_dir call with ``target=`` so that running
a single test file compiles only the binary that test needs.  All three targets
share the same ``CMakeLists.txt`` (same ``src``), but the ``--target`` flag
restricts cmake to building only the requested executable.

Build output layout::

    output/test-binaries/hwq_heuristic/hwq_heuristic_test/build/hwq_heuristic_test
    output/test-binaries/hwq_heuristic/hwq_null_stream_protection_regr/build/hwq_null_stream_protection_regr
    output/test-binaries/hwq_heuristic/hwq_compute_copy_overlap_test/build/hwq_compute_copy_overlap_test
    output/test-binaries/hwq_heuristic/hwq_robustness/build/hwq_robustness
    output/test-binaries/hwq_heuristic/hwq_single_stream_no_regr/build/hwq_single_stream_no_regr
    output/test-binaries/hwq_heuristic/hwq_per_device_independence_test/build/hwq_per_device_independence_test

GPU architecture is forwarded from ``--gpu-arch`` when provided; the
CMakeLists.txt raises FATAL_ERROR when GPU_ARCH is absent.
"""

from __future__ import annotations

import logging
import os

import pytest

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/hwq_heuristic/src"

_COMMON_BUILD_KWARGS = dict(
    src=_SRC_DIR,
    compiler_mode="optional_auto",
    sync_dirs=[_SRC_DIR],
)


@pytest.fixture(scope="session")
def hwq_heuristic_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the ``hwq_heuristic_test`` binary path."""
    require_gpu_arch_for("hwq_heuristic")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="hwq_heuristic/hwq_heuristic_test",
        gpu_arch=gpu_arch,
        label="hwq_heuristic/hwq_heuristic_test",
        artifact="hwq_heuristic_test",
        target="hwq_heuristic_test",
    )
    return built_binary(os.path.join(build_dir, "hwq_heuristic_test"), "hwq_heuristic_test")


@pytest.fixture(scope="session")
def hwq_null_stream_protection_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the ``hwq_null_stream_protection_regr`` binary path."""
    require_gpu_arch_for("hwq_heuristic")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="hwq_heuristic/hwq_null_stream_protection_regr",
        gpu_arch=gpu_arch,
        label="hwq_heuristic/hwq_null_stream_protection_regr",
        artifact="hwq_null_stream_protection_regr",
        target="hwq_null_stream_protection_regr",
    )
    return built_binary(
        os.path.join(build_dir, "hwq_null_stream_protection_regr"),
        "hwq_null_stream_protection_regr",
    )


@pytest.fixture(scope="session")
def hwq_compute_copy_overlap_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the ``hwq_compute_copy_overlap_test`` binary path."""
    require_gpu_arch_for("hwq_heuristic")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="hwq_heuristic/hwq_compute_copy_overlap_test",
        gpu_arch=gpu_arch,
        label="hwq_heuristic/hwq_compute_copy_overlap_test",
        artifact="hwq_compute_copy_overlap_test",
        target="hwq_compute_copy_overlap_test",
    )
    return built_binary(os.path.join(build_dir, "hwq_compute_copy_overlap_test"), "hwq_compute_copy_overlap_test")


@pytest.fixture(scope="session")
def hwq_robustness_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the ``hwq_robustness`` binary path."""
    require_gpu_arch_for("hwq_heuristic")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="hwq_heuristic/hwq_robustness",
        gpu_arch=gpu_arch,
        label="hwq_heuristic/hwq_robustness",
        artifact="hwq_robustness",
        target="hwq_robustness",
    )
    return built_binary(os.path.join(build_dir, "hwq_robustness"), "hwq_robustness")


@pytest.fixture(scope="session")
def hwq_single_stream_no_regr_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the ``hwq_single_stream_no_regr`` binary path."""
    require_gpu_arch_for("hwq_heuristic")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="hwq_heuristic/hwq_single_stream_no_regr",
        gpu_arch=gpu_arch,
        label="hwq_heuristic/hwq_single_stream_no_regr",
        artifact="hwq_single_stream_no_regr",
        target="hwq_single_stream_no_regr",
    )
    return built_binary(os.path.join(build_dir, "hwq_single_stream_no_regr"), "hwq_single_stream_no_regr")


@pytest.fixture(scope="session")
def hwq_per_device_independence_binary(
    gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary
) -> str:
    """Compile and return the ``hwq_per_device_independence_test`` binary path."""
    require_gpu_arch_for("hwq_heuristic")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="hwq_heuristic/hwq_per_device_independence_test",
        gpu_arch=gpu_arch,
        label="hwq_heuristic/hwq_per_device_independence_test",
        artifact="hwq_per_device_independence_test",
        target="hwq_per_device_independence_test",
    )
    return built_binary(os.path.join(build_dir, "hwq_per_device_independence_test"), "hwq_per_device_independence_test")
