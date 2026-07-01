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
import pathlib

import pytest

logger = logging.getLogger(__name__)
_SRC_DIR = "tests/e2e/hip_runtime/src"

# HIP samples upstream suite (ROCm/hip-tests "samples/" subtree).  The legacy
# ROCmTest driver built the samples that ship in the installed ROCm package at
# share/hip/samples — those are the same sources published in ROCm/hip-tests.
# Cloned at runtime (never vendored); pin via env override.
_HIP_TESTS_URL = "https://github.com/ROCm/hip-tests.git"
_HIP_TESTS_REF = os.environ.get("ROCM_TEST_HIP_TESTS_REF", "develop")


@pytest.fixture(scope="session")
def _hip_host_cmake_build_dir(cmake_build_dir) -> str:
    """Build ``hip_invalid_codeobject_load_test`` (HIP driver API only, no GPU_ARCH required)."""
    return cmake_build_dir(
        src=_SRC_DIR,
        subdir="hip_runtime/host_build",
        extra_cmake_args=["-DBUILD_HIP_KERNEL_TESTS=OFF"],
        compiler_mode="optional_auto",
        label="hip_runtime/host",
        sync_dirs=[_SRC_DIR],
        artifact="hip_invalid_codeobject_load_test",
    )


@pytest.fixture(scope="session")
def _hip_stream_cmake_build_dir(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for) -> str:
    """Build ``multi_stream_serialization`` (HIP kernel code; pass ``--gpu-arch``)."""
    require_gpu_arch_for("hip_runtime/stream")
    return cmake_build_dir(
        src=_SRC_DIR,
        subdir="hip_runtime/stream_build",
        gpu_arch=gpu_arch,
        extra_cmake_args=["-DBUILD_HOST_ONLY_TESTS=OFF"],
        compiler_mode="optional_cxx_hip",
        label="hip_runtime/stream",
        sync_dirs=[_SRC_DIR],
        artifact="multi_stream_serialization",
    )


@pytest.fixture(scope="session")
def hip_invalid_codeobject_load_binary(_hip_host_cmake_build_dir: str, built_binary) -> str:
    """Compiled ``hip_invalid_codeobject_load_test`` binary path."""
    return built_binary(
        os.path.join(_hip_host_cmake_build_dir, "hip_invalid_codeobject_load_test"), "hip_invalid_codeobject_load_test"
    )


@pytest.fixture(scope="session")
def multi_stream_serialization_binary(_hip_stream_cmake_build_dir: str, built_binary) -> str:
    """Compiled ``multi_stream_serialization`` binary path."""
    return built_binary(
        os.path.join(_hip_stream_cmake_build_dir, "multi_stream_serialization"), "multi_stream_serialization"
    )


# HIP samples are cloned from ROCm/hip-tests rather than vendored.
# The legacy suite built each installed sample in its own CMake directory with
# CMAKE_PREFIX_PATH pointing at ROCm; ``compiler_mode="none"`` keeps that path,
# letting each sample's CMakeLists locate hipcc on local and remote nodes.


@pytest.fixture(scope="session")
def hip_samples_repo(external_build, compiler_build_dir: str) -> str:
    """Clone the ROCm/hip-tests ``samples/`` subtree once per session; return its path."""
    dest = pathlib.Path(compiler_build_dir) / "hip_runtime" / "hip-tests"
    samples_dir = external_build.clone_repo(_HIP_TESTS_URL, dest, ref=_HIP_TESTS_REF, sparse_subtree="samples")
    # Cone-mode sparse checkout materialises top-level files (LICENSE.md) at the
    # repo root, one level above the samples subtree — verify provenance there.
    external_build.assert_license_present(samples_dir.parent)
    return str(samples_dir)


@pytest.fixture(scope="session")
def hip_sample_build(cmake_build_dir, hip_samples_repo: str, built_binary):
    """Return a factory that builds one HIP sample's CMake project and returns its binary path.

    Mirrors the legacy per-sample ``cmake .. -DCMAKE_PREFIX_PATH=<rocm>`` + ``make``:
    each sample is its own self-contained CMake project built into a dedicated
    sub-directory.  Session scope + fingerprint caching (``artifact=exec_name``)
    ensure each sample is configured/built at most once per session.
    """

    def _build(sample_relpath: str, exec_name: str) -> str:
        sample_src = os.path.join(hip_samples_repo, *sample_relpath.split("/"))
        subdir = "hip_runtime/hip_samples/" + sample_relpath.replace("/", "_")
        build_dir = cmake_build_dir(
            src=sample_src,
            subdir=subdir,
            compiler_mode="none",
            artifact=exec_name,
            label="hip_samples/" + sample_relpath,
        )
        return built_binary(os.path.join(build_dir, exec_name), exec_name)

    return _build
