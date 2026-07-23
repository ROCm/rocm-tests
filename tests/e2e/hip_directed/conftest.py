# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build fixtures for the public ROCm/rocm-systems hip-tests catch2 suite."""

from __future__ import annotations

import contextlib
import os
import pathlib

import pytest

_ROCM_SYSTEMS_URL = "https://github.com/ROCm/rocm-systems"
_ROCM_SYSTEMS_REF = os.environ.get("ROCM_TEST_ROCM_SYSTEMS_REF", "develop")
_SUBDIR = "hip_directed"

# Only the catch2 executables that contain the directed tests are built (not the
# whole ``build_tests`` meta-target). This keeps the build fast and skips modules
# like ``coopGrpTest`` that require bleeding-edge HIP headers not present in every
# ROCm install.
_EXE_TARGETS = ("DeviceTest", "StreamTest", "MemoryTest1", "ModuleTest")


@contextlib.contextmanager
def _single_visible_gpu():
    """Limit GPU visibility during the build.

    The catch2 CMake auto-detects the offload arch via ``rocm_agent_enumerator``,
    which returns one entry per visible GPU. On multi-GPU hosts that yields a
    duplicated ``--offload-arch`` (``clang-offload-bundler: Duplicate targets``).
    Restricting visibility to a single device during configure/build makes the
    enumerator return a single arch.
    """
    saved = {k: os.environ.get(k) for k in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")}
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    os.environ["HIP_VISIBLE_DEVICES"] = "0"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="session")
def hip_catch_repo(external_build, compiler_build_dir: str):
    """Clone ROCm/rocm-systems once per session; return the checkout path."""
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-systems"
    repo = external_build.clone_repo(_ROCM_SYSTEMS_URL, dest, ref=_ROCM_SYSTEMS_REF)
    external_build.assert_license_present(repo)
    return repo


@pytest.fixture(scope="session")
def hip_catch_build_dir(cmake_build_dir, rock_dir: str, hip_catch_repo) -> str:
    """Configure and build only the catch2 executables holding the directed tests."""
    catch_src = pathlib.Path(hip_catch_repo) / "projects" / "hip-tests" / "catch"
    build_dir = ""
    with _single_visible_gpu():
        for target in _EXE_TARGETS:
            build_dir = cmake_build_dir(
                src=str(catch_src),
                subdir=_SUBDIR,
                extra_cmake_args=["-DHIP_PLATFORM=amd", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}"],
                compiler_mode="none",
                gpu_arch_var="GPU_TARGETS",
                target=target,
                label=f"hip_directed_catch2:{target}",
            )
    return build_dir
