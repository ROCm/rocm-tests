# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixture for tests/e2e/kfd/ (KFD / kfdtest suite).

``kfdtest`` is the GTest suite for the Kernel Fusion Driver (KFD) — the AMD GPU
kernel driver / thunk layer (libhsakmt) at the base of the ROCm stack. Upstream
it lives in the ROCm/rocm-systems monorepo under
``projects/rocr-runtime/libhsakmt/tests/kfdtest``.

    1. Sparse-clone the ``libhsakmt`` project subtree from ROCm/rocm-systems.
       The whole project (not just ``tests/kfdtest``) is checked out because
       kfdtest's CMakeLists.txt resolves the thunk headers via the relative path
       ``${PROJECT_SOURCE_DIR}/../../include`` -> ``libhsakmt/include`` (which
       carries ``hsakmt/hsakmt.h`` and ``hsakmt/linux/kfd_ioctl.h``). A narrower
       checkout of only ``tests/kfdtest`` omits that sibling ``include`` tree and
       the build fails with ``'hsakmt/hsakmt.h' file not found``.
    2. Configure + build it with CMake against the resolved TheRock/ROCm install,
       replicating the original test's CMake arguments:
         -DCMAKE_C_COMPILER   = <rocm>/.../clang (amdclang)
         -DCMAKE_CXX_COMPILER = <rocm>/.../clang++ (amdclang++)
         -DCMAKE_PREFIX_PATH  = <rocm>            (added automatically by cmake_build)
         -DROCM_PATH          = <rocm>            (added automatically by cmake_build)
         -DCMAKE_EXE_LINKER_FLAGS = -ldl
         -DROCM_DIR           = <rocm>
         -DOPENCL_DIR         = <rocm>

Provisioning assumptions (see the module docstring in test_kfd.py for the full
list): the target node must have a loaded ``amdgpu`` kernel module and the KFD
character device (``/dev/kfd``). OS build/runtime dependencies (libdrm, libnuma)
are expected to be provisioned on the fleet via ``--pre-install`` rather than
installed from inside the test (the original test installed these itself, which
is host-provisioning and out of scope for a framework test).
"""

from __future__ import annotations

import logging

import pytest

from tests.common.kfd import build_kfdtest_binary

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def kfdtest_binary(
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    gpu_arch: str | None,
    built_binary,
    cmake_build_dir,
) -> str:
    """Sparse-clone and CMake-build ``kfdtest``; return the absolute binary path.

    Session-scoped so the clone + build happens once per session regardless of
    how many test functions request it.
    """
    return build_kfdtest_binary(
        rock_dir=rock_dir,
        external_build=external_build,
        cmake_executor=cmake_executor,
        framework_config=framework_config,
        cmake_build_dir_fn=cmake_build_dir,
        built_binary_fn=built_binary,
        gpu_arch=gpu_arch,
        compiler_build_dir=compiler_build_dir,
    )
