# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU checkpoint/restore suite.

Provides the CRIU runtime prefix and the LLNL RAJAPerf HIP build, grouped into the
COMMON / RAJAPERF sections below.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pytest
from tests.common.criu import ensure_criu_runtime

from framework.builder.binary_builder import find_rocm_clangpp
from framework.executors.cpu_executor import CpuExecutor

logger = logging.getLogger(__name__)


# ###########################################################################
# #### COMMON #### -- CRIU runtime prefix (host/SSH)
# ###########################################################################


@pytest.fixture(scope="session")
def criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Make CRIU + amdgpu_plugin ready on the test node (host/SSH); auto-install if not.

    Session-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return ensure_criu_runtime(external_build, cmake_executor, framework_config)


# ###########################################################################
# #### RAJAPERF #### -- LLNL RAJAPerf HIP build
# Upstream LLNL/rajaperf; cloned/built at runtime, not vendored. See NOTICES.md.
# ###########################################################################

_RAJAPERF_URL = os.environ.get(
    "ROCM_TEST_RAJAPERF_URL",
    "https://github.com/LLNL/rajaperf.git",
)
# Empty ref clones the default branch; set ROCM_TEST_RAJAPERF_REF to pin a tag/branch/SHA.
_RAJAPERF_REF = os.environ.get("ROCM_TEST_RAJAPERF_REF", "")

_RAJAPERF_SUBDIR = "recovery/rajaperf"


@dataclass(frozen=True)
class RajaPerfBuild:
    """Compiled RAJAPerf: ``binary`` path and its build ``workdir`` (CRIU dumps into the CWD)."""

    binary: str
    workdir: str


def _build_executor(cmake_executor, rock_dir: str):
    """Return the build executor: the session SSH executor when remote, else a ``CpuExecutor``
    with ``<rock_dir>/bin`` on PATH so the ROCm toolchain resolves (build is CPU-only).
    """
    if cmake_executor is not None:
        return cmake_executor
    env = {"PATH": f"{os.path.join(rock_dir, 'bin')}:{os.environ.get('PATH', '')}"}
    return CpuExecutor(env_overrides=env, suppress_output_log=True)


def _resolve_dest(cmake_executor, compiler_build_dir: str) -> str:
    """Return the absolute checkout path on the build node."""
    base = os.path.join(compiler_build_dir, _RAJAPERF_SUBDIR)
    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        return str(cmake_executor.workspace_path_for(base))
    return os.path.abspath(base)


@pytest.fixture(scope="session")
def rajaperf_build(
    cmake_executor, rock_dir: str, compiler_build_dir: str, gpu_arch: str | None, framework_config
) -> RajaPerfBuild:
    """Clone and build LLNL RAJAPerf once per session; return a RajaPerfBuild.

    ``git clone --recursive`` then CMake configure (Release, static, HIP) + ``make -j``. Skips when
    the ROCm clang toolchain is absent; fails if the build fails. MPI is off (its socket blocks
    ``criu dump``); the offload arch is auto-detected when ``--gpu-arch`` is not given.
    """
    build_exec = _build_executor(cmake_executor, rock_dir)

    # Locate the ROCm clang toolchain across TheRock (lib/llvm/bin) and standard ROCm (llvm/bin).
    clangpp = find_rocm_clangpp(rock_dir)
    if clangpp is None:
        pytest.skip(f"ROCm clang toolchain (clang++/amdclang++) not found under {rock_dir} -- cannot build RAJAPerf.")
    clang_dir = str(clangpp.parent)

    dest = _resolve_dest(cmake_executor, compiler_build_dir)

    # Only pass -b when a ref is explicitly pinned; otherwise track the repo's default branch.
    branch_flag = f"-b {_RAJAPERF_REF} " if _RAJAPERF_REF else ""

    # CMake configure. Compiler/HIP paths use the resolved clang dir; $ARCH_FLAGS is expanded by the
    # shell. ENABLE_HIP=ON builds the HIP variants the workload runs; CMAKE_PREFIX_PATH lets BLT's
    # find_package(hip) resolve the hip CMake config under --rock-dir. C++17 is the RAJA minimum.
    # ENABLE_MPI=OFF: OpenMPI's connected TCP socket blocks criu dump and MPI is not used here.
    cmake_cmd = (
        "cmake .. -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=OFF"
        " -D ENABLE_HIP=ON"
        f" -D ROCM_ROOT_DIR={rock_dir} -D ROCM_PATH={rock_dir} -D CMAKE_PREFIX_PATH={rock_dir}"
        f" -D HIP_ROOT_DIR={rock_dir}/share/hip"
        ' -D HIP_PATH="$CLANG_DIR"'
        ' -D CMAKE_C_COMPILER="$CC" -D CMAKE_CXX_COMPILER="$CXX"'
        " $ARCH_FLAGS"
        ' -D CMAKE_CXX_FLAGS="-munsafe-fp-atomics" -D BLT_CXX_STD=c++17'
        " -D ENABLE_MPI=OFF -D ENABLE_OPENMP=ON -D ENABLE_CUDA=OFF"
        ' -D RAJA_COMPILER="RAJA_COMPILER_CLANG"'
        ' -D CMAKE_CXX_FLAGS_RELEASE="-O2" -D RAJA_HIPCC_FLAGS="-fPIC -O2"'
    )

    # ROCM_PATH / HIP_PATH are exported so the HIP CMake modules and the compiler resolve their
    # internal toolchain under --rock-dir.
    script = "\n".join(
        (
            "set -e",
            f"export ROCM_PATH={rock_dir}",
            f"export HIP_PATH={rock_dir}",
            f'CLANG_DIR="{clang_dir}"',
            # Prefer amdclang/amdclang++ when present; fall back to clang/clang++.
            'CXX="$CLANG_DIR/amdclang++"; [ -x "$CXX" ] || CXX="$CLANG_DIR/clang++"',
            'CC="$CLANG_DIR/amdclang"; [ -x "$CC" ] || CC="$CLANG_DIR/clang"',
            # Offload arch: --gpu-arch when given, else auto-detect (amdgpu-arch / rocm_agent_enumerator).
            f'ARCH="{gpu_arch or ""}"',
            '[ -n "$ARCH" ] || ARCH=$("$CLANG_DIR/amdgpu-arch" 2>/dev/null | head -n1 || true)',
            f'[ -n "$ARCH" ] || ARCH=$("{rock_dir}/bin/rocm_agent_enumerator" 2>/dev/null '
            '| grep -v "^gfx000" | head -n1 || true)',
            'if [ -z "$ARCH" ]; then echo "RAJAPERF_ARCH_DETECT_FAILED"; exit 3; fi',
            'echo "RAJAPERF_HIP_ARCH=$ARCH"',
            'ARCH_FLAGS="-D CMAKE_HIP_ARCHITECTURES=$ARCH -D GPU_TARGETS=$ARCH -D AMDGPU_TARGETS=$ARCH"',
            # Wipe + re-clone each session for a deterministic build tree.
            f"rm -rf {dest}",
            f"mkdir -p {os.path.dirname(dest)}",
            # No -b unless a ref is pinned -> a plain clone checks out the repo's default branch.
            f"git clone --recursive {branch_flag}{_RAJAPERF_URL} {dest}",
            f"cd {dest}",
            "mkdir -p build",
            "cd build",
            cmake_cmd,
            "make -j$(nproc)",
            # Static binary runs directly from the build tree; no install step needed.
            "test -x bin/raja-perf.exe && echo BUILD_OK",
        )
    )
    logger.info("Building RAJAPerf in %s (ref=%s)", dest, _RAJAPERF_REF or "<default branch>")
    result = build_exec.run(script, timeout=float(framework_config.therock.build_timeout_secs))
    if "BUILD_OK" not in (result.stdout or ""):
        pytest.fail(
            f"RAJAPerf build failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        )
    build_dir = os.path.join(dest, "build")
    return RajaPerfBuild(binary=os.path.join(build_dir, "bin", "raja-perf.exe"), workdir=build_dir)


# #### END RAJAPERF ####
