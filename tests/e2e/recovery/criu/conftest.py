# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU checkpoint/restore suite.

Provides the CRIU runtime prefix (host and in-target), the cuda_memtest HIP build, the PyTorch
MNIST checkout, and the LLNL RAJAPerf HIP build, grouped into the COMMON / CUDAMEM / MNIST /
RAJAPERF sections below.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pytest

from framework.builder.binary_builder import find_rocm_clangpp
from framework.executors.cpu_executor import CpuExecutor
from tests.common.criu import ensure_criu_runtime, ensure_criu_runtime_target

logger = logging.getLogger(__name__)


# ###########################################################################
# #### COMMON #### -- CRIU runtime prefix (host and in-target)
# ###########################################################################


@pytest.fixture(scope="session")
def criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Make CRIU + amdgpu_plugin ready on the test node (host/SSH); auto-install if not.

    Session-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return ensure_criu_runtime(external_build, cmake_executor, framework_config)


@pytest.fixture
def criu_runtime_target(target_executor, framework_config) -> str:
    """Make CRIU ready *inside* ``target_executor`` so it lives where the workload runs.

    Function-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return ensure_criu_runtime_target(target_executor, framework_config)


# Shared build helpers (used by the CUDAMEM and RAJAPERF sections below).


def _build_executor(cmake_executor, rock_dir: str):
    """Return the build executor: the session SSH executor when remote, else a ``CpuExecutor``
    with ``<rock_dir>/bin`` on PATH so the ROCm toolchain resolves (build is CPU-only).
    """
    if cmake_executor is not None:
        return cmake_executor
    env = {"PATH": f"{os.path.join(rock_dir, 'bin')}:{os.environ.get('PATH', '')}"}
    return CpuExecutor(env_overrides=env, suppress_output_log=True)


def _resolve_dest(cmake_executor, compiler_build_dir: str, subdir: str) -> str:
    """Return the absolute checkout path on the build node."""
    base = os.path.join(compiler_build_dir, subdir)
    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        return str(cmake_executor.workspace_path_for(base))
    return os.path.abspath(base)


# ###########################################################################
# #### CUDAMEM #### -- cuda_memtest HIP build
# cuda_memtest (NCSA) and CRIU (GPL-2.0) are fetched/built at runtime, not vendored.
# See NOTICES.md in this directory for licenses and obligations.
# ###########################################################################

_CUDA_MEMTEST_URL = os.environ.get(
    "ROCM_TEST_CUDA_MEMTEST_URL",
    "https://github.com/ComputationalRadiationPhysics/cuda_memtest.git",
)
_CUDA_MEMTEST_REF = os.environ.get(
    "ROCM_TEST_CUDA_MEMTEST_REF",
    "0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d",
)

_SUBDIR = "recovery/cuda_memtest"


@dataclass(frozen=True)
class CudaMemtestBuild:
    """Compiled cuda_memtest: ``binary`` path and its ``workdir`` (CRIU dumps into the CWD)."""

    binary: str
    workdir: str


@pytest.fixture(scope="session")
def cuda_memtest_build(cmake_executor, rock_dir: str, compiler_build_dir: str, framework_config) -> CudaMemtestBuild:
    """Clone, hipify, patch, and compile cuda_memtest once per session with ``hipcc``.

    Skips (never fails) when ``hipcc`` / ``hipify-perl`` are absent. Returns a CudaMemtestBuild.
    """
    build_exec = _build_executor(cmake_executor, rock_dir)
    hipcc = f"{rock_dir}/bin/hipcc"
    hipify = f"{rock_dir}/bin/hipify-perl"

    tool_check = build_exec.run(
        f"(command -v {hipcc} >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1) && "
        f"(command -v {hipify} >/dev/null 2>&1 || command -v hipify-perl >/dev/null 2>&1) && echo TOOLS_OK"
    )
    if "TOOLS_OK" not in (tool_check.stdout or ""):
        pytest.skip(f"hipcc / hipify-perl not found under {rock_dir}/bin or on PATH -- cannot build cuda_memtest.")

    dest = _resolve_dest(cmake_executor, compiler_build_dir, _SUBDIR)

    # Re-clone each session (hipify-perl cannot re-process an already-hipified tree) and export
    # ROCM_PATH / HIP_PATH so hipcc/hipify-perl resolve their toolchain under --rock-dir.
    script = "\n".join(
        (
            "set -e",
            f"export ROCM_PATH={rock_dir}",
            f"export HIP_PATH={rock_dir}",
            f"rm -rf {dest}",
            f"mkdir -p {os.path.dirname(dest)}",
            f"git clone {_CUDA_MEMTEST_URL} {dest}",
            f"cd {dest}",
            f"git reset --hard {_CUDA_MEMTEST_REF}",
            "cp cuda_memtest.cu cuda_memtest.cu.tmp",
            "ls cuda_memtest.* misc.* tests.cu | " f"xargs -t -I % sh -c '{hipify} % > hip_%; rm %; mv hip_% %;'",
            r"sed -i 's/hipHostGetDevicePointer(&ptr,mappedHostPtr,0);/"
            r"hipHostGetDevicePointer((void **)\&ptr,mappedHostPtr,0);/' cuda_memtest.cu",
            f"{hipcc} -DENABLE_NVML=0 cuda_memtest.cu misc.cpp tests.cu -o cuda_memtest -lpthread",
            "test -x cuda_memtest && echo BUILD_OK",
        )
    )
    logger.info("Building cuda_memtest in %s", dest)
    result = build_exec.run(script, timeout=float(framework_config.therock.build_timeout_secs))
    if "BUILD_OK" not in (result.stdout or ""):
        pytest.fail(
            f"cuda_memtest build failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        )
    return CudaMemtestBuild(binary=os.path.join(dest, "cuda_memtest"), workdir=dest)


# #### END CUDAMEM ####


# ###########################################################################
# #### MNIST #### -- pytorch/examples MNIST setup
# Upstream pytorch/examples (BSD-3-Clause); cloned at runtime, not vendored. See NOTICES.md.
# ###########################################################################

_PYT_EXAMPLES_URL = os.environ.get(
    "ROCM_TEST_PYT_EXAMPLES_URL",
    "https://github.com/pytorch/examples.git",
)
# Writable checkout dir inside the target; the workload runs in target_executor, not on the host.
_PYT_WORKDIR = os.environ.get("ROCM_TEST_PYT_WORKDIR", "/tmp/rocm-tests/pyt_examples")

# Interpreters probed for ROCm (HIP) PyTorch, in order; the ambient torch is used as-is.
_PYTHON_CANDIDATES = ("python3", "python", "/opt/venv/bin/python3", "/opt/conda/bin/python3")


@dataclass(frozen=True)
class PytMnistSetup:
    """MNIST checkout: ``workdir`` (examples/mnist in the target) and the ROCm ``python``."""

    workdir: str
    python: str


def _detect_rocm_python(probe_exec) -> str | None:
    """Return the first interpreter whose ``torch.version.hip`` is truthy, else None.

    Probes ``ROCM_TEST_MNIST_PYTHON`` then common interpreters, running from ``/tmp`` so a pytorch
    source checkout on the default WORKDIR cannot shadow the installed torch on ``sys.path``.
    """
    override = os.environ.get("ROCM_TEST_MNIST_PYTHON")
    candidates = [override] if override else list(_PYTHON_CANDIDATES)
    probe = "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"
    for interp in candidates:
        if not interp or not probe_exec.run(f"command -v {interp} >/dev/null 2>&1").ok:
            continue
        if probe_exec.run(f'cd /tmp && {interp} -c "{probe}"').ok:
            return interp
    return None


@pytest.fixture
def pyt_mnist_setup(target_executor, framework_config) -> PytMnistSetup:
    """Clone pytorch/examples inside ``target_executor`` and resolve the ambient ROCm python.

    Git-clone only (ambient ROCm PyTorch is used as-is). Skips when git or a ROCm-torch python is absent.
    """
    ex = target_executor
    if "GIT_OK" not in (ex.run("command -v git >/dev/null 2>&1 && echo GIT_OK").stdout or ""):
        pytest.skip("git is not available in the target environment -- cannot clone pytorch/examples.")

    python = _detect_rocm_python(ex)
    if not python:
        pytest.skip(
            "No python interpreter with ROCm (HIP) PyTorch found -- run this test inside a ROCm "
            "PyTorch container (set ROCM_TEST_MNIST_PYTHON to point at the interpreter)."
        )

    # POSIX paths inside the target (never os.path.join -- the pytest host may be Windows).
    clone_dir = f"{_PYT_WORKDIR}/examples"
    workdir = f"{clone_dir}/mnist"

    script = "\n".join(
        (
            "set -e",
            f"mkdir -p {_PYT_WORKDIR}",
            f"rm -rf {clone_dir}",
            f"git clone --depth 1 {_PYT_EXAMPLES_URL} {clone_dir}",
            f"test -f {workdir}/main.py && echo CLONE_OK",
        )
    )
    logger.info("Cloning pytorch/examples into %s", clone_dir)
    result = ex.run(script, timeout=float(framework_config.therock.build_timeout_secs))
    if "CLONE_OK" not in (result.stdout or ""):
        pytest.fail(
            f"pytorch/examples clone failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return PytMnistSetup(workdir=workdir, python=python)


# #### END MNIST ####


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

    dest = _resolve_dest(cmake_executor, compiler_build_dir, _RAJAPERF_SUBDIR)

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
