# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU checkpoint/restore suite.

Provides the CRIU runtime prefix (host and in-target), the cuda_memtest HIP build, and the
PyTorch MNIST checkout, grouped into the COMMON / CUDAMEM / MNIST sections below.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pytest

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


def _build_executor(cmake_executor, rock_dir: str):
    """Return the build executor: the session SSH executor when remote, else a ``CpuExecutor``
    with ``<rock_dir>/bin`` on PATH so ``hipcc`` / ``hipify-perl`` resolve (build is CPU-only).
    """
    if cmake_executor is not None:
        return cmake_executor
    env = {"PATH": f"{os.path.join(rock_dir, 'bin')}:{os.environ.get('PATH', '')}"}
    return CpuExecutor(env_overrides=env, suppress_output_log=True)


def _resolve_dest(cmake_executor, compiler_build_dir: str) -> str:
    """Return the absolute checkout path on the build node."""
    base = os.path.join(compiler_build_dir, _SUBDIR)
    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        return str(cmake_executor.workspace_path_for(base))
    return os.path.abspath(base)


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

    dest = _resolve_dest(cmake_executor, compiler_build_dir)

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
