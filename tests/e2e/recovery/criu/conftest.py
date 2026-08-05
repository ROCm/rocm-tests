# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU cuda_memtest suite.

``cuda_memtest_build`` (session): clone cuda_memtest at a pinned commit, hipify + patch the
sources, and build the HIP binary with ``hipcc`` on the node the tests run on; skips when the
ROCm toolchain is absent.

``criu_runtime`` (session): delegate to :func:`tests.common.criu.ensure_criu_runtime` to ensure
CRIU + amdgpu_plugin are ready and return the ``sudo -n env PATH=... criu`` command prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pytest

from framework.executors.cpu_executor import CpuExecutor
from tests.common.criu import ensure_criu_runtime

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream cuda_memtest repo (pinned commit for reproducibility)
# cuda_memtest is cloned and built at runtime, not vendored.
# See NOTICES.md in this directory for its license and obligations.
# ---------------------------------------------------------------------------

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
    """Result of the session-scoped cuda_memtest build.

    Attributes:
        binary:  Absolute path to the compiled ``cuda_memtest`` binary on the
                 node where the tests execute.
        workdir: Directory holding the binary and (at runtime) the CRIU image
                 files / dump.log / restore.log. CRIU dumps into the CWD, so all
                 checkpoint/restore commands ``cd`` here.
    """

    binary: str
    workdir: str


def _build_executor(cmake_executor, rock_dir: str):
    """Return an executor that runs build commands on the test node.

    Remote (``--remote-node``): reuse the session ``SshExecutor`` so the checkout
    and compile happen on the same host that later runs CRIU. Local: a
    ``CpuExecutor`` with ``<rock_dir>/bin`` prepended to PATH so ``hipcc`` /
    ``hipify-perl`` resolve (no GPU env is injected -- compilation is CPU-only).
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
    """Clone, hipify, patch, and compile cuda_memtest once per session.

    Runs: git clone + reset to the pinned commit, ``hipify-perl`` on the sources, the
    ``hipHostGetDevicePointer`` cast patch, then ``hipcc -DENABLE_NVML=0 ... -lpthread``.
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

    # hipify-perl cannot re-process an already-hipified tree, so the checkout is
    # wiped and re-cloned each session to keep the build deterministic.
    #
    # ROCM_PATH / HIP_PATH are exported so hipcc/hipify-perl resolve their internal
    # toolchain (clang++, rocm_agent_enumerator) under --rock-dir. Without this
    # hipcc falls back to a stale default (e.g. /opt/rocm-<ver>) and fails with
    # "clang++: No such file or directory" when rock_dir is a non-default install.
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


@pytest.fixture(scope="session")
def criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Ensure CRIU + amdgpu_plugin are ready on the test node; return the ``sudo -n ... criu`` prefix."""
    return ensure_criu_runtime(external_build, cmake_executor, framework_config)
