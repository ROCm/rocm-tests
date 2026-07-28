# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU cuda_memtest suite.

``cuda_memtest_build`` (session): clone cuda_memtest at a pinned commit, hipify + patch the
sources, and build the HIP binary with ``hipcc`` on the node the tests run on; skips when the
ROCm toolchain is absent.

``criu_runtime`` (session): ensure CRIU + amdgpu_plugin are ready (``criu check`` + plugin file),
auto-installing via scripts/install_criu.py when missing, and return the ``sudo -n env PATH=... criu``
prefix. CRIU needs root; passwordless ``sudo -n`` is required or the suite skips cleanly.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
import os
import re

import pytest

from framework.executors.cpu_executor import CpuExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream cuda_memtest repo (pinned commit for reproducibility)
# cuda_memtest (NCSA) and CRIU (GPL-2.0) are fetched/built at runtime, not vendored.
# See NOTICES.md in this directory for licenses and obligations.
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

# ---------------------------------------------------------------------------
# CRIU invocation prefix
# ---------------------------------------------------------------------------

# CRIU installs to /usr/local/sbin (see scripts/install_criu.py). sudo resets the
# environment, so PATH is set explicitly for the elevated process.
_CRIU_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin"
CRIU = f'sudo -n env "PATH={_CRIU_PATH}" criu'

# Path where the amdgpu CRIU plugin is installed.
_AMDGPU_PLUGIN = "/usr/lib/criu/amdgpu_plugin.so"

_INSTALL_HINT = (
    "CRIU + amdgpu_plugin is required by this suite. When missing it is auto-installed "
    "on the test node via tests/e2e/recovery/criu/scripts/install_criu.py (set "
    "ROCM_TEST_CRIU_AUTO_INSTALL=0 to disable, ROCM_TEST_CRIU_VERSION=<tag> to pin the "
    "version). Auto-install needs passwordless sudo, git, a C toolchain, and network "
    "access; you can also run install_criu.py manually on the node beforehand."
)

# CRIU git tag to build when auto-installing.
_DEFAULT_CRIU_VERSION = "v4.1"
# Shell-safe git ref (tag/branch) pattern -- validated before interpolation.
_CRIU_VERSION_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")


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
        return cmake_executor.workspace_path_for(base)
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


def _criu_ready(probe_exec) -> tuple[bool, str]:
    """Return ``(ready, diagnostic)``: ready when the amdgpu plugin exists and ``criu check`` says "Looks good"."""
    if not probe_exec.run(f"test -f {_AMDGPU_PLUGIN}").ok:
        return False, f"amdgpu plugin not found at {_AMDGPU_PLUGIN}"
    check = probe_exec.run(f"{CRIU} check")
    combined = f"{check.stdout}\n{check.stderr}"
    if "Looks good" not in combined:
        return False, f"`criu check` did not report 'Looks good':\n{combined[-1500:]}"
    return True, ""


def _install_criu(probe_exec, is_remote: bool, version: str, timeout: float):
    """Run scripts/install_criu.py on the test node; return its ExecutionResult.

    Local runs execute the script directly; remote runs base64-transfer it to ``/tmp`` on the
    SSH node first (the repo may not be checked out there), then run it.
    """
    local_script = os.path.join(os.path.dirname(__file__), "scripts", "install_criu.py")
    if is_remote:
        with open(local_script, "rb") as handle:
            payload = base64.b64encode(handle.read()).decode()
        remote_script = "/tmp/rocm_test_install_criu.py"
        transfer = probe_exec.run(f"echo {payload} | base64 -d > {remote_script}")
        if not transfer.ok:
            return transfer
        script_path = remote_script
    else:
        script_path = local_script
    return probe_exec.run(f"python3 {script_path} {version}", timeout=timeout)


@pytest.fixture(scope="session")
def criu_runtime(cmake_executor, framework_config) -> str:
    """Ensure CRIU + amdgpu_plugin are ready on the test node; auto-install if not.

    Uses CRIU as-is when ``criu check`` says "Looks good" and the plugin is present; otherwise
    installs via scripts/install_criu.py and re-verifies. Disable with ``ROCM_TEST_CRIU_AUTO_INSTALL=0``;
    pick the tag with ``ROCM_TEST_CRIU_VERSION`` (default ``v4.1``). Returns the ``sudo -n ... criu`` prefix.
    """
    is_remote = cmake_executor is not None
    probe_exec = cmake_executor if is_remote else CpuExecutor(suppress_output_log=True)

    # Passwordless sudo is required to both install and run CRIU (see Privilege note).
    if not probe_exec.run("sudo -n true").ok:
        pytest.skip("Passwordless sudo is not available for the test user. " + _INSTALL_HINT)

    ready, diagnostic = _criu_ready(probe_exec)
    if ready:
        logger.info("CRIU runtime available: amdgpu_plugin present and 'criu check' passed.")
        return CRIU

    if os.environ.get("ROCM_TEST_CRIU_AUTO_INSTALL", "1") == "0":
        pytest.skip(f"CRIU not ready ({diagnostic}) and auto-install disabled. " + _INSTALL_HINT)

    version = os.environ.get("ROCM_TEST_CRIU_VERSION", _DEFAULT_CRIU_VERSION)
    if not _CRIU_VERSION_RE.match(version):
        pytest.fail(f"Invalid ROCM_TEST_CRIU_VERSION {version!r}; expected a git tag/branch name.")

    logger.warning("CRIU not ready (%s). Auto-installing CRIU %s via install_criu.py ...", diagnostic, version)
    install = _install_criu(probe_exec, is_remote, version, float(framework_config.therock.build_timeout_secs))
    if not install.ok:
        pytest.fail(
            f"Automatic CRIU installation failed (exit={install.exit_code}).\n"
            f"stdout: {install.stdout[-2000:]}\nstderr: {install.stderr[-2000:]}\n" + _INSTALL_HINT
        )

    ready, diagnostic = _criu_ready(probe_exec)
    if not ready:
        pytest.fail(f"CRIU still not ready after auto-installation ({diagnostic}). " + _INSTALL_HINT)

    logger.info("CRIU installed via install_criu.py; runtime available.")
    return CRIU
