# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build and environment fixtures for tests/e2e/recovery/criu/.

Ported from: an external AMD test framework (criu_cuda_memtest_stressTest.py +
CRIU/utils.py) that used ``execution_APIs.test`` / ``platforms.BareMetal`` /
``get_rocm_utils`` -- none of which exist in rocm-tests. This conftest re-expresses
the setup phase (clone + hipify + patch + hipcc build) and the CRIU environment
precondition (``criu check`` + ``amdgpu_plugin.so`` + passwordless sudo) using
only rocm-tests fixtures.

Fixtures
--------
``cuda_memtest_build`` (session)
    Clones ComputationalRadiationPhysics/cuda_memtest, pins the upstream commit,
    hipifies the CUDA sources with ``hipify-perl``, applies the
    ``hipHostGetDevicePointer`` cast patch, and compiles the HIP binary with
    ``hipcc -DENABLE_NVML=0 ... -lpthread`` on the SAME node the tests run on
    (the remote SSH node when ``--remote-node`` is set, else localhost). Skips
    when ``hipcc`` / ``hipify-perl`` are unavailable.

``criu_runtime`` (session)
    Ensures CRIU + amdgpu_plugin are available on the test node, mirroring the
    source's ``check_criu_installed``: probes the amdgpu plugin and ``criu check``
    and, when either is missing, AUTO-INSTALLS by running scripts/install_criu.py
    on that node (the remote SSH node when ``--remote-node`` is set, else
    localhost), then re-verifies. Returns the ``sudo -n env PATH=... criu`` prefix.
    Auto-install requires passwordless sudo; set ``ROCM_TEST_CRIU_AUTO_INSTALL=0``
    to opt out (skip instead) and ``ROCM_TEST_CRIU_VERSION`` to pick the CRIU tag.

Privilege note
--------------
``criu dump`` / ``criu restore`` (and the source-from-scratch install) require
root. rocm-tests executors expose no privilege API (see framework/executors/), so
CRIU is invoked through a ``sudo -n`` (non-interactive) prefix. ``sudo -n`` never
blocks on a password prompt over SSH -- if passwordless sudo is not configured the
probe fails and the suite skips cleanly (it can neither install nor run CRIU).
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
# Upstream cuda_memtest source (pinned for reproducibility, per the manual steps)
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

# Shell-safe gfx arch pattern (interpolated into the hipcc --offload-arch flag).
_GFX_RE = re.compile(r"^gfx[0-9a-fA-F]+$")

# ---------------------------------------------------------------------------
# CRIU invocation prefix
# ---------------------------------------------------------------------------

# CRIU installs to /usr/local/sbin (see scripts/install_criu.py). sudo resets the
# environment, so PATH is set explicitly for the elevated process.
_CRIU_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin"
CRIU = f'sudo -n env "PATH={_CRIU_PATH}" criu'

# Absolute path (per the manual steps) where the amdgpu CRIU plugin is installed.
_AMDGPU_PLUGIN = "/usr/lib/criu/amdgpu_plugin.so"

_INSTALL_HINT = (
    "CRIU + amdgpu_plugin is required by this suite. When missing it is auto-installed "
    "on the test node via tests/e2e/recovery/criu/scripts/install_criu.py (set "
    "ROCM_TEST_CRIU_AUTO_INSTALL=0 to disable, ROCM_TEST_CRIU_VERSION=<tag> to pin the "
    "version). Auto-install needs passwordless sudo, git, a C toolchain, and network "
    "access; you can also run install_criu.py manually on the node beforehand."
)

# CRIU git tag to build when auto-installing (mirrors the source's ``criu_branch``).
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
def cuda_memtest_build(
    cmake_executor, rock_dir: str, gpu_arch: str | None, compiler_build_dir: str, framework_config
) -> CudaMemtestBuild:
    """Clone, hipify, patch, and compile cuda_memtest once per session.

    Mirrors the manual setup steps exactly:
        git clone <repo> && git reset --hard <sha>
        cp cuda_memtest.cu cuda_memtest.cu.tmp            # original backup
        ls cuda_memtest.* misc.* tests.cu | xargs ... hipify-perl
        sed -i 's/hipHostGetDevicePointer(&ptr,.../((void **)&ptr,.../'
        hipcc -DENABLE_NVML=0 cuda_memtest.cu misc.cpp tests.cu -o cuda_memtest -lpthread

    Skips (never fails) when the ROCm toolchain (``hipcc`` / ``hipify-perl``) is
    absent, so a node without a compiler degrades gracefully.

    Returns:
        CudaMemtestBuild with the binary path and working directory.
    """
    build_exec = _build_executor(cmake_executor, rock_dir)
    hipcc = f"{rock_dir}/bin/hipcc"
    hipify = f"{rock_dir}/bin/hipify-perl"

    tool_check = build_exec.run(
        f"(command -v {hipcc} >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1) && "
        f"(command -v {hipify} >/dev/null 2>&1 || command -v hipify-perl >/dev/null 2>&1) && echo TOOLS_OK"
    )
    if "TOOLS_OK" not in (tool_check.stdout or ""):
        pytest.skip(
            f"hipcc / hipify-perl not found under {rock_dir}/bin or on PATH -- "
            "cannot build cuda_memtest for the CRIU stress suite."
        )

    dest = _resolve_dest(cmake_executor, compiler_build_dir)
    offload = ""
    if gpu_arch:
        if not _GFX_RE.match(gpu_arch):
            pytest.fail(f"--gpu-arch {gpu_arch!r} is not a valid gfx target for --offload-arch")
        offload = f"--offload-arch={gpu_arch} "

    # hipify-perl cannot re-process an already-hipified tree, so the checkout is
    # wiped and re-cloned each session to keep the build deterministic.
    script = "\n".join(
        (
            "set -e",
            f"rm -rf {dest}",
            f"mkdir -p {os.path.dirname(dest)}",
            f"git clone {_CUDA_MEMTEST_URL} {dest}",
            f"cd {dest}",
            f"git reset --hard {_CUDA_MEMTEST_REF}",
            "cp cuda_memtest.cu cuda_memtest.cu.tmp",
            "ls cuda_memtest.* misc.* tests.cu | " f"xargs -t -I % sh -c '{hipify} % > hip_%; rm %; mv hip_% %;'",
            r"sed -i 's/hipHostGetDevicePointer(&ptr,mappedHostPtr,0);/"
            r"hipHostGetDevicePointer((void **)\&ptr,mappedHostPtr,0);/' cuda_memtest.cu",
            f"{hipcc} -DENABLE_NVML=0 {offload}cuda_memtest.cu misc.cpp tests.cu -o cuda_memtest -lpthread",
            "test -x cuda_memtest && echo BUILD_OK",
        )
    )
    logger.info("Building cuda_memtest for CRIU stress suite in %s", dest)
    result = build_exec.run(script, timeout=float(framework_config.therock.build_timeout_secs))
    if "BUILD_OK" not in (result.stdout or ""):
        pytest.fail(
            f"cuda_memtest build failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        )
    return CudaMemtestBuild(binary=os.path.join(dest, "cuda_memtest"), workdir=dest)


def _criu_ready(probe_exec) -> tuple[bool, str]:
    """Return ``(ready, diagnostic)`` for CRIU + amdgpu_plugin on the node.

    Ready means the amdgpu plugin file exists and ``criu check`` reports
    "Looks good" -- the same two conditions the source's ``check_criu_installed``
    used to decide CRIU was already installed.
    """
    if not probe_exec.run(f"test -f {_AMDGPU_PLUGIN}").ok:
        return False, f"amdgpu plugin not found at {_AMDGPU_PLUGIN}"
    check = probe_exec.run(f"{CRIU} check")
    combined = f"{check.stdout}\n{check.stderr}"
    if "Looks good" not in combined:
        return False, f"`criu check` did not report 'Looks good':\n{combined[-1500:]}"
    return True, ""


def _install_criu(probe_exec, is_remote: bool, version: str, timeout: float):
    """Run scripts/install_criu.py on the test node; return its ExecutionResult.

    Local runs execute the checked-in script directly. Remote runs base64-transfer
    the script to ``/tmp`` on the SSH node first (the repo may not be checked out
    there), then run it -- keeping the installer a single source of truth.
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

    Mirrors the source's ``check_criu_installed``: if ``criu check`` already reports
    "Looks good" and the amdgpu plugin is present, use it as-is; otherwise install
    CRIU from source via scripts/install_criu.py on the same node the tests run on
    (the remote SSH node when ``--remote-node`` is set, else localhost) and
    re-verify. Auto-install can be disabled with ``ROCM_TEST_CRIU_AUTO_INSTALL=0``
    and the CRIU tag chosen with ``ROCM_TEST_CRIU_VERSION`` (default ``v4.1``).

    Returns the ``sudo -n env PATH=... criu`` prefix used to invoke
    ``criu dump`` / ``criu restore``.
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
