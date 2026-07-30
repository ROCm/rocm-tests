# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared build and environment fixtures for the CRIU tests in this directory.

These session fixtures are consumed by every test module under ``tests/e2e/recovery/criu/``
(cuda_memtest checkpoint/restore, the zip/unzip round-trip, and the Kokkos benchmark).

``cuda_memtest_build`` (session): clone cuda_memtest at a pinned commit, hipify + patch the
sources, and build the HIP binary with ``hipcc`` on the node the tests run on; skips when the
ROCm toolchain is absent.

``kokkos_build`` (session): clone the pinned Kokkos tag, configure with the HIP backend +
benchmarks, and build the benchmark target on the node the tests run on (arch auto-detected,
gfx942 APU handled explicitly); skips when the ``hipcc``/``cmake``/``make`` toolchain is absent.

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
# Upstream Kokkos HPC benchmark (pinned tag for reproducibility)
# Kokkos (Apache-2.0 WITH LLVM-exception) is a full CMake project cloned and
# built at runtime, not vendored. See NOTICES.md for licenses and obligations.
# ---------------------------------------------------------------------------

_KOKKOS_URL = os.environ.get("ROCM_TEST_KOKKOS_URL", "https://github.com/kokkos/kokkos.git")
_KOKKOS_REF = os.environ.get("ROCM_TEST_KOKKOS_REF", "4.2.01")
_KOKKOS_SUBDIR = "recovery/kokkos"

# No explicit device arch is passed for discrete GPUs: with HIP enabled the compiler
# auto-detects the GFX target of the build node, so one recipe covers every discrete part.
# The sole special case is the gfx942 APU, which needs this arch flag plus ``HSA_XNACK=1``
# and a Release build type.
_KOKKOS_APU_ARCH_FLAG = "Kokkos_ARCH_AMD_GFX942_APU"

# Benchmark binary produced by ``-DKokkos_ENABLE_BENCHMARKS=On`` (relative to build dir).
_KOKKOS_BENCHMARK_RELPATH = "core/perf_test/Kokkos_PerformanceTest_Benchmark"
# CMake target for the benchmark. Only this target is built (not the whole tree): the unit-test
# executables link the system gtest, which on some hosts is a non-PIC static archive that fails
# with ``R_X86_64_32 ... recompile with -fPIC``. The benchmark links google-benchmark, not gtest,
# so building just it sidesteps that while keeping the configure line unchanged.
_KOKKOS_BENCHMARK_TARGET = "Kokkos_PerformanceTest_Benchmark"

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


@dataclass(frozen=True)
class KokkosBuild:
    """Result of the session-scoped Kokkos performance-benchmark build.

    Attributes:
        binary:  Absolute path to the compiled ``Kokkos_PerformanceTest_Benchmark``
                 binary on the node where the tests execute.
        workdir: The CMake build directory. The benchmark is launched from here and
                 CRIU dumps its image files / dump.log / restore.log into it, so all
                 checkpoint/restore commands ``cd`` here.
        is_apu:  True when built for the gfx942 APU. The launch step then exports
                 ``HSA_XNACK=1``, which the APU build requires at run time.
    """

    binary: str
    workdir: str
    is_apu: bool = False


def _is_gfx942_apu(build_exec, rock_dir: str) -> bool:
    """Return True when the build node's GPU is the gfx942 APU.

    The gfx942 APU needs a dedicated build (explicit ``Kokkos_ARCH_AMD_GFX942_APU`` flag,
    ``HSA_XNACK=1`` and a Release build), whereas every other target builds with the compiler's
    arch auto-detection. The APU and the gfx942 discrete part report the same ``gfx942`` arch id,
    so the ``rocminfo`` marketing name is the only field that separates them.
    """
    out = build_exec.run(f"{rock_dir}/bin/rocminfo 2>/dev/null || rocminfo 2>/dev/null").stdout or ""
    # Marketing name emitted by rocminfo for the gfx942 APU; the only signal distinguishing it
    # from the gfx942 discrete part, which reports an identical gfx942 arch id.
    return "MI300A" in out


def _detect_gpu_arch(build_exec, rock_dir: str) -> str | None:
    """Return the GFX arch of the first real GPU on the build node, or None.

    Probes ``rocm_agent_enumerator`` (under --rock-dir, falling back to PATH), which lists one
    GFX id per agent plus the ``gfx000`` CPU pseudo-agent; the first non-``gfx000`` entry is
    returned with any feature suffix (``:sramecc+:xnack-``) stripped. Used only to precheck
    Kokkos arch support -- the build itself passes no arch flag (compiler auto-detects).
    """
    out = (
        build_exec.run(f"{rock_dir}/bin/rocm_agent_enumerator 2>/dev/null || rocm_agent_enumerator 2>/dev/null").stdout
        or ""
    )
    for token in out.split():
        arch = token.strip().split(":")[0]
        if arch.startswith("gfx") and arch != "gfx000":
            return arch
    return None


def _kokkos_supports_arch(build_exec, kokkos_dir: str, arch: str) -> bool:
    """Return True when the cloned Kokkos tree can build for GFX *arch*.

    Kokkos enumerates the AMD targets it can compile as ``KOKKOS_ARCH_OPTION(AMD_GFX<nnn> ...)``
    entries under ``cmake/``. Checking the actual checkout -- rather than a static list that goes
    stale -- means that bumping ``ROCM_TEST_KOKKOS_REF`` to a release which adds a newer part
    (e.g. gfx950, gfx1250) makes this build it with no code change here.
    """
    token = f"AMD_{arch.upper()}"  # gfx950 -> AMD_GFX950
    out = build_exec.run(f"grep -rqi '{token}' {kokkos_dir}/cmake && echo SUPPORTED").stdout or ""
    return "SUPPORTED" in out


@pytest.fixture(scope="session")
def kokkos_build(
    external_build,
    cmake_executor,
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
) -> KokkosBuild:
    """Clone and CMake-build the Kokkos performance benchmark once per session.

    Clones the pinned tag with the remote-transparent ``external_build.clone_repo``, configures
    with the HIP backend + benchmarks, and builds the benchmark target (run in place, no install).
    The device arch is left to the compiler's auto-detection on the build node -- no ``Kokkos_ARCH``
    flag is passed -- except on the gfx942 APU, where the build adds ``-DKokkos_ARCH_AMD_GFX942_APU=On``
    with ``HSA_XNACK=1`` and a Release build type; every other target uses a Debug build. Only the
    benchmark target is built, so the unit-test suite (which links the system gtest) is never
    compiled. Skips only when the ``hipcc``/``cmake``/``make`` toolchain is absent. Returns a
    KokkosBuild.
    """
    build_exec = _build_executor(cmake_executor, rock_dir)
    hipcc = f"{rock_dir}/bin/hipcc"
    tool_check = build_exec.run(
        f"(command -v {hipcc} >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1) && "
        "command -v cmake >/dev/null 2>&1 && command -v make >/dev/null 2>&1 && echo TOOLS_OK"
    )
    if "TOOLS_OK" not in (tool_check.stdout or ""):
        pytest.skip(f"hipcc / cmake / make not found under {rock_dir}/bin or on PATH -- cannot build Kokkos.")

    # Idempotent clone into the shared external tree; provenance guard on the checkout.
    dest = os.path.join(compiler_build_dir, _KOKKOS_SUBDIR, f"kokkos-{_KOKKOS_REF}")
    kokkos_dir = str(external_build.clone_repo(url=_KOKKOS_URL, dest=dest, ref=_KOKKOS_REF, timeout=1800.0))
    external_build.assert_license_present(kokkos_dir)

    # Resolve to an absolute path: the test cd's into workdir before invoking build.binary, so a
    # repo-relative binary path would not resolve there. Remote uses the SSH workspace mapping.
    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        kokkos_dir = cmake_executor.workspace_path_for(kokkos_dir)
    else:
        kokkos_dir = os.path.abspath(kokkos_dir)

    build_dir = f"{kokkos_dir}/build"
    binary = os.path.join(build_dir, _KOKKOS_BENCHMARK_RELPATH)

    # The gfx942 APU gets an explicit arch flag + HSA_XNACK + Release; every other target relies
    # on the compiler's arch auto-detection with a Debug build and unit tests enabled (no
    # Kokkos_ARCH flag passed).
    is_apu = _is_gfx942_apu(build_exec, rock_dir)
    if is_apu:
        arch_export = "export HSA_XNACK=1"
        arch_flags = f"-D{_KOKKOS_APU_ARCH_FLAG}=On -DCMAKE_BUILD_TYPE=Release"
    else:
        # Discrete GPUs use the compiler's arch auto-detection (no Kokkos_ARCH flag). Preflight:
        # if the detected GFX target is not one the pinned Kokkos ref can build, skip with an
        # actionable hint rather than failing deep in cmake. Unknown arch (detection failed)
        # falls through and lets the build attempt proceed.
        arch = _detect_gpu_arch(build_exec, rock_dir)
        if arch and not _kokkos_supports_arch(build_exec, kokkos_dir, arch):
            pytest.skip(
                f"Kokkos {_KOKKOS_REF} has no build target for {arch} (newer parts such as "
                f"gfx950 or gfx1250 need a newer Kokkos); set ROCM_TEST_KOKKOS_REF "
                f"to a release that supports {arch}."
            )
        arch_export = "true"
        arch_flags = "-DKokkos_ENABLE_TESTS=On -DCMAKE_BUILD_TYPE=Debug"

    # ROCM_PATH / HIP_PATH / PATH / LD_LIBRARY_PATH are exported so hipcc resolves its bundled
    # toolchain under --rock-dir instead of a stale default install. The build dir is wiped and
    # recreated each run, then only the benchmark target is built (run in place, no install).
    script = "\n".join(
        (
            "set -e",
            f"export ROCM_PATH={rock_dir}",
            f"export HIP_PATH={rock_dir}",
            f"export PATH={rock_dir}/bin:$PATH",
            f"export LD_LIBRARY_PATH={rock_dir}/lib:$LD_LIBRARY_PATH",
            arch_export,
            f"rm -rf {build_dir}",
            f"mkdir -p {build_dir}",
            f"cd {build_dir}",
            "cmake -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_INSTALL_PREFIX=../install "
            "-DKokkos_ENABLE_BENCHMARKS=On -DKokkos_ENABLE_HIP=On "
            "-DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=On -DKokkos_ENABLE_SERIAL=on "
            f"-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=off {arch_flags} ..",
            f"make -j$(nproc) {_KOKKOS_BENCHMARK_TARGET}",
            f"test -x {_KOKKOS_BENCHMARK_RELPATH} && echo BUILD_OK",
        )
    )
    logger.info("Building Kokkos benchmark in %s (apu=%s)", build_dir, is_apu)
    result = build_exec.run(script, timeout=float(framework_config.therock.build_timeout_secs))
    if "BUILD_OK" not in (result.stdout or ""):
        pytest.fail(
            f"Kokkos build failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        )
    return KokkosBuild(binary=binary, workdir=build_dir, is_apu=is_apu)


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
