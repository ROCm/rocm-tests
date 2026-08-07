# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU checkpoint/restore suite.

Provides the CRIU runtime prefix and the Kokkos performance-benchmark HIP build, grouped into the
COMMON / KOKKOS sections below.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pytest

from framework.executors.cpu_executor import CpuExecutor
from tests.common import criu as criu_common

logger = logging.getLogger(__name__)


# ###########################################################################
# #### COMMON #### -- CRIU runtime prefix (host/SSH)
# ###########################################################################


@pytest.fixture(scope="session")
def criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Make CRIU + amdgpu_plugin ready on the test node (host/SSH); auto-install if not.

    Session-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return criu_common.ensure_criu_runtime(external_build, cmake_executor, framework_config)


def _build_executor(cmake_executor, rock_dir: str):
    """Return the build executor: the session SSH executor when remote, else a ``CpuExecutor``
    with ``<rock_dir>/bin`` on PATH so the ROCm toolchain resolves (build is CPU-only).
    """
    if cmake_executor is not None:
        return cmake_executor
    env = {"PATH": f"{os.path.join(rock_dir, 'bin')}:{os.environ.get('PATH', '')}"}
    return CpuExecutor(env_overrides=env, suppress_output_log=True)


# ###########################################################################
# #### KOKKOS #### -- Kokkos performance-benchmark HIP build
# Upstream kokkos/kokkos; cloned/built at runtime, not vendored. See NOTICES.md.
# ###########################################################################

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


# #### END KOKKOS ####
