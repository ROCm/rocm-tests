# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build fixtures for the public ROCm/rocm-systems hip-tests catch2 suite."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import shlex

import pytest

logger = logging.getLogger(__name__)

_ROCM_SYSTEMS_URL = "https://github.com/ROCm/rocm-systems"
_SUBDIR = "hip_directed"

# TheRock installs bundle their own copy of the OS build dependencies (including
# libnuma + numa.h, which the catch2 memory unit requires) under this subdir, so
# the suite needs no apt packages on the host / CI container.
_SYSDEPS = "lib/rocm_sysdeps"


# Only the catch2 executables that contain the directed tests are built (not the
# whole ``build_tests`` meta-target). This keeps the build fast and skips modules
# like ``coopGrpTest`` that require bleeding-edge HIP headers.
_EXE_TARGETS = ("DeviceTest", "StreamTest", "MemoryTest1", "ModuleTest")

# The memory performance scenarios build into their own executable. Kept out of
# ``_EXE_TARGETS`` so the directed unit tests do not pay for compiling the
# 17-source performance suite. Upstream emits catch2 binaries under
# ``CATCH_BUILD_DIR``, mirroring the source layout beneath it.
_PERF_MEMORY_TARGET = "MemoryPerformance"
_PERF_MEMORY_SUBPATH = ("catch_tests", "performance", "scenarios", "memory", _PERF_MEMORY_TARGET)


def _read_manifest(rock_dir: str, cmake_executor) -> str | None:
    """Read the TheRock manifest from the node that owns the ROCm install.

    In ``--remote-node`` mode ``rock_dir`` lives on the remote host, so the file
    must be read over SSH; a local read would always miss and silently degrade the
    ref resolution below. Returns ``None`` when the manifest is absent.
    """
    manifest = f"{rock_dir}/share/therock/therock_manifest.json"
    if cmake_executor is not None:
        result = cmake_executor.run(f"cat {shlex.quote(manifest)}", timeout=60)
        return result.stdout if result.ok else None
    try:
        return pathlib.Path(manifest).read_text()
    except OSError:
        return None


def _resolve_rocm_systems_ref(rock_dir: str, cmake_executor) -> str:
    """Pick the rocm-systems ref to clone.

    Order: explicit ``ROCM_TEST_ROCM_SYSTEMS_REF`` env override, else the exact
    commit the installed ROCm was built from (TheRock manifest ``pin_sha``), else
    ``develop``. Pinning to the manifest commit keeps the hip-tests source in sync
    with the installed HIP headers (avoids version skew).
    """
    override = os.environ.get("ROCM_TEST_ROCM_SYSTEMS_REF")
    if override:
        return override
    raw = _read_manifest(rock_dir, cmake_executor)
    if raw is not None:
        try:
            data = json.loads(raw)
            for sm in data.get("submodules", []):
                name = sm.get("submodule_name", "")
                url = sm.get("submodule_url", "")
                if name == "rocm-systems" or "rocm-systems" in url:
                    sha = sm.get("pin_sha")
                    if sha:
                        logger.info("hip_directed: pinning rocm-systems to manifest commit %s", sha)
                        return sha
        except ValueError as exc:
            logger.warning("hip_directed: TheRock manifest is not valid JSON (%s)", exc)
    # Non-TheRock ROCm installs ship no manifest, so this stays a fallback rather
    # than a hard failure — but the source may then skew from the installed HIP
    # headers. Set ROCM_TEST_ROCM_SYSTEMS_REF to pin explicitly.
    logger.warning(
        "hip_directed: no rocm-systems pin_sha found in %s; falling back to 'develop', "
        "which may skew from the installed HIP headers",
        rock_dir,
    )
    return "develop"


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


def _dir_exists(path: pathlib.Path, cmake_executor) -> bool:
    """Test for a directory on the node that owns it.

    Same split as :func:`_read_manifest`: under ``--remote-node`` a ``pathlib``
    check would consult the local filesystem and report a false negative.
    """
    if cmake_executor is not None:
        return cmake_executor.run(f"test -d {shlex.quote(str(path))}", timeout=60).ok
    return path.is_dir()


def _catch_cmake_args(rock_dir: str, gpu_arch: str | None, cmake_executor) -> list[str]:
    """CMake arguments shared by every build of the hip-tests catch2 suite."""
    # BUILD_PERF_TESTS is OFF upstream, leaving the performance tree out of the
    # configure and its targets undefined. Enabled for both builds so they share
    # one configured tree; the executables are EXCLUDE_FROM_ALL, so nothing extra
    # compiles until a target asks for it.
    args = ["-DHIP_PLATFORM=amd", "-DBUILD_PERF_TESTS=ON", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}"]
    # Resolve libnuma / numa.h from the ROCm install's bundled sysdeps rather than
    # requiring host apt packages (the memory units do find_library(numa REQUIRED)
    # + find_path(numa.h)).
    sysdeps = pathlib.Path(rock_dir) / _SYSDEPS
    if _dir_exists(sysdeps, cmake_executor):
        args.append(f"-DCMAKE_LIBRARY_PATH={sysdeps / 'lib'}")
        args.append(f"-DCMAKE_INCLUDE_PATH={sysdeps / 'include'}")
    # Pin the offload arch so CMake's HIP-compiler ABI check does not fall back to
    # rocm_agent_enumerator (which reads sysfs, ignores ROCR_VISIBLE_DEVICES) and
    # emit one duplicated --offload-arch per GPU on multi-GPU CI runners.
    args.append(f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}")
    return args


@pytest.fixture(scope="session")
def hip_catch_repo(external_build, compiler_build_dir: str, rock_dir: str, cmake_executor):
    """Clone ROCm/rocm-systems (at the ROCm's manifest-pinned commit) once per session."""
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-systems"
    ref = _resolve_rocm_systems_ref(rock_dir, cmake_executor)
    repo = external_build.clone_repo(_ROCM_SYSTEMS_URL, dest, ref=ref)
    external_build.assert_license_present(repo)
    return repo


@pytest.fixture(scope="session")
def hip_catch_build_dir(
    cmake_build_dir, rock_dir: str, gpu_arch: str | None, hip_catch_repo, require_gpu_arch_for, cmake_executor
) -> str:
    """Configure and build only the catch2 executables holding the directed tests."""
    require_gpu_arch_for("hip_directed")
    catch_src = pathlib.Path(hip_catch_repo) / "projects" / "hip-tests" / "catch"
    extra_args = _catch_cmake_args(rock_dir, gpu_arch, cmake_executor)
    build_dir = ""
    with _single_visible_gpu():
        for target in _EXE_TARGETS:
            build_dir = cmake_build_dir(
                src=str(catch_src),
                subdir=_SUBDIR,
                extra_cmake_args=extra_args,
                # cxx_hip sets CMAKE_HIP_COMPILER to the ROCm clang++ so the build
                # uses the installed toolchain's device libraries (a bare system
                # clang++ cannot find the ROCm device library).
                compiler_mode="cxx_hip",
                gpu_arch=gpu_arch,
                gpu_arch_var="GPU_TARGETS",
                target=target,
                label=f"hip_directed_catch2:{target}",
            )
    return build_dir


@pytest.fixture(scope="session")
def hip_perf_memory_binary(
    cmake_build_dir,
    rock_dir: str,
    gpu_arch: str | None,
    hip_catch_repo,
    require_gpu_arch_for,
    built_binary,
    cmake_executor,
) -> str:
    """Build the memory performance catch2 executable and return its path.

    Shares the clone and build tree with ``hip_catch_build_dir`` (same
    ``subdir``), so requesting both in one session configures CMake once.
    """
    require_gpu_arch_for("hip_directed")
    catch_src = pathlib.Path(hip_catch_repo) / "projects" / "hip-tests" / "catch"
    with _single_visible_gpu():
        build_dir = cmake_build_dir(
            src=str(catch_src),
            subdir=_SUBDIR,
            extra_cmake_args=_catch_cmake_args(rock_dir, gpu_arch, cmake_executor),
            compiler_mode="cxx_hip",
            gpu_arch=gpu_arch,
            gpu_arch_var="GPU_TARGETS",
            target=_PERF_MEMORY_TARGET,
            label=f"hip_directed_catch2:{_PERF_MEMORY_TARGET}",
        )
    return built_binary(os.path.join(build_dir, *_PERF_MEMORY_SUBPATH), _PERF_MEMORY_TARGET)
