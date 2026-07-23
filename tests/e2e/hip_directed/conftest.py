# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build fixtures for the public ROCm/rocm-systems hip-tests catch2 suite."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import pathlib
import subprocess

import pytest

logger = logging.getLogger(__name__)

_ROCM_SYSTEMS_URL = "https://github.com/ROCm/rocm-systems"
_SUBDIR = "hip_directed"

# The catch2 memory unit does find_library(numa REQUIRED). Bare CI containers
# lack it; install best-effort before configure (container runs as root).
_SYSTEM_DEPS = "libnuma-dev"


def _install_system_deps() -> None:
    """Best-effort apt install of hip-tests catch2 system build dependencies."""
    sudo = "" if os.geteuid() == 0 else "sudo "
    cmd = f"{sudo}apt-get update && {sudo}apt-get install -y --no-install-recommends {_SYSTEM_DEPS}"
    try:
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            logger.warning(
                "hip_directed system-deps install returned %d:\n%s",
                result.returncode,
                result.stderr[-2000:],
            )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("hip_directed system-deps install failed: %s", exc)


# Only the catch2 executables that contain the directed tests are built (not the
# whole ``build_tests`` meta-target). This keeps the build fast and skips modules
# like ``coopGrpTest`` that require bleeding-edge HIP headers.
_EXE_TARGETS = ("DeviceTest", "StreamTest", "MemoryTest1", "ModuleTest")


def _resolve_rocm_systems_ref(rock_dir: str) -> str:
    """Pick the rocm-systems ref to clone.

    Order: explicit ``ROCM_TEST_ROCM_SYSTEMS_REF`` env override, else the exact
    commit the installed ROCm was built from (TheRock manifest ``pin_sha``), else
    ``develop``. Pinning to the manifest commit keeps the hip-tests source in sync
    with the installed HIP headers (avoids version skew).
    """
    override = os.environ.get("ROCM_TEST_ROCM_SYSTEMS_REF")
    if override:
        return override
    manifest = pathlib.Path(rock_dir) / "share" / "therock" / "therock_manifest.json"
    try:
        data = json.loads(manifest.read_text())
        for sm in data.get("submodules", []):
            name = sm.get("submodule_name", "")
            url = sm.get("submodule_url", "")
            if name == "rocm-systems" or "rocm-systems" in url:
                sha = sm.get("pin_sha")
                if sha:
                    logger.info("hip_directed: pinning rocm-systems to manifest commit %s", sha)
                    return sha
    except (OSError, ValueError) as exc:
        logger.warning("hip_directed: could not read TheRock manifest (%s); using develop", exc)
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


@pytest.fixture(scope="session")
def hip_catch_repo(external_build, compiler_build_dir: str, rock_dir: str):
    """Clone ROCm/rocm-systems (at the ROCm's manifest-pinned commit) once per session."""
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-systems"
    ref = _resolve_rocm_systems_ref(rock_dir)
    repo = external_build.clone_repo(_ROCM_SYSTEMS_URL, dest, ref=ref)
    external_build.assert_license_present(repo)
    return repo


@pytest.fixture(scope="session")
def hip_catch_build_dir(cmake_build_dir, rock_dir: str, gpu_arch: str | None, hip_catch_repo) -> str:
    """Configure and build only the catch2 executables holding the directed tests."""
    catch_src = pathlib.Path(hip_catch_repo) / "projects" / "hip-tests" / "catch"
    extra_args = ["-DHIP_PLATFORM=amd", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}"]
    # Pin the offload arch so CMake's HIP-compiler ABI check does not fall back to
    # rocm_agent_enumerator (which reads sysfs, ignores ROCR_VISIBLE_DEVICES) and
    # emit one duplicated --offload-arch per GPU on multi-GPU CI runners.
    if gpu_arch:
        extra_args.append(f"-DCMAKE_HIP_ARCHITECTURES={gpu_arch}")
    _install_system_deps()
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
