# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build fixtures for public ROCm/rocm-examples ports."""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess

import pytest

logger = logging.getLogger(__name__)

_ROCM_EXAMPLES_URL = "https://github.com/ROCm/rocm-examples.git"
_ROCM_EXAMPLES_REF = os.environ.get("ROCM_TEST_ROCM_EXAMPLES_REF", "amd-mainline")
_SUBDIR = "rocm_examples"

# Tier-1 system build deps rocm-examples CMake looks for (FFmpeg/OpenCV/GLFW/
# Vulkan/GLEW/GLM/VAAPI/elfutils). The minimal OSSCI container lacks these, so
# install them best-effort before configure (container runs as root). Missing
# deps otherwise fail the CMake configure (e.g. FindFFmpeg).
_SYSTEM_DEPS = (
    "libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavdevice-dev "
    "libopencv-dev libglfw3-dev libvulkan-dev glslang-tools libglew-dev libglm-dev "
    "libva-dev libdw-dev"
)


def _install_system_deps() -> None:
    """Best-effort apt install of rocm-examples system build dependencies."""
    sudo = "" if os.geteuid() == 0 else "sudo "
    cmd = f"{sudo}apt-get update && {sudo}apt-get install -y --no-install-recommends {_SYSTEM_DEPS}"
    try:
        result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            logger.warning(
                "rocm_examples system-deps install returned %d:\n%s", result.returncode, result.stderr[-2000:]
            )
    except Exception as exc:
        logger.warning("rocm_examples system-deps install failed: %s", exc)


@pytest.fixture(scope="session")
def rocm_examples_repo(external_build, compiler_build_dir: str):
    """Clone ROCm/rocm-examples once per session; return the checkout path."""
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-examples"
    repo = external_build.clone_repo(_ROCM_EXAMPLES_URL, dest, ref=_ROCM_EXAMPLES_REF)
    external_build.assert_license_present(repo)
    return repo


@pytest.fixture(scope="session")
def rocm_examples_build_dir(cmake_build_dir, rock_dir: str, rocm_examples_repo) -> str:
    """Configure and build the ROCm/rocm-examples CTest suite."""
    _install_system_deps()
    return cmake_build_dir(
        src=str(rocm_examples_repo),
        subdir=_SUBDIR,
        extra_cmake_args=[f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}"],
        compiler_mode="optional_cxx_hip",
        artifact="bin/HIP-Basic/hip_bit_extract",
        label="rocm_examples",
    )
