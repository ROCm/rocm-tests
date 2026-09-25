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


def _run_on_target(cmake_executor, cmd: str, timeout: float):
    """Run an install command on the build target.

    Uses ``cmake_executor`` (the session ``SshExecutor``) so the deps land on the
    remote GPU node when ``--remote-node`` is set; falls back to a local
    ``subprocess`` when ``cmake_executor`` is ``None`` (local mode). Returns
    ``(exit_code, stderr)``.
    """
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout)
        return result.exit_code, result.stderr
    proc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stderr


def _install_system_deps(cmake_executor) -> None:
    """Install rocm-examples system build dependencies on the build target.

    Runs on the remote node via ``cmake_executor`` when present, else locally.
    Aborts on failure: a missing dep otherwise surfaces later as an opaque CMake
    ``Find<Pkg>`` error, so fail fast here with the apt diagnostics.
    """
    sudo = "" if os.geteuid() == 0 else "sudo "
    cmd = f"{sudo}apt-get update && {sudo}apt-get install -y --no-install-recommends {_SYSTEM_DEPS}"
    try:
        rc, err = _run_on_target(cmake_executor, cmd, timeout=1800)
    except Exception as exc:
        pytest.fail(f"rocm_examples system-deps install failed: {exc}")
    if rc != 0:
        pytest.fail(f"rocm_examples system-deps install failed (exit={rc}):\n{err[-2000:]}")


def _install_runtime_python_deps(cmake_executor, rock_dir: str) -> None:
    """Best-effort pip install of rocprofiler-compute requirements (needed by tools samples).

    Runs on the build target via ``cmake_executor`` (remote) or locally. The
    requirements file only ships with the profiler artifact, so a no-op when it
    is absent; best-effort because it only gates the optional ``tools`` bucket.
    """
    reqs = os.path.join(rock_dir, "libexec", "rocprofiler-compute", "requirements.txt")
    cmd = f"test -f {reqs} && python3 -m pip install -q -r {reqs} || true"
    try:
        rc, err = _run_on_target(cmake_executor, cmd, timeout=900)
    except Exception as exc:
        logger.warning("rocm_examples runtime python-deps install failed: %s", exc)
        return
    if rc != 0:
        logger.warning("rocm_examples runtime python-deps install returned %d:\n%s", rc, err[-2000:])


@pytest.fixture(scope="session")
def rocm_examples_repo(external_build, compiler_build_dir: str):
    """Clone ROCm/rocm-examples once per session; return the checkout path."""
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "rocm-examples"
    repo = external_build.clone_repo(_ROCM_EXAMPLES_URL, dest, ref=_ROCM_EXAMPLES_REF)
    external_build.assert_license_present(repo)
    return repo


@pytest.fixture(scope="session")
def rocm_examples_build_dir(cmake_build_dir, cmake_executor, rock_dir: str, rocm_examples_repo) -> str:
    """Configure and build the ROCm/rocm-examples CTest suite."""
    _install_system_deps(cmake_executor)
    _install_runtime_python_deps(cmake_executor, rock_dir)
    # tolerate_build_failure: rocm-examples bundles samples for optional ROCm
    # components (e.g. Applications/monte_carlo_pi needs hipCUB headers) that some
    # installs -- notably the nightly split-artifact rockrel install -- do not
    # ship. Keep building the rest instead of aborting; the per-category test
    # reports any sample whose binary was not produced as "binary not available"
    # (skip) rather than a hard failure.
    return cmake_build_dir(
        src=str(rocm_examples_repo),
        subdir=_SUBDIR,
        extra_cmake_args=[
            f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}",
            "-DCMAKE_DISABLE_FIND_PACKAGE_rpp=ON",
        ],
        compiler_mode="optional_cxx_hip",
        artifact="bin/HIP-Basic/hip_bit_extract",
        label="rocm_examples",
        tolerate_build_failure=True,
    )
