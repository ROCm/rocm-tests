# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixture for tests/e2e/hwq_heuristic/.

Builds ``hwq_heuristic_test`` via CMake (not hipcc directly) because the
source uses GPU architecture-specific compile flags set in CMakeLists.txt.

Build output layout::

    output/test-binaries/hwq_heuristic/build/hwq_heuristic_test

GPU architecture is forwarded from ``--gpu-arch`` when provided; otherwise
CMakeLists.txt falls back to its built-in default (``gfx950``).

Unlike HIP binaries compiled by ``compile_binary`` (hipcc + xdist lock),
this fixture uses ``subprocess.run`` directly because CMake manages its own
incremental build state via the ``build/`` directory.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess

import pytest

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/hwq_heuristic/src"
_BINARY_NAME = "hwq_heuristic_test"


@pytest.fixture(scope="session")
def hwq_heuristic_binary(request, rock_dir: str, compiler_build_dir: str) -> str:
    """Build hwq_heuristic_test via CMake; return absolute binary path.

    Runs ``cmake -S <src> -B <build> -DROCM_PATH=<rock_dir> [-DGPU_ARCH=<arch>]``
    followed by ``cmake --build <build> --parallel``.  Both steps raise
    ``AssertionError`` on failure so pytest reports them as ``ERROR`` on every
    test that depends on this fixture.

    Args:
        request:             Pytest fixture request (provides config access).
        rock_dir:            Path to the ROCm/TheRock install (``--rock-dir``
                             / ``ROCK_DIR``).  Passed as ``ROCM_PATH`` to cmake.
        compiler_build_dir:  Session-scoped output root
                             (``output/test-binaries/`` by default).

    Returns:
        Absolute path to the compiled ``hwq_heuristic_test`` binary.
    """
    build_dir = os.path.join(compiler_build_dir, "hwq_heuristic", "build")
    pathlib.Path(build_dir).mkdir(parents=True, exist_ok=True)

    gpu_arch: str | None = request.config.getoption("--gpu-arch", default=None)
    rocm_path = os.path.realpath(rock_dir)  # resolve symlinks (e.g. /opt/rocm → /opt/rocm-6.4.x)

    cmake_args = [
        "cmake",
        "-S",
        os.path.abspath(_SRC_DIR),
        "-B",
        build_dir,
        f"-DROCM_PATH={rocm_path}",
    ]
    if gpu_arch:
        cmake_args.append(f"-DGPU_ARCH={gpu_arch}")

    logger.info("hwq_heuristic: cmake configure (ROCM_PATH=%s): %s", rocm_path, " ".join(cmake_args))
    r = subprocess.run(cmake_args, capture_output=True, text=True)
    assert r.returncode == 0, f"hwq_heuristic cmake configure failed:\n{r.stdout}\n{r.stderr}"

    build_args = ["cmake", "--build", build_dir, "--parallel"]
    logger.info("hwq_heuristic: cmake build: %s", " ".join(build_args))
    r = subprocess.run(build_args, capture_output=True, text=True)
    assert r.returncode == 0, f"hwq_heuristic cmake build failed:\n{r.stdout}\n{r.stderr}"

    binary = os.path.join(build_dir, _BINARY_NAME)
    assert os.path.isfile(binary), f"hwq_heuristic: binary not found at {binary} after successful build"
    return binary
