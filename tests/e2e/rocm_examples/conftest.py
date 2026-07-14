# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build fixtures for public ROCm/rocm-examples ports."""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

logger = logging.getLogger(__name__)

_ROCM_EXAMPLES_URL = "https://github.com/ROCm/rocm-examples.git"
_ROCM_EXAMPLES_REF = os.environ.get("ROCM_TEST_ROCM_EXAMPLES_REF", "amd-mainline")
_SUBDIR = "rocm_examples"


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
    return cmake_build_dir(
        src=str(rocm_examples_repo),
        subdir=_SUBDIR,
        extra_cmake_args=[f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}"],
        compiler_mode="optional_cxx_hip",
        artifact="bin/HIP-Basic/hip_bit_extract",
        label="rocm_examples",
    )
