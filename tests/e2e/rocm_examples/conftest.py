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
def rocm_example_build(cmake_build_dir, rock_dir: str, rocm_examples_repo, built_binary):
    """Return a factory that builds one ROCm/rocm-examples sample."""

    def _build(example_relpath: str, exec_name: str) -> str:
        build_dir = cmake_build_dir(
            src=str(rocm_examples_repo / example_relpath),
            subdir=f"{_SUBDIR}/{example_relpath}",
            extra_cmake_args=[f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={rock_dir}"],
            compiler_mode="optional_cxx_hip",
            artifact=exec_name,
            label=f"rocm_examples/{example_relpath}",
        )
        return built_binary(os.path.join(build_dir, exec_name), exec_name)

    return _build
