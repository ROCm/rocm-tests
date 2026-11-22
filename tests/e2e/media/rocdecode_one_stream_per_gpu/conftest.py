# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixture for the rocDecode one-stream-per-partition suite.

This directory lives at ``tests/e2e/media/rocdecode_one_stream_per_gpu/``. The
fixture builds the three mode binaries in one CMake invocation:

    output/test-binaries/rocdecode_one_stream_per_gpu/build/rocdecode_spx_one_stream_per_card
    output/test-binaries/rocdecode_one_stream_per_gpu/build/rocdecode_dpx_one_stream_per_half
    output/test-binaries/rocdecode_one_stream_per_gpu/build/rocdecode_cpx_one_stream_per_slice

``xcd_burn`` is built beside them and is used only by the optional ``smi_one_xcd``
check, which the pytest module does not run.
"""

from __future__ import annotations

import os

import pytest

_SRC_DIR = "tests/e2e/media/rocdecode_one_stream_per_gpu"
_BINARIES = (
    "rocdecode_spx_one_stream_per_card",
    "rocdecode_dpx_one_stream_per_half",
    "rocdecode_cpx_one_stream_per_slice",
)


@pytest.fixture(scope="session")
def rocdecode_one_stream_binaries(
    gpu_arch: str | None,
    cmake_build_dir,
    built_binary,
) -> dict[str, str]:
    """Build the SPX, DPX, and CPX binaries; return ``{name: absolute path}``."""
    build_dir = cmake_build_dir(
        src=_SRC_DIR,
        subdir="rocdecode_one_stream_per_gpu",
        gpu_arch=gpu_arch,
        compiler_mode="auto",
        label="rocdecode_one_stream_per_gpu",
        sync_dirs=[_SRC_DIR],
        artifact=_BINARIES[-1],
    )
    return {name: built_binary(os.path.join(build_dir, name), name) for name in _BINARIES}
