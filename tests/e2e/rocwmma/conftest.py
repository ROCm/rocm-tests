# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixtures for the rocWMMA sample suite."""

from __future__ import annotations

import os
import pathlib
import shlex

import pytest

_HIPRTC_GEMM_BINARY = "hipRTC_gemm"


@pytest.fixture(scope="session")
def hiprtc_gemm_binary(rock_dir: str, cmake_executor) -> str:
    """Return the preinstalled ``hipRTC_gemm`` sample, skipping when absent.

    It ships in the rocwmma-clients package rather than being built here, so it
    is run by absolute path out of ``<rock_dir>/bin``.
    """
    binary = pathlib.Path(rock_dir) / "bin" / _HIPRTC_GEMM_BINARY
    reason = f"{_HIPRTC_GEMM_BINARY} not found at {binary} — install the rocwmma-clients package"
    if cmake_executor is None:
        if not (binary.is_file() and os.access(binary, os.X_OK)):
            pytest.skip(reason)
    elif not cmake_executor.run(f"test -x {shlex.quote(str(binary))}").ok:
        pytest.skip(reason)
    return str(binary)
