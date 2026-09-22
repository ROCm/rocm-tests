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
    """Return the preinstalled ``hipRTC_gemm`` sample, failing when absent.

    It ships in the rocwmma-clients package rather than being built here, so it
    is run by absolute path out of ``<rock_dir>/bin``.

    A failure rather than a skip: the sample is the whole test, and this suite
    builds nothing of its own, so its absence means the rocWMMA artifact was not
    installed. A skip counts towards a green run and would report that rocWMMA
    was exercised when nothing ran at all.
    """
    binary = pathlib.Path(rock_dir) / "bin" / _HIPRTC_GEMM_BINARY
    reason = (
        f"{_HIPRTC_GEMM_BINARY} not found at {binary} — ensure the rocwmma "
        f"artifact was installed and extracted correctly (rocwmma-clients "
        f"provides this sample)"
    )
    if cmake_executor is None:
        if not (binary.is_file() and os.access(binary, os.X_OK)):
            pytest.fail(reason)
    elif not cmake_executor.run(f"test -x {shlex.quote(str(binary))}").ok:
        pytest.fail(reason)
    return str(binary)
