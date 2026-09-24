# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixtures for the rocSOLVER benchmark suite."""

from __future__ import annotations

import os
import pathlib
import shlex

import pytest

_ROCSOLVER_BENCH_BINARY = "rocsolver-bench"


@pytest.fixture(scope="session")
def rocsolver_bench_binary(rock_dir: str, cmake_executor) -> str:
    """Return the preinstalled ``rocsolver-bench`` client, skipping when absent.

    It ships in the rocSOLVER clients package rather than being built here, so
    it is run by absolute path out of ``<rock_dir>/bin``.
    """
    binary = pathlib.Path(rock_dir) / "bin" / _ROCSOLVER_BENCH_BINARY
    reason = f"{_ROCSOLVER_BENCH_BINARY} not found at {binary} — install the rocSOLVER clients package"
    if cmake_executor is None:
        if not (binary.is_file() and os.access(binary, os.X_OK)):
            pytest.skip(reason)
    elif not cmake_executor.run(f"test -x {shlex.quote(str(binary))}").ok:
        pytest.skip(reason)
    return str(binary)
