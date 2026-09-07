# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixture for tests/e2e/hipblaslt/mini_residual_app/.

Compiles mini_residual_app.cpp via hipcc and returns the binary path.
The fixture is session-scoped and xdist-safe via the compile_binary factory.

hipblaslt/hipblaslt.h must be present under ``{rock_dir}/include/`` — the parent
conftest ``_check_hipblaslt_headers`` guard is reused to surface a clear message
when the header is missing rather than a cryptic hipcc error.
"""

from __future__ import annotations

import pathlib

import pytest

_SUBDIR = "hipblaslt/mini_residual_app"
_SRC = "tests/e2e/hipblaslt/mini_residual_app/src/mini_residual_app.cpp"
_NAME = "mini_residual_app"


def _check_headers(rock_dir: str) -> None:
    """Fail with an actionable message when hipblaslt dev headers are absent.

    Args:
        rock_dir: Path to the ROCm/TheRock install root.
    """
    header = pathlib.Path(rock_dir) / "include" / "hipblaslt" / "hipblaslt.h"
    if not header.exists():
        pytest.fail(
            f"hipblaslt/hipblaslt.h not found under {rock_dir}/include/. "
            "Verify --rock-dir points to a ROCm install that includes the BLAS headers "
            "(pass --blas to install_rocm_from_artifacts.py)."
        )


@pytest.fixture(scope="session")
def mini_residual_app_binary(compile_binary, rock_dir: str, cmake_executor) -> str:
    """Compile mini_residual_app.cpp via hipcc; return absolute binary path.

    Compilation flags mirror the canonical build command:
        hipcc -std=c++17 -O2 -I{rock_dir}/include -L{rock_dir}/lib -lhipblaslt

    Args:
        compile_binary:  Session-scoped factory from ``builder_plugin``.
        rock_dir:        Resolved ROCm/TheRock install path.
        cmake_executor:  ``SshExecutor`` when running remotely, ``None`` for local.

    Returns:
        Absolute path to the compiled ``mini_residual_app`` binary.
    """
    if cmake_executor is None:
        _check_headers(rock_dir)
    return compile_binary(
        src=_SRC,
        output_name=_NAME,
        std="c++17",
        opt="-O2",
        include_dirs=[f"{rock_dir}/include"],
        extra_flags=[f"-L{rock_dir}/lib", "-lhipblaslt"],
        subdir=_SUBDIR,
    )
