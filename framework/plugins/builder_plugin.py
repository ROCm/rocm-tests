# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
builder_plugin.py -- HIP/C++ binary compilation fixtures.

Provides: rock_dir, compiler_build_dir, compile_binary, ld_path, gpu_arch, arch_lib_path.

rock_dir resolution order: --rock-dir CLI → ROCK_DIR env → rocm-test.toml → "" (empty).
compile_binary wraps BinaryBuilder — session-scoped, incremental, xdist-safe via file lock.
For .hip sources or CMake builds use cmake_build() from tests/common/_cmake_build.py instead.

CLI options added: --rock-dir, --compiler-build-dir.
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from framework.builder.binary_builder import BinaryBuilder

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ROCm compiler CLI options."""
    group = parser.getgroup("rocm-compiler", "ROCm Compiler options")
    group.addoption(
        "--rock-dir",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Path to TheRock/ROCm install dir (contains bin/hipcc, lib/). "
            "Also read from ROCK_DIR or ROCM_TEST_THEROCK_ROCK_DIR env vars."
        ),
    )
    group.addoption(
        "--compiler-build-dir",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Output directory for compiled test binaries. "
            "Defaults to output/test-binaries/ (or rocm-test.toml [therock] build_dir)."
        ),
    )


@pytest.fixture(scope="session")
def rock_dir(request: pytest.FixtureRequest, framework_config) -> str:
    """Resolve the TheRock/ROCm installation path.

    Resolution order (first non-empty wins):
        1. ``--rock-dir`` CLI flag
        2. ``ROCK_DIR`` environment variable
        3. ``ROCM_TEST_THEROCK_ROCK_DIR`` environment variable
        4. ``framework_config.therock.rock_dir`` (rocm-test.toml or defaults)

    Returns:
        Absolute, realpath-resolved path string.

    Raises:
        pytest.fail.Exception: When no path is available from any source — this is
            a CI environment defect, not a resource shortage.
    """
    path: str | None = (
        request.config.getoption("--rock-dir", default=None)
        or os.environ.get("ROCK_DIR")
        or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
        or framework_config.therock.rock_dir
        or None
    )
    if not path:
        pytest.fail(
            "rock_dir not configured — pass --rock-dir=<path>, "
            "set ROCK_DIR=<path>, or set [therock] rock_dir in rocm-test.toml"
        )
    resolved = os.path.realpath(path)
    logger.info("rock_dir resolved to: %s", resolved)
    return resolved


@pytest.fixture(scope="session")
def compiler_build_dir(request: pytest.FixtureRequest, framework_config) -> str:
    """Return the root directory for compiled test binaries, creating it if needed.

    Resolution order (first non-empty wins):
        1. ``--compiler-build-dir`` CLI flag
        2. ``framework_config.therock.build_dir`` (rocm-test.toml or default)

    The default is ``output/test-binaries/``.  Component-specific sub-directories
    (e.g. ``output/test-binaries/compiler/``, ``output/test-binaries/multi_gpu/``)
    are created on demand by ``compile_binary(subdir=...)``.

    Returns:
        Path string (relative or absolute, as configured).
    """
    path: str = request.config.getoption("--compiler-build-dir", default=None) or framework_config.therock.build_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("compiler_build_dir: %s", path)
    return path


@pytest.fixture(scope="session")
def ld_path(rock_dir: str) -> dict:  # pylint: disable=redefined-outer-name
    """Build an LD_LIBRARY_PATH env dict for binaries linked against TheRock libs.

    Prepends ``{rock_dir}/lib`` to the existing ``LD_LIBRARY_PATH`` so that
    the TheRock-built ``libamdhip64.so`` and related libraries are found at
    runtime without requiring a system-level install.

    Args:
        rock_dir: Resolved path to the TheRock/ROCm installation.

    Returns:
        Dict ``{"LD_LIBRARY_PATH": "<rock_dir>/lib:<existing_path>"}``
        suitable for merging into a GPU execution env dict.
    """
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    lib_path = f"{rock_dir}/lib:{existing}" if existing else f"{rock_dir}/lib"
    return {"LD_LIBRARY_PATH": lib_path}


@pytest.fixture(scope="session")
def arch_lib_path(gpu_arch: str | None):
    """Return a callable that resolves an arch-specific library sub-directory path.

    Appends ``/<gpu_arch>`` to the base path when ``--gpu-arch`` is set so that
    ROCm library runtimes (e.g. hipBLASLt/Tensile) locate their arch-specific files.
    Returns ``str(base)`` unchanged when ``gpu_arch`` is ``None``.

    Args:
        gpu_arch: Architecture string from the ``gpu_arch`` fixture, or ``None``.

    Returns:
        Callable ``(base: str | pathlib.Path) -> str``.
    """
    import pathlib as _pathlib

    def _resolve(base) -> str:
        p = _pathlib.Path(base)
        return str(p / gpu_arch) if gpu_arch else str(p)

    return _resolve


@pytest.fixture(scope="session")
def compile_binary(
    rock_dir: str, compiler_build_dir: str, framework_config, gpu_arch: str | None, cmake_executor
):  # pylint: disable=redefined-outer-name
    """Return a compile factory pre-bound to rock_dir and compiler_build_dir.

    The returned callable wraps ``BinaryBuilder`` so that e2e tests can compile
    a HIP/C++ binary without importing ``BinaryBuilder`` or wiring up path
    resolution directly.  See ``BinaryBuilder.compile()`` for parameter details.

    Example::

        @pytest.fixture(scope="session")
        def allreduce_binary(compile_binary):
            return compile_binary(
                src="tests/e2e/multi_gpu/src/allreduce.cpp",
                output_name="allreduce",
                subdir="multi_gpu",
            )
    """

    def _compile(
        src: str,
        output_name: str,
        *,
        include_dirs: list[str] | None = None,
        extra_flags: list[str] | None = None,
        std: str = "c++17",
        opt: str = "-O2",
        arch: str | None = None,
        subdir: str | None = None,
    ) -> str:
        resolved_arch = arch if arch is not None else gpu_arch
        out_dir = os.path.join(compiler_build_dir, subdir) if subdir else compiler_build_dir
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(out_dir, output_name + ".build.log")
        return BinaryBuilder().compile(
            rocm_dir=rock_dir,
            src=src,
            output=os.path.join(out_dir, output_name),
            std=std,
            opt=opt,
            arch=resolved_arch,
            include_dirs=include_dirs,
            extra_flags=extra_flags,
            timeout=framework_config.therock.build_timeout_secs,
            inactivity_timeout=framework_config.therock.build_inactivity_timeout_secs,
            log_path=log_path,
            remote_executor=cmake_executor,
        )

    return _compile
