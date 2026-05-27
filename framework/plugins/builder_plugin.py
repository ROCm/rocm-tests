# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
builder_plugin.py -- Compiler path options, build-directory fixtures, and generic
                     binary compilation factory for any e2e test component.

Responsibilities:
    - Register ``--rock-dir`` and ``--compiler-build-dir`` pytest CLI options.
    - Provide ``rock_dir`` session fixture that resolves the TheRock/ROCm
      installation path from (highest to lowest priority):
          1. ``--rock-dir`` CLI flag
          2. ``ROCK_DIR`` environment variable
          3. ``ROCM_TEST_THEROCK_ROCK_DIR`` environment variable
          4. ``framework_config.therock.rock_dir`` (rocm-test.toml / defaults)
    - Provide ``compiler_build_dir`` session fixture that creates the binary
      output directory on first use (default: ``output/test-binaries/``).
    - Provide ``ld_path`` session fixture that prepends ``{rock_dir}/lib`` to
      ``LD_LIBRARY_PATH`` for running TheRock-linked binaries.
    - Provide ``compile_binary`` session fixture: a factory callable that wraps
      ``BinaryBuilder`` so any e2e component can compile a HIP/C++ binary
      without importing ``BinaryBuilder`` directly.

Loaded automatically via pytest_plugins in conftest.py.

pytest options added:
    --rock-dir PATH           Path to TheRock/ROCm installation.
    --compiler-build-dir PATH Override for compiled-binary output directory.

Example usage in any e2e conftest::

    @pytest.fixture(scope="session")
    def my_kernel_binary(compile_binary):
        return compile_binary(
            src="tests/e2e/myarea/src/my_kernel.cpp",
            output_name="my_kernel",
            include_dirs=["tests/common/include"],
            subdir="myarea",    # → output/test-binaries/myarea/my_kernel
            arch="gfx942",      # optional
        )
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from framework.builder.binary_builder import BinaryBuilder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


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

    Generic — any e2e component that runs TheRock-linked binaries needs this,
    not just compiler tests.

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

    Many ROCm libraries (hipBLASLt/Tensile, etc.) store arch-specific loadable
    files — ``.dat``, ``.hsaco``, ``.co`` — under a per-arch subdirectory:

    .. code-block:: text

        <base>/<arch>/TensileLibrary_lazy_<arch>.dat
        <base>/<arch>/Kernels.so-000-<arch>.hsaco

    When ``--gpu-arch`` is supplied the callable appends ``/<arch>`` to *base*
    so the runtime finds its files without relying on internal auto-detection.
    When ``--gpu-arch`` is absent the base path is returned unchanged (runtime
    fallback; may fail if files only exist under an arch subdir).

    Usage in a test::

        def test_foo(target_executor, arch_lib_path, rock_dir):
            tensile_base = pathlib.Path(rock_dir) / "lib" / "hipblaslt" / "library"
            tensile_lib = arch_lib_path(tensile_base)
            result = target_executor.run(
                f"env HIPBLASLT_TENSILE_LIBPATH={tensile_lib} ./my_binary"
            )

    Args:
        gpu_arch: Architecture string from the session-scoped ``gpu_arch`` fixture
                  (e.g. ``"gfx90a"``), or ``None``.

    Returns:
        Callable ``(base: str | pathlib.Path) -> str`` that appends ``/<arch>``
        when *gpu_arch* is set, or returns ``str(base)`` otherwise.
    """
    import pathlib as _pathlib

    def _resolve(base) -> str:
        p = _pathlib.Path(base)
        return str(p / gpu_arch) if gpu_arch else str(p)

    return _resolve


@pytest.fixture(scope="session")
def compile_binary(
    rock_dir: str, compiler_build_dir: str, framework_config, gpu_arch: str | None
):  # pylint: disable=redefined-outer-name
    """Return a compile factory pre-bound to rock_dir and compiler_build_dir.

    The returned callable wraps ``BinaryBuilder`` so that any e2e test can
    compile a HIP/C++ binary without importing ``BinaryBuilder`` directly or
    wiring up path-resolution boilerplate.

    Factory signature::

        compile_binary(
            src,
            output_name,
            *,
            include_dirs=None,
            extra_flags=None,
            std="c++17",
            opt="-O2",
            arch=None,
            subdir=None,
        ) -> str

    Args (of the returned factory):
        src:          Path to the C++ source file (relative to repo root or
                      absolute).
        output_name:  Base filename for the compiled binary.
        include_dirs: Extra ``-I`` paths forwarded to hipcc.
        extra_flags:  Verbatim extra compiler flags forwarded to hipcc.
        std:          C++ standard (default ``"c++17"``).
        opt:          Optimisation flag (default ``"-O2"``).
        arch:         GPU arch target, e.g. ``"gfx942"`` → ``--offload-arch=gfx942``.
                      ``None`` falls back to ``--gpu-arch`` CLI option; if that is
                      also absent hipcc auto-detects from the ROCm device list.
        subdir:       Sub-directory under ``compiler_build_dir`` for this
                      component's binaries (e.g. ``"compiler"``, ``"multi_gpu"``).
                      Creates ``output/test-binaries/<subdir>/`` on first call.

    Returns (factory):
        Path to the compiled binary.

    Example usage in an e2e conftest::

        @pytest.fixture(scope="session")
        def allreduce_binary(compile_binary):
            return compile_binary(
                src="tests/e2e/multi_gpu/src/allreduce.cpp",
                output_name="allreduce",
                include_dirs=["tests/common/include"],
                subdir="multi_gpu",   # → output/test-binaries/multi_gpu/allreduce
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
        )

    return _compile
