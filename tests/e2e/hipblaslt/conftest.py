# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixtures for tests/e2e/hipblaslt/.

Binary registry
---------------
This area mixes two build strategies:

1. ``compile_binary`` (hipcc direct) for pure C++ binaries:
   - ``mini_residual_app``
   - ``gemm_heuristic_workspace_budget``

2. CMake build for the ``.hip`` source that requires HIP language mode:
   - ``hipblaslt_heuristic_workspace`` (binary: ``hipblaslt-heuristic-test``)

CompileSpec pattern
-------------------
Each C++ binary is declared in ``_SPECS`` as a ``CompileSpec`` entry.
Adding a new binary is two steps:

    1. Add one ``CompileSpec`` entry to ``_SPECS``.
    2. Add one ``@pytest.fixture(scope="session")`` that calls ``_build()``.

All per-binary compile options (std, opt, arch, flags, include_dirs) live in
``_SPECS`` — never scattered across test files.

CMake pattern (hipblaslt_heuristic_workspace)
---------------------------------------------
The ``.hip`` source requires the HIP language property (``set_source_files_properties``),
which is not available through direct hipcc invocation.  The ``_hip_heuristic_cmake_build_dir``
fixture runs ``cmake -S ... -B ...`` followed by ``cmake --build``, the same pattern
as ``hwq_heuristic`` conftest.

Layout
------
tests/e2e/hipblaslt/
├── src/
│   ├── mini_residual_app.cpp
│   ├── gemm_heuristic_workspace_budget.cpp
│   ├── hipblaslt_shape_boundary.py
│   └── hipblaslt_heuristic_workspace/
│       ├── hipblaslt_heuristic_workspace.hip
│       └── CMakeLists.txt
├── conftest.py  (this file)
├── test_mini_residual_app.py
├── test_gemm_heuristic_workspace.py
├── test_hipblaslt_heuristic_workspace.py
└── test_hipblaslt_shape_boundary.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import pathlib
import shlex
import shutil

import pytest

from tests.common._cmake_build import cmake_build, find_rocm_clangpp

logger = logging.getLogger(__name__)

_SUBDIR = "hipblaslt"
_CMAKE_SRC_DIR = "tests/e2e/hipblaslt/src/hipblaslt_heuristic_workspace"
_CMAKE_BINARY_NAME = "hipblaslt-heuristic-test"


# ---------------------------------------------------------------------------
# CompileSpec — per-binary compile options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompileSpec:
    """Compile-time options for a single HIP/C++ binary.

    All source paths are relative to the repo root (where pytest is invoked).
    ``include_dirs`` defaults to an empty list; extend it for binaries that
    need extra headers beyond the system paths.

    Attributes:
        src:          Source file path, relative to repo root.
        output_name:  Binary filename written under ``output/test-binaries/hipblaslt/``.
        std:          C++ standard (default ``"c++17"``).
        opt:          Optimisation flag (default ``"-O2"``).
        arch:         GFX target for ``--offload-arch`` (``None`` = hipcc auto-detect).
        include_dirs: ``-I`` paths.
        flags:        Extra compiler flags as a single string.  Split by
                      ``shlex.split()`` before being forwarded to hipcc.
    """

    src: str
    output_name: str
    std: str = "c++17"
    opt: str = "-O2"
    arch: str | None = None
    include_dirs: list[str] = field(default_factory=list)
    flags: str = ""


def _specs(rock_dir: str) -> dict[str, CompileSpec]:
    """Build the binary registry, substituting ``rock_dir`` into include/link paths.

    Args:
        rock_dir: Resolved ROCm/TheRock install path (from ``rock_dir`` fixture).

    Returns:
        Mapping of output_name → CompileSpec.
    """
    include = f"{rock_dir}/include"
    return {
        "mini_residual_app": CompileSpec(
            src="tests/e2e/hipblaslt/src/mini_residual_app.cpp",
            output_name="mini_residual_app",
            std="c++17",
            opt="-O2",
            include_dirs=[include],
            flags=f"-L{rock_dir}/lib -lhipblaslt",
        ),
        "gemm_heuristic_workspace_budget": CompileSpec(
            src="tests/e2e/hipblaslt/src/gemm_heuristic_workspace_budget.cpp",
            output_name="gemm_heuristic_workspace_budget",
            std="c++17",
            opt="-O2",
            include_dirs=[include],
            flags=f"-L{rock_dir}/lib -lhipblaslt -lhipblas -lamdhip64",
        ),
    }


# ---------------------------------------------------------------------------
# Internal helper — eliminates the repetition in every fixture body.
# ---------------------------------------------------------------------------


def _build(compile_binary, name: str, rock_dir: str) -> str:
    """Look up *name* in the spec registry and delegate to the ``compile_binary`` factory.

    Args:
        compile_binary: Session-scoped factory fixture from ``builder_plugin``.
        name:           Key in the spec registry (equals ``output_name``).
        rock_dir:       Path to the ROCm/TheRock install; forwarded to ``_specs()``.

    Returns:
        Path to the compiled binary.
    """
    specs = _specs(rock_dir)
    spec = specs[name]
    return compile_binary(
        src=spec.src,
        output_name=spec.output_name,
        include_dirs=spec.include_dirs,
        std=spec.std,
        opt=spec.opt,
        arch=spec.arch,
        extra_flags=shlex.split(spec.flags) if spec.flags else None,
        subdir=_SUBDIR,
    )


# ---------------------------------------------------------------------------
# compile_binary fixtures (C++ sources via hipcc)
# ---------------------------------------------------------------------------


def _tree_snapshot(root: pathlib.Path, *, max_depth: int = 3, max_entries: int = 150) -> str:
    """Return an indented tree of *root* up to *max_depth* levels deep.

    Entries are capped at *max_entries* to avoid flooding the failure message.
    Directories that do not exist emit a single ``(not found)`` line.

    Args:
        root:        Directory to walk.
        max_depth:   Maximum recursion depth (1 = direct children only).
        max_entries: Hard cap on total lines emitted.

    Returns:
        Multi-line string ready to embed in a ``pytest.fail`` message.
    """
    if not root.exists():
        return f"  {root}  (not found)"

    lines: list[str] = [f"  {root}/"]
    count = 0
    for entry in sorted(root.rglob("*")):
        depth = len(entry.relative_to(root).parts)
        if depth > max_depth:
            continue
        indent = "    " + "  " * (depth - 1)
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"{indent}{entry.name}{suffix}")
        count += 1
        if count >= max_entries:
            lines.append(f"    ... (truncated at {max_entries} entries)")
            break
    return "\n".join(lines)


def _check_hipblaslt_headers(rock_dir: str) -> None:
    """Fail with actionable message if hipblaslt dev headers are missing.

    hipblaslt/hipblaslt.h must be present under ``{rock_dir}/include/`` for direct
    hipcc compilation.  When absent, the error from hipcc is cryptic; this guard
    surfaces a clear message with searched paths and a directory snapshot of
    ``include/``, ``lib/``, and ``bin/`` so the missing artifact can be identified
    immediately.

    Args:
        rock_dir: Path to the ROCm/TheRock install root.
    """
    hipblaslt_h = pathlib.Path(rock_dir) / "include" / "hipblaslt" / "hipblaslt.h"
    if not hipblaslt_h.exists():
        searched = [
            pathlib.Path(rock_dir) / "include" / "hipblaslt",
            pathlib.Path(rock_dir) / "include",
        ]
        root = pathlib.Path(rock_dir)
        snapshot = "\n\n".join(f"[{name}]\n{_tree_snapshot(root / name)}" for name in ("include", "lib", "bin"))
        pytest.fail(
            "hipblaslt/hipblaslt.h not found. Searched:\n"
            + "\n".join(f"  {p}" for p in searched)
            + "\n\ntherock_build dir snapshot (include / lib / bin):\n"
            + snapshot
            + "\n\nhipBLASLt headers are in blas_lib — verify the --blas artifact was downloaded "
            "and extracted correctly (pass --blas to install_rocm_from_artifacts.py)."
        )


@pytest.fixture(scope="session")
def mini_residual_app_binary(compile_binary, rock_dir: str, cmake_executor) -> str:
    """Compile mini_residual_app.cpp → binary path.  Used by test_mini_residual_app.py."""
    if cmake_executor is None:
        _check_hipblaslt_headers(rock_dir)
    return _build(compile_binary, "mini_residual_app", rock_dir)


@pytest.fixture(scope="session")
def gemm_heuristic_workspace_budget_binary(compile_binary, rock_dir: str, cmake_executor) -> str:
    """Compile gemm_heuristic_workspace_budget.cpp → binary path.  Used by test_gemm_heuristic_workspace.py."""
    if cmake_executor is None:
        _check_hipblaslt_headers(rock_dir)
    return _build(compile_binary, "gemm_heuristic_workspace_budget", rock_dir)


# ---------------------------------------------------------------------------
# CMake fixture for the .hip source (hipblaslt_heuristic_workspace)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _hip_heuristic_cmake_build_dir(gpu_arch: str | None, rock_dir: str, compiler_build_dir: str, cmake_executor) -> str:
    """Build hipblaslt-heuristic-test via CMake; return build directory path.

    Runs ``cmake -S <src> -B <build> -DROCM_PATH=<rock_dir> [-DGPU_ARCH=<arch>]``
    followed by ``cmake --build <build> --parallel``.  Both steps raise
    ``AssertionError`` on failure so pytest reports them as ``ERROR`` on every
    test that depends on this fixture.

    In ``--remote-node`` mode, cmake runs on the remote host via ``cmake_executor``
    (an ``SshExecutor``).  In local mode, cmake must be present in ``PATH``; the
    fixture skips when it is absent rather than raising ``FileNotFoundError``.

    Args:
        gpu_arch:            Target GPU architecture from the ``gpu_arch`` fixture (``--gpu-arch``).
        rock_dir:            Path to the ROCm/TheRock install (``--rock-dir`` / ``ROCK_DIR``).
        compiler_build_dir:  Session-scoped output root (``output/test-binaries/`` by default).
        cmake_executor:      Session-scoped ``SshExecutor`` for remote cmake; ``None`` for local builds.

    Returns:
        Absolute path to the CMake build directory containing the binary.
    """
    if cmake_executor is None and not shutil.which("cmake"):
        pytest.skip("cmake not found in PATH — install cmake to run this test locally")

    rocm_path = os.path.realpath(rock_dir)
    build_dir = os.path.abspath(os.path.join(compiler_build_dir, "hipblaslt_heuristic_workspace", "build"))

    # clang++ is used as the host CXX compiler; the CMakeLists.txt handles HIP compiler
    # detection (find_program amdclang++) independently. Both may arrive as -D flags.
    clangpp = find_rocm_clangpp(rocm_path)
    compiler_args = [f"-DCMAKE_CXX_COMPILER={clangpp}"] if clangpp else []

    cmake_build(
        _CMAKE_SRC_DIR,
        build_dir,
        rocm_path,
        gpu_arch=gpu_arch,
        compiler_args=compiler_args,
        label="hipblaslt_heuristic_workspace",
        remote_executor=cmake_executor,
        sync_dirs=[os.path.abspath(_CMAKE_SRC_DIR)],
    )
    return build_dir


@pytest.fixture(scope="session")
def tensile_lib_path(rock_dir: str, gpu_arch: str | None, arch_lib_path, cmake_executor) -> str:
    """Resolve and validate the hipBLASLt Tensile library directory path.

    Constructs the per-arch Tensile kernel directory path and validates that
    ``TensileLibrary_lazy_<arch>.dat`` is present before any test runs.

    - **Local mode** (``cmake_executor is None``): uses ``pathlib.Path.exists()``
      — fast and requires no subprocess.
    - **Remote-node mode**: runs ``test -f <path>`` via ``cmake_executor`` so the
      check executes on the host where the ROCm stack is actually installed,
      not on the pytest coordinator.

    When ``gpu_arch`` is ``None`` the base library directory is returned without
    validation (runtime auto-detection fallback — may fail if files only exist
    under an arch sub-directory).

    Args:
        rock_dir:       Resolved ROCm/TheRock install path.
        gpu_arch:       Target GPU architecture string (e.g. ``"gfx90a"``), or ``None``.
        arch_lib_path:  Session callable that appends ``/<arch>`` to a base path.
        cmake_executor: Session-scoped ``SshExecutor`` when ``--remote-node`` is
                        active; ``None`` in local mode.

    Returns:
        Absolute path string to the resolved Tensile library directory.
    """
    library_base = pathlib.Path(rock_dir) / "lib" / "hipblaslt" / "library"
    tensile_lib = arch_lib_path(library_base)

    if gpu_arch:
        tensile_dat = f"{tensile_lib}/TensileLibrary_lazy_{gpu_arch}.dat"
        fail_msg = (
            f"hipBLASLt Tensile kernels missing for arch {gpu_arch!r}: {tensile_dat}\n"
            "Install the BLAS artifact package (pass --blas to install_rocm_from_artifacts.py)."
        )
        if cmake_executor is None:
            if not pathlib.Path(tensile_dat).exists():
                pytest.fail(fail_msg)
        else:
            result = cmake_executor.run(f"test -f {shlex.quote(tensile_dat)}")
            if not result.ok:
                pytest.fail(fail_msg)

    return tensile_lib


@pytest.fixture(scope="session")
def hipblaslt_heuristic_workspace_binary(_hip_heuristic_cmake_build_dir: str, cmake_executor) -> str:
    """Return absolute path to the compiled hipblaslt-heuristic-test binary.

    Args:
        _hip_heuristic_cmake_build_dir: Build directory from the shared CMake fixture.
        cmake_executor:                 ``SshExecutor`` when running remotely, ``None`` for local.

    Returns:
        Absolute path to the ``hipblaslt-heuristic-test`` binary.
    """
    binary = os.path.join(_hip_heuristic_cmake_build_dir, _CMAKE_BINARY_NAME)
    if cmake_executor is None:
        assert os.path.isfile(
            binary
        ), f"hipblaslt_heuristic_workspace: binary not found at {binary} after successful build"
    return binary
