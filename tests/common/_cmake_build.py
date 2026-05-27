# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared cmake configure+build helper for all e2e test areas.

Usage
-----
Import ``cmake_build`` and ``find_rocm_clangpp`` in a conftest.py::

    from tests.e2e._cmake_build import cmake_build, find_rocm_clangpp

Design
------
Compiler enforcement is **not** centralised here — each conftest owns its
policy (``pytest.fail``, ``RuntimeError``, or optional guard) because the
required vs. optional distinction varies per area:

- ``rocprim`` — compiler is mandatory
  (``pytest.fail`` before cmake runs).
- ``hwq_heuristic``, ``hip_runtime`` core — compiler is optional
  (CMakeLists.txt falls back to hipcc if clang++ is absent).
- ``hipblaslt`` cmake — compiler is mandatory at cmake configure time
  (``find_program`` fails silently, then cmake errors on HIP language init).

This helper guarantees only what must be universal:

- ``-DROCM_PATH`` and ``-DCMAKE_PREFIX_PATH`` are **always** passed.
- ``ROCM_PATH`` is **always** set in the subprocess environment.
- ``cmake --build --parallel`` is run after configure.

All per-area compiler decisions (which flag to pass, whether to fail) stay in
the conftest that calls this helper.
"""

from __future__ import annotations

import os
import pathlib
import subprocess


def find_rocm_clangpp(rocm_path: str) -> pathlib.Path | None:
    """Probe candidate locations for ``amdclang++`` / ``clang++`` in a TheRock install.

    Mirrors the ``HINTS`` order used in
    ``hipblaslt/src/hipblaslt_heuristic_workspace/CMakeLists.txt``::

        ${ROCM_PATH}/lib/llvm/bin/clang++   (TheRock flattened layout)
        ${ROCM_PATH}/llvm/bin/clang++       (standard ROCm layout)
        ${ROCM_PATH}/bin/amdclang++         (some ROCm packaging variants)

    Args:
        rocm_path: Path to the TheRock / ROCm install root.

    Returns:
        The first candidate path that exists, or ``None`` if none are found.
    """
    candidates = [
        pathlib.Path(rocm_path) / "lib" / "llvm" / "bin" / "clang++",
        pathlib.Path(rocm_path) / "llvm" / "bin" / "clang++",
        pathlib.Path(rocm_path) / "bin" / "amdclang++",
    ]
    return next((p for p in candidates if p.exists()), None)


def cmake_build(
    src: str,
    build_dir: str,
    rocm_path: str,
    *,
    gpu_arch: str | None = None,
    gpu_arch_var: str = "GPU_ARCH",
    compiler_args: list[str] | None = None,
    extra_cmake_args: list[str] | None = None,
    label: str | None = None,
) -> pathlib.Path:
    """Configure and build a cmake project against a TheRock / ROCm install.

    Always passes ``-DROCM_PATH`` and ``-DCMAKE_PREFIX_PATH`` so that
    ``find_package(hip CONFIG REQUIRED)`` can locate ``hipConfig.cmake``
    under the ROCm install root (``lib/cmake/hip/``).

    The caller is responsible for:

    - Resolving and validating the compiler via :func:`find_rocm_clangpp`
      before calling this function.
    - Deciding whether a missing compiler is a ``pytest.fail``,
      ``RuntimeError``, or a no-op (optional).
    - Passing the resolved compiler via ``compiler_args``, e.g.
      ``["-DCMAKE_CXX_COMPILER=/path/to/clang++"]``.

    Args:
        src:               Path to the directory containing ``CMakeLists.txt``
                           (relative to repo root or absolute).
        build_dir:         Path where cmake should write build artefacts.
        rocm_path:         Resolved TheRock / ROCm install root
                           (value of ``rock_dir`` fixture, already ``realpath``-ed).
        gpu_arch:          GPU architecture string (e.g. ``"gfx942"``).
                           Passed as ``-D{gpu_arch_var}={gpu_arch}`` when set.
        gpu_arch_var:      CMake variable name for the architecture (default
                           ``"GPU_ARCH"``; use ``"AMDGPU_TARGETS"`` for rocprim
                           until its CMakeLists.txt is standardised).
        compiler_args:     Extra ``-DCMAKE_*_COMPILER=…`` flags to inject.
                           The caller determines which compilers to pass and
                           whether they are required.
        extra_cmake_args:  Any additional ``-D`` or other cmake flags.
        label:             Short name used in assertion messages (defaults to
                           the basename of *src*).

    Returns:
        ``pathlib.Path`` pointing to the cmake build directory.

    Raises:
        AssertionError: If cmake configure or build exits non-zero.
    """
    _label = label or pathlib.Path(src).name
    build_path = pathlib.Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)

    cmake_env = {**os.environ, "ROCM_PATH": rocm_path}

    configure_cmd: list[str] = [
        "cmake",
        "-S",
        str(pathlib.Path(src).resolve()),
        "-B",
        str(build_path),
        f"-DROCM_PATH={rocm_path}",
        f"-DCMAKE_PREFIX_PATH={rocm_path}",
    ]
    if compiler_args:
        configure_cmd.extend(compiler_args)
    if gpu_arch:
        configure_cmd.append(f"-D{gpu_arch_var}={gpu_arch}")
    if extra_cmake_args:
        configure_cmd.extend(extra_cmake_args)

    r = subprocess.run(configure_cmd, capture_output=True, text=True, env=cmake_env)
    assert r.returncode == 0, f"{_label} cmake configure failed:\n{r.stdout}\n{r.stderr}"

    build_cmd = ["cmake", "--build", str(build_path), "--parallel"]
    r = subprocess.run(build_cmd, capture_output=True, text=True, env=cmake_env)
    assert r.returncode == 0, f"{_label} cmake build failed:\n{r.stdout}\n{r.stderr}"

    return build_path
