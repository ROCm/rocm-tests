# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared cmake configure+build helper for all e2e test areas.

Compiler enforcement is **not** centralised here — each conftest owns its policy
because the required vs. optional distinction varies per area.  This helper only
guarantees that ``-DROCM_PATH``, ``-DCMAKE_PREFIX_PATH``, and ``ROCM_PATH`` (env)
are always passed, and that ``cmake --build --parallel`` follows configure.
"""

from __future__ import annotations

import os
import pathlib
import shlex
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
    remote_executor=None,
    sync_dirs: list[str] | None = None,
) -> pathlib.Path:
    """Configure and build a cmake project against a TheRock / ROCm install.

    Always passes ``-DROCM_PATH`` and ``-DCMAKE_PREFIX_PATH``; the caller is
    responsible for resolving the compiler and deciding its enforcement policy.

    Args:
        src:               Path to the directory containing ``CMakeLists.txt``.
        build_dir:         Path where cmake should write build artefacts.
        rocm_path:         Resolved TheRock / ROCm install root.
        gpu_arch:          Architecture string passed as ``-D{gpu_arch_var}={gpu_arch}`` when set.
        gpu_arch_var:      CMake variable for the architecture (default ``"GPU_ARCH"``).
        compiler_args:     ``-DCMAKE_*_COMPILER=…`` flags; caller determines policy.
        extra_cmake_args:  Additional cmake ``-D`` flags.
        label:             Short name for assertion messages (defaults to basename of *src*).
        remote_executor:   ``SshExecutor`` to run cmake on a remote host; ``None`` for local.
        sync_dirs:         Local directories to SFTP to the remote host before cmake runs.

    Returns:
        ``pathlib.Path`` pointing to the cmake build directory.

    Raises:
        AssertionError: If cmake exits non-zero in local mode.
        RuntimeError:   If cmake fails on the remote host.
    """
    _label = label or pathlib.Path(src).name
    build_path = pathlib.Path(os.path.abspath(build_dir))

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

    build_cmd = ["cmake", "--build", str(build_path), "--parallel"]

    if remote_executor is not None:
        # Remote path: SFTP source dirs to remote, then run cmake via SSH.
        for d in sync_dirs or []:
            remote_executor.upload_tree(d)

        mk = remote_executor.run(f"mkdir -p {shlex.quote(str(build_path))}")
        if not mk.ok:
            raise RuntimeError(f"{_label} remote mkdir failed (exit={mk.exit_code}):\n{mk.stderr}")

        cfg = remote_executor.run(
            shlex.join(configure_cmd),
            timeout=600.0,
            env_overrides={"ROCM_PATH": rocm_path},
        )
        if not cfg.ok:
            raise RuntimeError(
                f"{_label} cmake configure failed on remote host (exit={cfg.exit_code}):\n"
                f"stdout: {cfg.stdout}\nstderr: {cfg.stderr}"
            )

        bld = remote_executor.run(shlex.join(build_cmd), timeout=900.0)
        if not bld.ok:
            raise RuntimeError(
                f"{_label} cmake build failed on remote host (exit={bld.exit_code}):\n"
                f"stdout: {bld.stdout}\nstderr: {bld.stderr}"
            )
    else:
        # Local path.
        build_path.mkdir(parents=True, exist_ok=True)
        cmake_env = {**os.environ, "ROCM_PATH": rocm_path}
        r = subprocess.run(configure_cmd, capture_output=True, text=True, env=cmake_env)
        assert r.returncode == 0, f"{_label} cmake configure failed:\n{r.stdout}\n{r.stderr}"
        r = subprocess.run(build_cmd, capture_output=True, text=True, env=cmake_env)
        assert r.returncode == 0, f"{_label} cmake build failed:\n{r.stdout}\n{r.stderr}"

    return build_path
