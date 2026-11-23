# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HPC build utility helpers shared across HPC test suites.

Provides filesystem-safe ref-name conversion, executor-transparent path
existence checks, and a streaming build step runner.  These helpers were
previously duplicated verbatim in every HPC conftest
(kokkos, quda, raja, rochpl); extracting them here eliminates that drift.
"""

from __future__ import annotations

import logging
import os
import re
import shlex

import pytest

from framework.builder.binary_builder import find_rocm_clangpp
from framework.executors.background_process import _blocking_stream_run

logger = logging.getLogger(__name__)


def safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref.

    Replaces any character that is not alphanumeric, a dot, an underscore,
    or a hyphen with an underscore and strips leading/trailing underscores.

    Args:
        ref: Git tag, branch, or SHA string.

    Returns:
        A string safe for use in directory names.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def path_exists(path: str, cmake_executor=None) -> bool:
    """Return ``True`` when *path* is an existing file, locally or remotely.

    Args:
        path:           Filesystem path to check.
        cmake_executor: Remote build executor; ``None`` means local.
    """
    if cmake_executor is not None:
        return bool(cmake_executor.run(f"test -f {shlex.quote(path)}", timeout=30.0).ok)
    return os.path.isfile(path)


def run_build_step(
    cmd: str,
    *,
    cmake_executor,
    timeout: float,
    label: str,
    log_path: str,
) -> str:
    """Run a long build step with live streaming output; return combined output.

    HPC builds (configure + compile) run for several minutes, so buffered
    ``subprocess.run(capture_output=True)`` is unusable.  Remote steps stream
    through the SSH executor's ``stream=True`` path; local steps use the
    framework's ``_blocking_stream_run`` which forwards to the console in real
    time and also appends to *log_path*.

    Live console output requires ``pytest -s``; *log_path* is written either way.

    Args:
        cmd:            Shell command to run.
        cmake_executor: Remote build executor; ``None`` means local.
        timeout:        Seconds before the step is abandoned.
        label:          Short name used in log and error messages.
        log_path:       Absolute path for the per-step log file.

    Returns:
        Combined stdout + stderr of the build step.

    Raises:
        RuntimeError: When the build step exits non-zero.
    """
    logger.info("HPC build %s -> streaming to %s", label, log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"HPC {label} failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return (result.stdout or "") + (result.stderr or "")

    result = _blocking_stream_run(
        command=cmd,
        env=os.environ.copy(),
        cwd=None,
        timeout=timeout,
        stream_stdout=True,
        stream_stderr=True,
        log_path=log_path,
    )
    if not result.ok:
        raise RuntimeError(
            f"HPC {label} failed (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )
    return (result.stdout or "") + (result.stderr or "")


def resolve_hip_compilers(rock_dir: str, cmake_executor=None) -> tuple[str, str]:
    """Return ``(clang, clang++)`` paths from the ROCm install.

    Locally uses ``find_rocm_clangpp`` from the framework builder.  On a remote
    build node the ROCm tree lives on the far host, so the same three canonical
    paths are probed there.  Calls ``pytest.fail`` when neither is found.

    Args:
        rock_dir:       TheRock/ROCm install root.
        cmake_executor: Remote build executor; ``None`` means local.

    Returns:
        ``(cc_path, cxx_path)`` as absolute strings.
    """
    candidates = (
        f"{rock_dir}/lib/llvm/bin/clang++",
        f"{rock_dir}/llvm/bin/clang++",
        f"{rock_dir}/bin/amdclang++",
    )

    if cmake_executor is None:
        clangpp = find_rocm_clangpp(rock_dir)
        if clangpp is None:
            pytest.fail(f"ROCm clang++ not found under {rock_dir}")
        cxx = str(clangpp)
        return cxx.replace("clang++", "clang"), cxx

    for cxx in candidates:
        if cmake_executor.run(f"test -f {shlex.quote(cxx)}", timeout=15.0).ok:
            cc = cxx.replace("clang++", "clang")
            return cc, cxx

    pytest.fail(f"ROCm clang++ not found under {rock_dir} on the build node. " f"Searched: {', '.join(candidates)}")
