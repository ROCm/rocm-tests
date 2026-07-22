# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
binary_builder.py -- Process-safe, incremental C++ binary compilation for test suites.

Design
------
- **CPU/GPU env separation**: all GPU device-selection env vars are stripped from the
  compiler subprocess; GPU execution is handled by the caller's executor fixture.
- **xdist parallel safety**: compilation is serialised via ``fcntl`` file lock; later
  workers skip if the binary is already current (mtime check).
- **Live streaming + dual-log**: stdout/stderr stream to the console and to
  ``<output>.build.log`` simultaneously; the log path appears in ``AssertionError``.
- **Two timeouts**: wall-clock ``timeout`` (default 7200 s) and ``inactivity_timeout``
  (default 600 s); SIGTERM then SIGKILL on expiry.

Usage::

    @pytest.fixture(scope="session")
    def my_binary(rock_dir, compiler_build_dir):
        return BinaryBuilder().compile(
            rocm_dir=rock_dir,
            src="tests/e2e/myarea/src/my_kernel.cpp",
            output=os.path.join(compiler_build_dir, "my_kernel"),
            arch="gfx942",
        )
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
import errno
import fcntl
import glob
import hashlib
import json
import logging
import os
import pathlib
import posixpath
import re
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time

import pytest

from framework.common.workspace_layout import REMOTE_WORKSPACE_DIR, category_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU device-selection env vars stripped from the compilation environment.
# These are set by the pytest GPU allocation machinery (gpu_plugin, gpu_fixture)
# and must not leak into hipcc — compilation is CPU-only.
# ---------------------------------------------------------------------------
_GPU_ENV_KEYS: frozenset[str] = frozenset(
    {
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "CUDA_VISIBLE_DEVICES",
        "ROCM_VISIBLE_DEVICES",
    }
)

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

_SIGTERM_GRACE_SECS: float = 30.0


def _write_build_skip_log(log_path: str, src: str, binary: str) -> None:
    """Write a status entry to *log_path* when compilation is skipped.

    Opens the file in write mode so each session reflects the current state
    (previous compilation output is replaced by this skip notice).

    Args:
        log_path: Path to the build log file.
        src:      Source file that was checked.
        binary:   Compiled binary whose mtime was newer than *src*.
    """
    pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H:%M:%S")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"[{ts}] BinaryBuilder: binary is up-to-date, skipped recompilation\n")
        fh.write(f"  Source: {src}\n")
        fh.write(f"  Binary: {binary}\n")


def _write_chunk(chunk: bytes, log_fh) -> None:
    """Write *chunk* to the live console and, when open, to *log_fh*."""
    sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()
    if log_fh is not None:
        log_fh.write(chunk)
        log_fh.flush()


def _graceful_kill(proc: subprocess.Popen, grace_secs: float = _SIGTERM_GRACE_SECS) -> None:
    """Send SIGTERM to *proc*, then SIGKILL if it has not exited within *grace_secs*."""
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=grace_secs)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class BinaryBuilder:
    """Compile C++ test binaries with hipcc on the CPU.

    Guarantees:
        - Compilation subprocess runs in a CPU-only environment (no GPU device
          vars).
        - Safe to call from concurrent pytest-xdist workers via file locking.
        - Skips recompilation if the binary is already newer than the source.
        - Streams compiler output to the console and a ``.build.log`` file
          simultaneously so long compilations are visible in real time.
        - Applies wall-clock and inactivity timeouts; kills the compiler with
          SIGTERM → SIGKILL on breach.
        - Raises ``AssertionError`` on failure → pytest reports ERROR status.
    """

    def compile(
        self,
        src: str,
        output: str,
        hipcc: str | None = None,
        rocm_dir: str | None = None,
        std: str = "c++17",
        opt: str = "-O2",
        arch: str | None = None,
        include_dirs: list[str] | None = None,
        extra_flags: list[str] | None = None,
        timeout: float = 7200.0,
        inactivity_timeout: float = 600.0,
        log_path: str | None = None,
        remote_executor=None,
    ) -> str:
        """Compile *src* to *output* using hipcc (CPU-only, xdist-safe).

        Supply exactly one of *hipcc* (absolute path) or *rocm_dir* (install root;
        hipcc derived as ``{rocm_dir}/bin/hipcc``).

        Args:
            src:                  Path to the C++ source file.
            output:               Destination path for the compiled binary.
            hipcc:                Absolute path to the hipcc binary (mutually exclusive with *rocm_dir*).
            rocm_dir:             ROCm/TheRock install root (mutually exclusive with *hipcc*).
            std:                  C++ standard (default ``"c++17"``).
            opt:                  Optimisation flag (default ``"-O2"``).
            arch:                 ``--offload-arch`` target; ``None`` lets hipcc auto-detect.
            include_dirs:         Additional ``-I`` paths.
            extra_flags:          Extra compiler flags appended verbatim before ``-o``.
            timeout:              Wall-clock seconds before the compiler is killed (default 7200).
            inactivity_timeout:   Seconds of silence before the process is killed (default 600).
            log_path:             Build log path; merged stdout+stderr written here and to console.
            remote_executor:      ``SshExecutor`` to run hipcc remotely; ``None`` for local.

        Returns:
            Path to the compiled binary (same as *output*).

        Raises:
            ValueError:     Neither or both of *hipcc* / *rocm_dir* supplied.
            TimeoutError:   Compilation exceeded *timeout* or *inactivity_timeout*.
            AssertionError: Compilation exited non-zero (pytest reports ``ERROR`` status).
            RuntimeError:   Remote compilation failed.
        """
        if remote_executor is not None:
            return self._remote_compile(
                src=src,
                output=output,
                hipcc=hipcc,
                rocm_dir=rocm_dir,
                std=std,
                opt=opt,
                arch=arch,
                include_dirs=include_dirs,
                extra_flags=extra_flags,
                timeout=timeout,
                remote_executor=remote_executor,
            )

        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        lock_path = output + ".lock"

        with open(lock_path, "w", encoding="utf-8") as lock_fh:
            # Exclusive lock with non-blocking poll; blocks other xdist workers
            # until this worker compiles or confirms the binary is current.
            lock_timeout = timeout if timeout is not None else 7200.0
            deadline = time.monotonic() + lock_timeout
            while True:
                try:
                    fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN):
                        raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"BinaryBuilder: could not acquire build lock for "
                            f"{output!r} within {lock_timeout}s — another "
                            "worker may be hung."
                        ) from exc
                    time.sleep(5)

            try:
                if self._is_up_to_date(src, output):
                    logger.info("BinaryBuilder : binary up-to-date, skipping — %s", output)
                    if log_path:
                        _write_build_skip_log(log_path, src, output)
                    return output

                cmd = self._build_cmd(hipcc, rocm_dir, src, output, std, opt, arch, include_dirs, extra_flags)
                compile_env = self._cpu_env()

                logger.info("BinaryBuilder : compiling — %s", " ".join(cmd))

                self._stream_compile(
                    cmd=cmd,
                    env=compile_env,
                    src=src,
                    timeout=timeout,
                    inactivity_timeout=inactivity_timeout,
                    log_path=log_path,
                )

                logger.info("BinaryBuilder : compiled → %s", output)
                return output

            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _remote_compile(
        self,
        src: str,
        output: str,
        hipcc: str | None,
        rocm_dir: str | None,
        std: str,
        opt: str,
        arch: str | None,
        include_dirs: list[str] | None,
        extra_flags: list[str] | None,
        timeout: float,
        remote_executor,
    ) -> str:
        """Compile *src* on a remote host via *remote_executor*.

        Uploads *src* and all *include_dirs* to the remote at the same absolute
        path, creates the output directory on the remote, then runs hipcc via SSH.

        Args:
            src:             Source file path (relative or absolute).
            output:          Destination binary path (same on remote).
            hipcc:           Absolute hipcc path (mutually exclusive with *rocm_dir*).
            rocm_dir:        ROCm install root; hipcc derived as ``{rocm_dir}/bin/hipcc``.
            std:             C++ standard flag value.
            opt:             Optimisation flag.
            arch:            GPU architecture for ``--offload-arch``, or ``None``.
            include_dirs:    Extra ``-I`` directories to upload and forward.
            extra_flags:     Additional compiler flags.
            timeout:         SSH command timeout in seconds.
            remote_executor: ``SshExecutor`` instance for SSH/SFTP operations.

        Returns:
            *output* path (binary lives on the remote at that path).

        Raises:
            RuntimeError: If any remote SSH command fails.
        """
        output_abs = os.path.abspath(output)
        src_str = str(src)

        if src_str.startswith(remote_executor.remote_workspace_root()):
            remote_src = src_str
        else:
            src_abs = os.path.abspath(src)
            remote_src_dir = remote_executor.upload_tree(os.path.dirname(src_abs))
            remote_src = posixpath.join(remote_src_dir, os.path.basename(src_abs))
        remote_output: str = str(remote_executor.workspace_path_for(output_abs, category="work"))
        remote_out_dir = posixpath.dirname(remote_output)
        remote_include_dirs: list[str] = []
        remote_rocm_dir = posixpath.normpath(rocm_dir) if rocm_dir else ""
        for inc in include_dirs or []:
            inc_str = str(inc)
            if inc_str.startswith(remote_executor.remote_workspace_root()) or (
                remote_rocm_dir
                and (
                    posixpath.normpath(inc_str) == remote_rocm_dir
                    or posixpath.normpath(inc_str).startswith(remote_rocm_dir.rstrip("/") + "/")
                )
            ):
                remote_include_dirs.append(inc_str)
            else:
                remote_include_dirs.append(remote_executor.upload_tree(os.path.abspath(inc_str)))

        mk = remote_executor.run(f"mkdir -p {shlex.quote(remote_out_dir)}")
        if not mk.ok:
            raise RuntimeError(f"BinaryBuilder remote mkdir failed (exit={mk.exit_code}):\n{mk.stderr}")

        cmd = self._build_cmd(
            hipcc,
            rocm_dir,
            remote_src,
            remote_output,
            std,
            opt,
            arch,
            remote_include_dirs,
            extra_flags,
        )
        result = remote_executor.run(shlex.join(cmd), timeout=timeout)
        if not result.ok:
            raise AssertionError(
                f"Compilation of '{src}' failed on remote host (exit={result.exit_code}).\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        logger.info("BinaryBuilder : remote compiled → %s", remote_output)
        return remote_output

    @staticmethod
    def _stream_compile(
        cmd: list[str],
        env: dict[str, str],
        src: str,
        timeout: float | None,
        inactivity_timeout: float,
        log_path: str | None,
    ) -> None:
        """Run *cmd* as a subprocess, streaming output to console + log file.

        Applies both a wall-clock timeout and an inactivity (no-output) timeout.
        Kills the process with SIGTERM → SIGKILL on either breach.

        Args:
            cmd:                Command and arguments list.
            env:                Full environment for the subprocess.
            src:                Source file path (used in error messages only).
            timeout:            Wall-clock seconds before forced termination.
            inactivity_timeout: Seconds of no output before forced termination.
            log_path:           If given, build output is also written here.

        Raises:
            TimeoutError:   Wall-clock or inactivity limit exceeded.
            AssertionError: Process exited with a non-zero return code.
        """
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merged — single chronological log
            close_fds=True,
        )
        os.set_blocking(process.stdout.fileno(), False)  # type: ignore[union-attr]

        buf = b""
        t_start = time.monotonic()
        t_last_output = t_start

        with contextlib.ExitStack() as stack:
            log_fh = stack.enter_context(open(log_path, "wb")) if log_path else None
            while True:
                now = time.monotonic()
                elapsed = now - t_start
                since_last = now - t_last_output

                if timeout is not None and elapsed >= timeout:
                    _graceful_kill(process)
                    raise TimeoutError(
                        f"Compilation of '{src}' timed out after {timeout:.0f}s "
                        f"(wall-clock)." + (f"  Log: {log_path}" if log_path else "")
                    )

                if since_last >= inactivity_timeout:
                    _graceful_kill(process)
                    raise TimeoutError(
                        f"Compilation of '{src}' stalled: no compiler output for "
                        f"{inactivity_timeout:.0f}s." + (f"  Log: {log_path}" if log_path else "")
                    )

                # Cap select() wait so we don't sleep past the nearest deadline.
                select_wait = min(5.0, inactivity_timeout - since_last)
                if timeout is not None:
                    select_wait = min(select_wait, timeout - elapsed)
                select_wait = max(select_wait, 0.1)

                if process.stdout is not None and select.select([process.stdout], [], [], select_wait)[0]:
                    chunk = process.stdout.read()
                    if chunk:
                        t_last_output = time.monotonic()
                        _write_chunk(chunk, log_fh)
                        buf += chunk

                if process.poll() is not None:
                    tail = process.stdout.read()  # type: ignore[union-attr]
                    if tail:
                        _write_chunk(tail, log_fh)
                        buf += tail
                    break

        rc = process.wait()
        if rc != 0:
            log_hint = f"\nFull build log: {log_path}" if log_path else ""
            output_hint = "" if log_path else f"\n{_ANSI_RE.sub('', buf.decode(errors='replace'))}"
            raise AssertionError(f"Compilation of '{src}' failed (exit={rc}).{log_hint}{output_hint}")

    @staticmethod
    def _build_cmd(
        hipcc: str | None,
        rocm_dir: str | None,
        src: str,
        output: str,
        std: str,
        opt: str,
        arch: str | None,
        include_dirs: list[str] | None,
        extra_flags: list[str] | None,
    ) -> list[str]:
        """Assemble the hipcc command list."""
        if hipcc is None and rocm_dir is None:
            raise ValueError("BinaryBuilder.compile(): supply either 'hipcc' or 'rocm_dir'")
        if hipcc is not None and rocm_dir is not None:
            raise ValueError("BinaryBuilder.compile(): supply exactly one of 'hipcc' or 'rocm_dir', not both")
        resolved_hipcc: str = hipcc or os.path.join(rocm_dir, "bin", "hipcc")  # type: ignore[arg-type]

        cmd = [resolved_hipcc, src, f"-std={std}", opt]
        for d in include_dirs or []:
            cmd += ["-I", d]
        if arch:
            cmd += [f"--offload-arch={arch}"]
        cmd += extra_flags or []
        cmd += ["-o", output]
        return cmd

    @staticmethod
    def _cpu_env() -> dict[str, str]:
        """Return the current environment with GPU device-selection vars removed.

        Stripping these vars ensures hipcc is never bound to a specific GPU
        ordinal that was set by the test framework for GPU execution isolation.
        The compiler only needs access to PATH (for linker, assembler), and
        standard system libraries — no GPU runtime context is required.
        """
        env = {k: v for k, v in os.environ.items() if k not in _GPU_ENV_KEYS}
        stripped = [k for k in _GPU_ENV_KEYS if k in os.environ]
        if stripped:
            logger.debug(
                "BinaryBuilder : stripped GPU env vars from compile env: %s",
                ", ".join(stripped),
            )
        return env

    @staticmethod
    def _is_up_to_date(src: str, binary: str) -> bool:
        """Return True if *binary* exists and its mtime is newer than *src*."""
        if not os.path.isfile(binary):
            return False
        return os.path.getmtime(binary) > os.path.getmtime(src)


# ---------------------------------------------------------------------------
# CMake helpers (formerly tests/common/_cmake_build.py)
# ---------------------------------------------------------------------------


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


def resolve_parallel_jobs(parallel_jobs: int | None = None, remote_executor=None) -> int:
    """Resolve a bounded job count for ``cmake --build --parallel N``.

    ``cmake --build --parallel`` with no number maps to an *unbounded* ``-j`` for
    the Makefiles generator, which can over-subscribe cores and OOM on heavy C++
    builds (GROMACS, AMReX).  This returns an explicit count instead.

    Args:
        parallel_jobs:   Explicit job count; used as-is when a positive integer.
        remote_executor: ``SshExecutor`` to query ``nproc`` on the remote host, or
                         ``None`` to use the local CPU count.

    Returns:
        A positive integer job count (falls back to 4 when undetectable).
    """
    if parallel_jobs is not None and parallel_jobs > 0:
        return parallel_jobs
    if remote_executor is not None:
        try:
            res = remote_executor.run("nproc", timeout=30.0)
            if res.ok and res.stdout.strip().isdigit():
                return max(1, int(res.stdout.strip()))
        except Exception:  # pragma: no cover - defensive; fall back to a safe default
            pass
        return 4
    return os.cpu_count() or 4


def cmake_build(  # noqa: C901
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
    parallel_jobs: int | None = None,
    target: str | None = None,
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
        parallel_jobs:     Explicit ``--parallel`` job count; ``None`` resolves to the
                           local CPU count (or remote ``nproc``).
        target:            CMake build target passed as ``--target <target>``; ``None`` builds
                           the default ``ALL`` target.

    Returns:
        ``pathlib.Path`` pointing to the cmake build directory.

    Raises:
        AssertionError: If cmake exits non-zero in local mode.
        RuntimeError:   If cmake fails on the remote host.
    """
    _label = label or pathlib.Path(src).name

    if remote_executor is not None:
        # In-tree builds ship sources via *sync_dirs* into the managed remote
        # workspace. External builds usually pass an already-remote source path
        # returned by clone_repo(); their build dir still belongs under the same
        # managed workspace rather than the remote shell's arbitrary cwd.
        if sync_dirs:
            src_path_str = remote_executor.remote_path_for(os.path.abspath(src))
            build_dir_str = str(build_dir)
            if build_dir_str.startswith(remote_executor.remote_workspace_root()):
                build_path = pathlib.Path(build_dir_str)
            else:
                build_path = pathlib.Path(remote_executor.remote_path_for(os.path.abspath(build_dir)))
        else:
            src_path_str = str(src)
            build_path = pathlib.Path(_remote_abspath(pathlib.Path(build_dir), remote_executor))
        build_path_str = str(build_path)
    else:
        build_path = pathlib.Path(os.path.abspath(build_dir))
        src_path_str = str(pathlib.Path(src).resolve())
        build_path_str = str(build_path)

    configure_cmd: list[str] = [
        "cmake",
        "-S",
        src_path_str,
        "-B",
        build_path_str,
        f"-DROCM_PATH={rocm_path}",
        f"-DCMAKE_PREFIX_PATH={rocm_path}",
    ]
    if compiler_args:
        configure_cmd.extend(compiler_args)
    if gpu_arch:
        configure_cmd.append(f"-D{gpu_arch_var}={gpu_arch}")
    if extra_cmake_args:
        configure_cmd.extend(extra_cmake_args)

    jobs = resolve_parallel_jobs(parallel_jobs, remote_executor)
    build_cmd = ["cmake", "--build", str(build_path), "--parallel", str(jobs)]
    if target is not None:
        build_cmd += ["--target", target]

    if remote_executor is not None:
        for d in sync_dirs or []:
            remote_executor.upload_tree(d)
        mk = remote_executor.run(f"mkdir -p {shlex.quote(build_path_str)}")
        if not mk.ok:
            raise RuntimeError(f"{_label} remote mkdir failed (exit={mk.exit_code}):\n{mk.stderr}")
        cfg = remote_executor.run(
            shlex.join(configure_cmd),
            timeout=600.0,
            env_overrides={"ROCM_PATH": rocm_path},
        )
        if not cfg.ok:
            stale_cache = _cmake_cache_mismatch(cfg.stdout + "\n" + cfg.stderr)
            if stale_cache:
                remote_executor.run(f"rm -rf {shlex.quote(build_path_str)} && mkdir -p {shlex.quote(build_path_str)}")
                cfg = remote_executor.run(
                    shlex.join(configure_cmd),
                    timeout=600.0,
                    env_overrides={"ROCM_PATH": rocm_path},
                )
            if not cfg.ok:
                retry_note = " after clean cache retry" if stale_cache else ""
                raise RuntimeError(
                    f"{_label} cmake configure failed on remote host{retry_note} (exit={cfg.exit_code}):\n"
                    f"stdout: {cfg.stdout}\nstderr: {cfg.stderr}"
                )
        bld = remote_executor.run(shlex.join(build_cmd), timeout=7200.0, stream=True)
        if not bld.ok:
            raise RuntimeError(
                f"{_label} cmake build failed on remote host (exit={bld.exit_code}):\n"
                f"stdout: {bld.stdout}\nstderr: {bld.stderr}"
            )
    else:
        build_path.mkdir(parents=True, exist_ok=True)
        cmake_env = {**os.environ, "ROCM_PATH": rocm_path}
        r = subprocess.run(configure_cmd, capture_output=True, text=True, env=cmake_env)
        if r.returncode != 0 and _cmake_cache_mismatch(r.stdout + "\n" + r.stderr):
            shutil.rmtree(build_path, ignore_errors=True)
            build_path.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(configure_cmd, capture_output=True, text=True, env=cmake_env)
        assert r.returncode == 0, f"{_label} cmake configure failed:\n{r.stdout}\n{r.stderr}"
        r = subprocess.run(build_cmd, capture_output=True, text=True, env=cmake_env, timeout=7200.0)
        assert r.returncode == 0, f"{_label} cmake build failed:\n{r.stdout}\n{r.stderr}"

    return build_path


def _cmake_cache_mismatch(output: str) -> bool:
    """Return True when CMake says the existing cache must be regenerated."""
    lowered = output.lower()
    return "you have changed variables that require your cache to be deleted" in lowered or (
        "cmakecache.txt" in lowered and "different" in lowered
    )


# ---------------------------------------------------------------------------
# External repo helpers (formerly tests/common/_external_build.py)
# ---------------------------------------------------------------------------

_EXT_CLONE_ATTEMPTS = 3
_EXT_CLONE_BACKOFF_SECS = 10

_KNOWN_MPI_PREFIXES: tuple[tuple[str, str, str], ...] = (
    (
        "/usr/lib64/openmpi/bin/mpirun",
        "/usr/include/openmpi-x86_64",
        "/usr/lib64/openmpi/lib",
    ),
    (
        "/usr/lib64/mpi/gcc/openmpi2/bin/mpirun",
        "/usr/lib64/mpi/gcc/openmpi2/include",
        "/usr/lib64/mpi/gcc/openmpi2/lib64",
    ),
)


@dataclass(frozen=True)
class MpiRuntime:
    """MPI launcher and build/run environment discovered on an execution node."""

    launcher: str
    env: dict[str, str]


def _mpi_env_for_launcher(launcher: str) -> dict[str, str]:
    """Return include/library env hints for known MPI layouts."""
    for known_launcher, include_dir, lib_dir in _KNOWN_MPI_PREFIXES:
        if launcher == known_launcher:
            return {
                "MPI_HOME": str(pathlib.Path(known_launcher).parents[1]),
                "C_INCLUDE_PATH": include_dir,
                "CPLUS_INCLUDE_PATH": include_dir,
                "LD_LIBRARY_PATH": lib_dir,
            }
    launcher_path = pathlib.Path(launcher)
    if launcher_path.name == "mpirun" and launcher_path.parent.name == "bin":
        prefix = launcher_path.parents[1]
        mpi_include_dir = prefix / "include"
        lib_dirs = [prefix / "lib", prefix / "lib64"]
        return {
            "MPI_HOME": str(prefix),
            "C_INCLUDE_PATH": str(mpi_include_dir),
            "CPLUS_INCLUDE_PATH": str(mpi_include_dir),
            "LD_LIBRARY_PATH": ":".join(str(path) for path in lib_dirs),
        }
    return {}


def detect_mpi_runtime(remote_executor=None) -> MpiRuntime | None:
    """Discover an MPI runtime on the local host or remote execution node.

    Discovery is read-only: it searches ``PATH`` first and then known OpenMPI
    installation prefixes used by ROCm lab hosts. It does not install packages,
    mutate upstream Makefiles, or alter global environment.
    """
    if remote_executor is not None:
        result = remote_executor.run("command -v mpirun", timeout=15.0)
        if result.ok and result.stdout.strip():
            launcher = result.stdout.strip()
            return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))
        for launcher, _include_dir, _lib_dir in _KNOWN_MPI_PREFIXES:
            check = remote_executor.run(f"test -x {shlex.quote(launcher)}", timeout=15.0)
            if check.ok:
                return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))
        return None

    launcher = shutil.which("mpirun")
    if launcher:
        return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))
    for launcher, _include_dir, _lib_dir in _KNOWN_MPI_PREFIXES:
        if os.access(launcher, os.X_OK):
            return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))
    return None


def provision_openmpi_runtime(
    compiler_build_dir: str | os.PathLike,
    *,
    version: str = "4.1.4",
    expected_sha256: str = "",
    remote_executor=None,
    timeout: float = 3600.0,
) -> MpiRuntime:
    """Build and install a private OpenMPI toolchain under the framework build dir.

    This is for tests that require MPI but should not depend on system package
    state. It is idempotent and remote-transparent: if ``bin/mpirun`` already
    exists under the install prefix, the build is skipped.

    **External dependency — source download**:

    - Package:  OpenMPI (https://www.open-mpi.org)
    - Version:  ``version`` parameter (default ``"4.1.4"``)
    - Source:   ``https://download.open-mpi.org/release/open-mpi/v{major}.{minor}/openmpi-{version}.tar.gz``
    - License:  BSD 3-Clause ("New BSD") — https://github.com/open-mpi/ompi/blob/main/LICENSE
    - Integrity: pass ``expected_sha256`` (hex digest of the tarball) to enable
      SHA-256 verification immediately after download; omit or pass ``""`` to skip.
      Obtain the digest with::

          curl -fsSL <tarball-url> | sha256sum

    Args:
        compiler_build_dir: Framework build workspace.
        version: OpenMPI source release version.
        expected_sha256: Hex SHA-256 digest of the downloaded tarball.  When non-empty,
            verification runs immediately after the download and raises ``RuntimeError``
            on mismatch, preventing a corrupted or tampered archive from being built.
            Pass ``""`` (default) to skip the check.
        remote_executor: Optional ``SshExecutor`` for remote builds.
        timeout: Build timeout in seconds.

    Returns:
        ``MpiRuntime`` pointing at the private OpenMPI install.

    Raises:
        RuntimeError: If the download, SHA-256 verification, or build fails.
    """
    root = pathlib.Path(compiler_build_dir) / "mpi" / f"openmpi-{version}"
    src_dir = root / "src"
    install_dir = root / "install"
    tarball_path = root / f"openmpi-{version}.tar.gz"
    major_minor = ".".join(version.split(".")[:2])
    tarball_url = f"https://download.open-mpi.org/release/open-mpi/v{major_minor}/" f"openmpi-{version}.tar.gz"

    if remote_executor is not None:
        abs_root = _remote_abspath(root, remote_executor)
        abs_src = _remote_abspath(src_dir, remote_executor)
        abs_install = _remote_abspath(install_dir, remote_executor)
        abs_tarball = _remote_abspath(tarball_path, remote_executor)
        launcher = f"{abs_install}/bin/mpirun"
        if remote_executor.run(f"test -x {launcher}", timeout=15.0).ok:
            return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))
        sha256_check = f" && echo '{expected_sha256}  {abs_tarball}' | sha256sum -c -" if expected_sha256 else ""
        build_cmd = (
            f"rm -rf {abs_src} {abs_root}/openmpi-{version}"
            f" && mkdir -p {abs_root} {abs_install}"
            f" && curl -fsSL {tarball_url} -o {abs_tarball}"
            + sha256_check
            + f" && tar -xzf {abs_tarball} -C {abs_root}"
            f" && mv {abs_root}/openmpi-{version} {abs_src}"
            f" && cd {abs_src}"
            f" && ./configure --prefix={abs_install} --with-hwloc=internal --disable-mpi-fortran"
            f" && make -j$(nproc) && make install"
        )
        result = remote_executor.run(f'bash -c "{build_cmd}"', timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"OpenMPI source build failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))

    root = root.resolve()
    src_dir = src_dir.resolve()
    install_dir = install_dir.resolve()
    tarball_path = tarball_path.resolve()
    launcher = str(install_dir / "bin" / "mpirun")
    if os.access(launcher, os.X_OK):
        return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))
    root.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(src_dir, ignore_errors=True)
    shutil.rmtree(root / f"openmpi-{version}", ignore_errors=True)
    sha256_check = (
        f" && echo {shlex.quote(expected_sha256 + '  ' + str(tarball_path))} | sha256sum -c -"
        if expected_sha256
        else ""
    )
    build_cmd = (
        f"curl -fsSL {shlex.quote(tarball_url)} -o {shlex.quote(str(tarball_path))}"
        + sha256_check
        + f" && tar -xzf {shlex.quote(str(tarball_path))} -C {shlex.quote(str(root))}"
        f" && mv {shlex.quote(str(root / f'openmpi-{version}'))} {shlex.quote(str(src_dir))}"
        f" && cd {shlex.quote(str(src_dir))}"
        f" && ./configure --prefix={shlex.quote(str(install_dir))} --with-hwloc=internal --disable-mpi-fortran"
        f" && make -j{os.cpu_count() or 4} && make install"
    )
    proc = subprocess.run(
        build_cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"OpenMPI source build failed locally (exit={proc.returncode}):\n"
            f"stdout: {proc.stdout[-4000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    if not os.access(launcher, os.X_OK):
        raise RuntimeError(f"OpenMPI source build completed but mpirun was not found at {launcher}")
    return MpiRuntime(launcher=launcher, env=_mpi_env_for_launcher(launcher))


def clone_repo(  # noqa: C901
    url: str,
    dest: str | os.PathLike,
    *,
    ref: str | None = None,
    timeout: float = 1800.0,
    sparse_subtree: str | None = None,
    remote_executor=None,
) -> pathlib.Path:
    """Idempotently clone *url* into *dest*, optionally as a sparse subtree.

    Reuses an existing checkout when ``.git`` already exists in *dest*.
    Handles both local (subprocess) and remote (SSH) execution transparently.

    When *sparse_subtree* is set, performs a blob-filtered sparse clone (``--filter=blob:none
    --sparse --no-checkout``) so only the specified subdirectory is fetched — avoiding
    a full download of large monorepos.

    Args:
        url:             Git URL to clone.
        dest:            Destination directory for the working tree.
        ref:             Branch, tag, or commit to check out (``None`` = repo default).
        timeout:         Maximum seconds for the clone.
        sparse_subtree:  When set, performs sparse-checkout of this subdirectory only
                         (e.g. ``"projects/rccl-tests"``).
        remote_executor: ``SshExecutor`` to run on a remote host; ``None`` for local.

    Returns:
        ``pathlib.Path`` to the checkout directory.

    Raises:
        RuntimeError:   If the clone fails after retries.
        AssertionError: If sparse-checkout does not produce the expected subtree.
    """
    repo_dir = pathlib.Path(dest)

    if remote_executor is not None:
        if repo_dir.is_absolute() and str(repo_dir).startswith(remote_executor.remote_workspace_root()):
            abs_dest = str(repo_dir)
        elif repo_dir.is_absolute() and hasattr(remote_executor, "workspace_path_for"):
            abs_dest = remote_executor.workspace_path_for(str(repo_dir), category="external")
        else:
            abs_dest = _remote_abspath(repo_dir, remote_executor, category="external")
        abs_dest_path = pathlib.Path(abs_dest)

        check = remote_executor.run(f"test -d {abs_dest}/.git", timeout=30.0)
        if not check.ok:
            remote_executor.run(f"mkdir -p {shlex.quote(str(abs_dest_path.parent))}", timeout=30.0)
            if sparse_subtree:
                clone_cmd = f"git clone --filter=blob:none --sparse --no-checkout {url} {abs_dest}"
            else:
                clone_cmd = f"git clone {url} {abs_dest}"
            last_err = ""
            for attempt in range(1, _EXT_CLONE_ATTEMPTS + 1):
                result = remote_executor.run(clone_cmd, timeout=timeout)
                if result.ok:
                    break
                last_err = f"exit={result.exit_code}\nstdout: {result.stdout[:2000]}\nstderr: {result.stderr[:1000]}"
                remote_executor.run(f"rm -rf {abs_dest}", timeout=60.0)
                logger.warning(
                    "clone_repo (remote) attempt %d/%d failed; retrying in %ds:\n%s",
                    attempt,
                    _EXT_CLONE_ATTEMPTS,
                    _EXT_CLONE_BACKOFF_SECS,
                    last_err,
                )
                time.sleep(_EXT_CLONE_BACKOFF_SECS)
            else:
                raise RuntimeError(
                    f"Remote git clone failed after {_EXT_CLONE_ATTEMPTS} attempts.\n" f"cmd: {clone_cmd}\n{last_err}"
                )
            if sparse_subtree:
                sc = remote_executor.run(f"git -C {abs_dest} sparse-checkout set {sparse_subtree}", timeout=timeout)
                if not sc.ok:
                    raise RuntimeError(
                        f"git sparse-checkout set failed on remote (exit={sc.exit_code}): {sc.stderr[:1000]}"
                    )
        else:
            logger.info("clone_repo (remote): %s already exists — skipping clone", abs_dest)

        if ref:
            co = remote_executor.run(f"git -C {abs_dest} checkout {ref}", timeout=timeout)
            if not co.ok:
                raise RuntimeError(f"git checkout {ref!r} failed on remote (exit={co.exit_code}): {co.stderr[:1000]}")
        if sparse_subtree:
            sub_path = f"{abs_dest}/{sparse_subtree}"
            check_sub = remote_executor.run(f"test -d {sub_path}", timeout=15.0)
            assert check_sub.ok, (
                f"sparse-checkout did not produce {sub_path} on remote — "
                f"check that '{sparse_subtree}' exists on ref '{ref}'"
            )
        return pathlib.Path(abs_dest) / sparse_subtree if sparse_subtree else pathlib.Path(abs_dest)

    # Local path
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if not (repo_dir / ".git").is_dir():
        if sparse_subtree:
            clone_parts = ["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", url, str(repo_dir)]
        else:
            clone_parts = ["git", "clone", url, str(repo_dir)]
        last_err = ""
        for attempt in range(1, _EXT_CLONE_ATTEMPTS + 1):
            proc = subprocess.run(clone_parts, capture_output=True, text=True)
            if proc.returncode == 0:
                break
            last_err = f"exit={proc.returncode}\nstdout: {proc.stdout[:2000]}\nstderr: {proc.stderr[:1000]}"
            shutil.rmtree(repo_dir, ignore_errors=True)
            logger.warning(
                "clone_repo (local) attempt %d/%d failed; retrying in %ds:\n%s",
                attempt,
                _EXT_CLONE_ATTEMPTS,
                _EXT_CLONE_BACKOFF_SECS,
                last_err,
            )
            time.sleep(_EXT_CLONE_BACKOFF_SECS)
        else:
            raise RuntimeError(
                f"Local git clone failed after {_EXT_CLONE_ATTEMPTS} attempts.\n"
                f"cmd: {' '.join(clone_parts)}\n{last_err}"
            )
        if sparse_subtree:
            proc = subprocess.run(
                ["git", "-C", str(repo_dir), "sparse-checkout", "set", sparse_subtree],
                capture_output=True,
                text=True,
            )
            assert (
                proc.returncode == 0
            ), f"git sparse-checkout set failed (exit={proc.returncode}):\n{proc.stderr[-2000:]}"
    else:
        logger.info("clone_repo (local): %s already exists — skipping clone", repo_dir)

    if ref:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", ref],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"git checkout {ref!r} failed (exit={proc.returncode}):\n{proc.stderr[-2000:]}"
    if sparse_subtree:
        assert (repo_dir / sparse_subtree).is_dir(), (
            f"sparse-checkout did not produce {repo_dir / sparse_subtree} — "
            f"check that '{sparse_subtree}' exists on ref '{ref}'"
        )
    return repo_dir / sparse_subtree if sparse_subtree else repo_dir


def assert_license_present(repo_dir: str | os.PathLike, remote_executor=None) -> str:
    """Check that the checkout carries an upstream LICENSE file; return its name.

    Remote-transparent: when *remote_executor* is set, probes the remote filesystem
    via SSH rather than the local filesystem.

    A missing license file emits a WARNING (not an error) so that a missing or
    unusually-named license does not abort the entire test session — the operator
    should verify provenance out-of-band before merging.

    Args:
        repo_dir:        Path to the cloned working tree.
        remote_executor: ``SshExecutor`` for remote checks; ``None`` for local.

    Returns:
        Basename of the discovered license file, or ``""`` if none was found.
    """
    if remote_executor is not None:
        result = remote_executor.run(
            f"test -f {repo_dir}/LICENSE.txt || test -f {repo_dir}/LICENSE || "
            f"test -f {repo_dir}/LICENSE.md || test -f {repo_dir}/COPYING",
            timeout=15.0,
        )
        if not result.ok:
            logger.warning(
                "external checkout missing a LICENSE file at %s on remote — "
                "verify provenance of this repository before merging",
                repo_dir,
            )
            return ""
        logger.info("external suite license present (remote): %s", repo_dir)
        return "LICENSE"
    matches = glob.glob(os.path.join(str(repo_dir), "LICENSE*")) + glob.glob(os.path.join(str(repo_dir), "*LICENSE*"))
    if not matches:
        logger.warning(
            "external checkout missing a LICENSE file at %s — verify provenance before merging",
            repo_dir,
        )
        return ""
    name = os.path.basename(matches[0])
    logger.info("external suite license: %s", name)
    return name


def make_build(
    repo_dir: str | os.PathLike,
    *,
    make_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    parallel: bool = True,
    timeout: float = 7200.0,
    remote_executor=None,
) -> None:
    """Run ``make`` in *repo_dir* with *make_args* and an optional env overlay.

    Remote-transparent: when *remote_executor* is set, resolves the job count via
    ``nproc`` on the remote host and runs make via SSH.

    Args:
        repo_dir:        Path to the cloned working tree containing the Makefile.
        make_args:       Extra ``make`` arguments (e.g. ``["MPI=0", "ROCM_PATH=..."]``).
        env:             Environment overlay merged onto ``os.environ`` (local only).
        parallel:        Pass ``-j<ncpu>`` when ``True``.
        timeout:         Maximum seconds for the build.
        remote_executor: ``SshExecutor`` to run on a remote host; ``None`` for local.
    """
    if remote_executor is not None:
        jobs = resolve_parallel_jobs(remote_executor=remote_executor)
        args_str = " ".join(make_args or [])
        env_exports = ""
        if env:
            env_exports = " ".join(f"export {key}={shlex.quote(str(value))};" for key, value in env.items()) + " "
        cmd = f'bash -c "{env_exports}cd {repo_dir} && make -j{jobs} {args_str}"'
        result = remote_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"external-build make failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return

    local_cmd = ["make"]
    if parallel:
        local_cmd.append(f"-j{os.cpu_count() or 4}")
    if make_args:
        local_cmd.extend(make_args)
    build_env = {**os.environ, **env} if env else None
    proc = subprocess.run(
        local_cmd, cwd=str(repo_dir), capture_output=True, text=True, timeout=timeout, env=build_env, check=False
    )
    assert proc.returncode == 0, (
        f"external-build make failed (exit={proc.returncode}):\n"
        f"$ {' '.join(local_cmd)} (cwd={repo_dir})\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# Generic source-tarball download + autotools (configure/make/install) build.
# ---------------------------------------------------------------------------
# These mirror the CMake ``cmake_build`` and monorepo ``clone_repo`` primitives
# for the many third-party suites (UCX, libfabric, ...) that ship an autotools
# ``configure`` (or a project ``contrib/configure-*`` wrapper) plus a release
# source tarball rather than a git-cloneable CMake project.  Both are
# remote-transparent (local subprocess or ``SshExecutor``) and idempotent.


def _run_stream_step(
    cmd: str,
    *,
    remote_executor,
    timeout: float,
    label: str,
    log_path: str | None = None,
) -> None:
    """Run one long build step, streaming output live and (locally) to *log_path*.

    A buffered ``subprocess.run(capture_output=True)`` shows nothing until the
    process exits, which is unusable for multi-minute configure/make steps. Remote
    steps stream through the SSH executor's ``stream=True`` path; local steps use
    the framework's shared streaming Popen runner and also append to *log_path*.

    Args:
        cmd:             Shell command to run.
        remote_executor: ``SshExecutor`` for remote builds, or ``None`` for local.
        timeout:         Wall-clock timeout in seconds.
        label:           Human-readable step name used in the error message.
        log_path:        Local-only persisted log file (``tail -f`` while it runs).

    Raises:
        RuntimeError: If the step exits non-zero.
    """
    logger.info("external-build %s -> streaming%s", label, f" to {log_path}" if log_path else "")
    if remote_executor is not None:
        result = remote_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"external-build {label} failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return
    # Local import avoids a module-load cycle (executors import from builder).
    from framework.executors.background_process import _blocking_stream_run

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
        detail = f" Full log: {log_path}" if log_path else ""
        raise RuntimeError(
            f"external-build {label} failed locally (exit={result.exit_code}).{detail}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


def _abspath_for(path: str, remote_executor) -> str:
    """Return an absolute path for *path*, transparently for local/remote nodes."""
    if remote_executor is not None:
        return _remote_abspath(pathlib.Path(path), remote_executor)
    return str(pathlib.Path(path).resolve())


def configure_make_build(
    source_dir: str,
    build_dir: str,
    *,
    configure_script: str = "./configure",
    configure_args: list[str] | None = None,
    bootstrap_script: str | None = None,
    env_prefix: str = "",
    make_install: bool = True,
    make_args: list[str] | None = None,
    jobs: int | None = None,
    sentinel: str | None = None,
    log_dir: str | None = None,
    remote_executor=None,
    use_lock: bool = True,
    timeout: float = 7200.0,
) -> str:
    """Autotools [bootstrap→]configure→make→[install] out-of-tree; return *build_dir*.

    Generic autotools counterpart to :func:`cmake_build`. Env is injected via an
    ``env_prefix`` (never ``os.environ``); idempotent when *sentinel* exists.

    Args:
        source_dir:       Source tree (holds ``configure`` / bootstrap script).
        build_dir:        Out-of-tree build dir (created if absent).
        configure_script: Configure entry point, relative to *build_dir*.
        configure_args:   Flags for the configure script.
        bootstrap_script: Optional pre-configure step (e.g. ``./autogen.sh``) run in
                          *source_dir*; needed for git checkouts.
        env_prefix:       ``VAR=val ...`` exported for every step.
        make_install:     Run ``make install`` after ``make``.
        make_args:        Extra ``make`` arguments.
        jobs:             Parallel jobs; auto-detected when None.
        sentinel:         Artifact path (relative to *build_dir*) for the skip check.
        log_dir:          Per-step local log dir.
        remote_executor:  ``SshExecutor`` for remote builds, or None for local.
        use_lock:         Acquire the internal build lock; set False under an outer
                          :func:`external_build_lock` to avoid self-deadlock.
        timeout:          Per-step wall-clock timeout in seconds.

    Returns:
        The *build_dir* path (absolute on the remote node when applicable).
    """
    jobs = resolve_parallel_jobs(jobs, remote_executor=remote_executor)
    cfg_args = " ".join(configure_args or [])
    mk_args = " ".join(make_args or [])
    env_kw = f"env {env_prefix} " if env_prefix else ""

    # configure_script is resolved relative to the build dir (e.g. "../configure"),
    # so only the build dir needs an absolute path here.
    abs_build = _abspath_for(build_dir, remote_executor)

    def _sentinel_present() -> bool:
        return sentinel is not None and _path_is_file(f"{abs_build}/{sentinel}", remote_executor)

    # Fast path: an existing sentinel means a previous run already built this tree.
    if _sentinel_present():
        logger.info("external-build: sentinel %s present — skipping configure/make", sentinel)
        return abs_build

    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    def _log(step: str) -> str | None:
        return os.path.join(log_dir, f"build-{step}.log") if (log_dir and remote_executor is None) else None

    configure_cmd = (
        f"mkdir -p {shlex.quote(abs_build)} && cd {shlex.quote(abs_build)} && {env_kw}{configure_script} {cfg_args}"
    )
    build_cmd = f"cd {shlex.quote(abs_build)} && {env_kw}make -j{jobs} {mk_args}"
    steps = [("configure", configure_cmd), ("compile", build_cmd)]
    if make_install:
        steps.append(("install", f"cd {shlex.quote(abs_build)} && {env_kw}make -j{jobs} install"))
    if bootstrap_script:
        # Bootstrap (e.g. ./autogen.sh) runs in the source tree; resolve its absolute
        # path only when needed (avoids an extra SSH round-trip on remote builds).
        abs_source = _abspath_for(source_dir, remote_executor)
        steps.insert(0, ("bootstrap", f"cd {shlex.quote(abs_source)} && {env_kw}{bootstrap_script}"))

    def _run_steps() -> None:
        if _sentinel_present():  # re-check under the lock
            logger.info("external-build: sentinel %s present — skipping configure/make", sentinel)
            return
        for step, cmd in steps:
            wrapped = f'bash -c "{cmd}"' if remote_executor is not None else cmd
            _run_stream_step(
                wrapped,
                remote_executor=remote_executor,
                timeout=timeout,
                label=f"{step} ({os.path.basename(source_dir)})",
                log_path=_log(step),
            )

    # Serialize concurrent xdist workers sharing one build tree (a local file lock
    # works even for remote builds, which dispatch via SSH from this host).
    if use_lock:
        with external_build_lock(abs_build, timeout=timeout):
            _run_steps()
    else:
        _run_steps()
    return abs_build


@contextlib.contextmanager
def external_build_lock(build_dir: str, *, timeout: float = 7200.0):
    """Exclusive cross-process ``fcntl`` lock keyed on *build_dir*.

    Args:
        build_dir: Build directory used only as the lock key.
        timeout:   Max seconds to wait for the lock.

    Yields:
        None, while the lock is held.

    Raises:
        TimeoutError: If the lock is not acquired within *timeout*.
    """
    import tempfile

    digest = hashlib.sha256(build_dir.encode("utf-8")).hexdigest()[:16]
    lock_path = os.path.join(tempfile.gettempdir(), f"rocm-test-extbuild-{digest}.lock")
    with open(lock_path, "w", encoding="utf-8") as lock_fh:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"external-build: could not acquire build lock for {build_dir!r} "
                        f"within {timeout}s — another worker may be hung."
                    ) from exc
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _path_is_file(path: str, remote_executor) -> bool:
    """Return True when *path* is a file, transparently for local/remote nodes."""
    if remote_executor is not None:
        return bool(remote_executor.run(f"test -f {shlex.quote(path)}", timeout=30.0).ok)
    return os.path.isfile(path)


# ---------------------------------------------------------------------------
# Build fingerprint helpers (formerly private to tests/e2e/hpc/conftest.py)
# ---------------------------------------------------------------------------
# A two-key SHA-256 fingerprint is written to ``<build_dir>/.cmake_fingerprint``
# after a successful build and compared on the next run.
#
# Two keys are stored:
#   "structural" — digest of inputs whose change requires a clean wipe+rebuild
#                  (branch, arch, ROCm path, compiler)
#   "full"       — digest of ALL cmake inputs (structural + extra cmake flags)
#
# Three cache actions:
#   "skip"        — full digest matches; reuse the existing build unchanged
#   "incremental" — structural matches but full differs; reconfigure + rebuild in place
#   "wipe"        — structural changed; delete the build dir and rebuild from scratch
# ---------------------------------------------------------------------------


def compute_fingerprint(key_inputs: list[str]) -> str:
    """Return a 16-char SHA-256 hex digest of *key_inputs* (sorted, JSON-encoded).

    Args:
        key_inputs: Strings that uniquely identify the build configuration
                    (cmake ``-D`` flags, ROCm path, GPU arch, version tags, …).

    Returns:
        A 16-character lowercase hex string.
    """
    content = json.dumps(sorted(str(x) for x in key_inputs))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def source_tree_fingerprint(paths: list[str | os.PathLike]) -> str:
    """Return a digest for local source trees used by synced CMake builds."""
    digest = hashlib.sha256()
    for root in sorted(pathlib.Path(p).resolve() for p in paths):
        if root.is_file():
            files = [root]
            base = root.parent
        else:
            files = sorted(path for path in root.rglob("*") if path.is_file())
            base = root
        for path in files:
            rel = path.relative_to(base).as_posix()
            digest.update(rel.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()[:16]


def _fingerprint_path(build_dir: pathlib.Path) -> pathlib.Path:
    return build_dir / ".cmake_fingerprint"


def read_build_fingerprint(build_dir: pathlib.Path, remote_executor=None) -> dict[str, str] | None:
    """Read the stored two-key fingerprint from *build_dir*.

    Args:
        build_dir:       Path to the cmake build directory.
        remote_executor: ``SshExecutor`` for remote reads, or ``None`` for local.

    Returns:
        ``{"structural": "<hex>", "full": "<hex>"}`` dict, or ``None`` when absent or invalid.
    """
    if remote_executor is not None:
        fp_path = pathlib.Path(_remote_abspath(build_dir, remote_executor)) / ".cmake_fingerprint"
        result = remote_executor.run(f"cat {shlex.quote(str(fp_path))}", timeout=10.0)
        raw = result.stdout.strip() if result.ok else ""
    else:
        fp_path = _fingerprint_path(build_dir)
        raw = fp_path.read_text().strip() if fp_path.exists() else ""
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "full" not in data or "structural" not in data:
        return None
    return data


def write_build_fingerprint(
    build_dir: pathlib.Path,
    full: str,
    remote_executor=None,
    *,
    structural: str | None = None,
) -> None:
    """Write the two-key fingerprint to ``<build_dir>/.cmake_fingerprint``.

    Args:
        build_dir:       Path to the cmake build directory.
        full:            Digest over all build inputs.
        remote_executor: ``SshExecutor`` for remote writes, or ``None`` for local.
        structural:      Digest over structural inputs only; defaults to *full*.
    """
    content = json.dumps({"structural": structural if structural is not None else full, "full": full})
    if remote_executor is not None:
        fp_path = pathlib.Path(_remote_abspath(build_dir, remote_executor)) / ".cmake_fingerprint"
        remote_executor.run(f"printf %s {shlex.quote(content)} > {shlex.quote(str(fp_path))}", timeout=10.0)
    else:
        fp_path = _fingerprint_path(build_dir)
        fp_path.write_text(content)


def build_cache_action(
    build_dir: pathlib.Path,
    structural_fp: str,
    full_fp: str,
    remote_executor=None,
) -> str:
    """Decide how to (re)build given the stored vs current fingerprints.

    Args:
        build_dir:       Path to the cmake build directory.
        structural_fp:   Current structural digest (branch/arch/ROCm path/compiler).
        full_fp:         Current full digest (all cmake inputs).
        remote_executor: ``SshExecutor`` for remote reads, or ``None`` for local.

    Returns:
        ``"skip"``, ``"incremental"``, or ``"wipe"``.
    """
    stored = read_build_fingerprint(build_dir, remote_executor)
    if stored is None or stored.get("structural") != structural_fp:
        return "wipe"
    if stored.get("full") != full_fp:
        return "incremental"
    return "skip"


def wipe_build_dir(build_dir: pathlib.Path, remote_executor=None) -> None:
    """Delete *build_dir* entirely so cmake starts from a clean slate.

    Args:
        build_dir:       Path to wipe.
        remote_executor: ``SshExecutor`` for remote deletion, or ``None`` for local.
    """
    if remote_executor is not None:
        remote_executor.run(f"rm -rf {shlex.quote(_remote_abspath(build_dir, remote_executor))}", timeout=120.0)
    else:
        shutil.rmtree(build_dir, ignore_errors=True)


def build_artifact_exists(build_dir: pathlib.Path, artifact: str, remote_executor=None) -> bool:
    """Return ``True`` if *artifact* (relative to *build_dir*) exists.

    Args:
        build_dir:       Path to the cmake build directory.
        artifact:        Relative path of the binary within *build_dir*.
        remote_executor: ``SshExecutor`` for remote checks, or ``None`` for local.

    Returns:
        Whether the artifact file exists.
    """
    if remote_executor is not None:
        artifact_path = pathlib.Path(_remote_abspath(build_dir, remote_executor)) / artifact
        return bool(remote_executor.run(f"test -f {shlex.quote(str(artifact_path))}", timeout=15.0).ok)
    artifact_path = build_dir / artifact
    return artifact_path.exists()


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def require_gpu_arch(gpu_arch: str | None, label: str) -> None:
    """Raise ``pytest.fail`` when ``--gpu-arch`` was not supplied.

    A missing GPU architecture is a CI **misconfiguration**, not an optional
    resource — call this at the top of any build fixture whose CMakeLists.txt
    or compiler requires an explicit ``--offload-arch`` target.  Produces an
    ``ERROR`` outcome in the pytest report, making the misconfiguration visible
    rather than silently skipping the test.

    Args:
        gpu_arch: Value of the ``gpu_arch`` fixture (``None`` when ``--gpu-arch``
                  was not passed).
        label:    Short name of the test area used in the failure message.

    Raises:
        pytest.fail.Exception: When *gpu_arch* is ``None``.
    """
    if gpu_arch is None:
        pytest.fail(
            f"--gpu-arch is required for {label} HIP build but was not passed. "
            "Pass --gpu-arch <arch> (e.g. --gpu-arch gfx942) to the pytest invocation."
        )


def assert_binary_exists(path: str, remote_executor=None, label: str = "") -> str:
    """Assert that a compiled binary exists at *path* (local or remote).

    Remote-transparent: when *remote_executor* is set, probes via SSH ``test -f``;
    otherwise checks the local filesystem with ``os.path.isfile``.

    Args:
        path:            Absolute path to the binary to verify.
        remote_executor: ``SshExecutor`` for remote checks; ``None`` for local.
        label:           Short name used in the assertion message.

    Returns:
        *path* unchanged (for inline use: ``return assert_binary_exists(binary, ...)``)

    Raises:
        AssertionError: When the binary is not found.
    """
    _name = label or os.path.basename(path)
    if remote_executor is not None:
        result = remote_executor.run(f"test -f {shlex.quote(path)}", timeout=15.0)
        assert result.ok, f"{_name}: binary not found at {path} on remote node after build"
    else:
        assert os.path.isfile(path), f"{_name}: binary not found at {path} after build"
    return path


def _remote_abspath(rel_path: pathlib.Path, remote_executor, *, category: str = "work") -> str:
    """Resolve *rel_path* to an absolute path on the remote node.

    Relative framework paths are resolved under the managed remote output
    workspace. Absolute paths already under that workspace are preserved. Other
    absolute paths are treated as local/coordinator paths and mapped into the
    workspace to avoid assuming they exist on the remote.

    Args:
        rel_path:        Relative (or already absolute) path.
        remote_executor: ``SshExecutor`` instance (must not be ``None``).

    Returns:
        Absolute path string as it would appear on the remote filesystem.
    """
    path_str = str(rel_path)
    if hasattr(remote_executor, "remote_workspace_root") and path_str.startswith(
        remote_executor.remote_workspace_root()
    ):
        return path_str
    if hasattr(remote_executor, "workspace_path_for"):
        return str(remote_executor.workspace_path_for(path_str, category=category))
    if rel_path.is_absolute():
        return path_str
    result = remote_executor.run("echo $HOME", timeout=10.0)
    remote_home = result.stdout.strip()
    return f"{remote_home}/{REMOTE_WORKSPACE_DIR}/{category_root(category)}/{rel_path}"
