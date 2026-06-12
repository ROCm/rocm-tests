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
from datetime import datetime
import errno
import fcntl
import logging
import os
import pathlib
import re
import select
import shlex
import signal
import subprocess
import sys
import time

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
        src_abs = os.path.abspath(src)
        output_abs = os.path.abspath(output)
        out_dir = os.path.dirname(output_abs)
        abs_include_dirs = [os.path.abspath(d) for d in (include_dirs or [])]

        remote_executor.upload_tree(os.path.dirname(src_abs))
        for inc in abs_include_dirs:
            # Absolute paths so the remote sees them at the same location.
            remote_executor.upload_tree(inc)

        mk = remote_executor.run(f"mkdir -p {shlex.quote(out_dir)}")
        if not mk.ok:
            raise RuntimeError(f"BinaryBuilder remote mkdir failed (exit={mk.exit_code}):\n{mk.stderr}")

        cmd = self._build_cmd(hipcc, rocm_dir, src_abs, output_abs, std, opt, arch, abs_include_dirs, extra_flags)
        result = remote_executor.run(shlex.join(cmd), timeout=timeout)
        if not result.ok:
            raise AssertionError(
                f"Compilation of '{src}' failed on remote host (exit={result.exit_code}).\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        logger.info("BinaryBuilder : remote compiled → %s", output_abs)
        return output_abs

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

                if select.select([process.stdout], [], [], select_wait)[0]:
                    chunk = process.stdout.read()  # type: ignore[union-attr]
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
