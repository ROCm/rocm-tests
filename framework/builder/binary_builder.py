# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
binary_builder.py -- Process-safe, incremental C++ binary compilation for test suites.

Design principles
-----------------
CPU/GPU environment separation
    Compilation is a CPU-only operation.  ``BinaryBuilder`` strips all GPU
    device-selection environment variables (``HIP_VISIBLE_DEVICES``,
    ``ROCR_VISIBLE_DEVICES``, etc.) from the subprocess environment so that
    the compiler is never inadvertently bound to a specific GPU ordinal by
    whatever the pytest GPU allocation machinery has set.

    GPU execution (running the compiled binary) is the sole responsibility of
    the caller — typically via ``target_executor`` or ``gpu_fixture``.
    ``ROCR_VISIBLE_DEVICES`` is injected automatically by the executor fixture.

pytest-xdist parallel safety
    When pytest distributes tests across multiple workers (``-n 4``), each
    worker has its own session and may try to compile the same binary
    simultaneously.  ``BinaryBuilder`` serialises compilation via an
    exclusive ``fcntl`` file lock on ``<output>.lock``.  The first worker
    that acquires the lock compiles; the others block (with a timeout), then
    see the binary is already up-to-date and return immediately.

Incremental builds
    If the binary already exists and its mtime is newer than the source file,
    compilation is skipped entirely — useful when the same test session is
    re-run after a partial failure.

Live streaming and dual-log
    The hipcc subprocess streams stdout+stderr to the pytest console in real
    time **and** writes the same bytes to ``<output>.build.log`` simultaneously.
    On failure, the ``AssertionError`` message includes the log path so CI
    runners can archive and inspect it without grepping test output.

Timeout handling
    Two independent timeout thresholds protect against hung builds:

    * ``timeout`` (wall-clock): hard upper bound on total compilation time.
      Default 7200 s (2 h), configurable via ``ROCM_TEST_THEROCK_BUILD_TIMEOUT_SECS``.
    * ``inactivity_timeout``: if the compiler emits no output for this many
      seconds the build is considered stalled (typically a linker OOM).
      Default 600 s (10 min).

    On either timeout: SIGTERM is sent first, then after a grace period
    SIGKILL is issued if the process has not exited.

Error reporting
    Compilation failure raises ``AssertionError``, which pytest records as
    ``ERROR`` on every test that depends on the fixture.  This correctly
    signals a broken test environment rather than a test logic failure.

Usage (from a session-scoped conftest fixture)::

    from framework.builder.binary_builder import BinaryBuilder

    # Option A — explicit hipcc path
    @pytest.fixture(scope="session")
    def my_binary(rock_dir, compiler_build_dir, include_dirs):
        return BinaryBuilder().compile(
            hipcc=os.path.join(rock_dir, "bin", "hipcc"),
            src="tests/e2e/myarea/src/my_kernel.cpp",
            output=os.path.join(compiler_build_dir, "my_kernel"),
            include_dirs=[include_dirs],
        )

    # Option B — pass rocm_dir and let BinaryBuilder derive hipcc
    @pytest.fixture(scope="session")
    def my_binary(rock_dir, compiler_build_dir, include_dirs):
        return BinaryBuilder().compile(
            rocm_dir=rock_dir,
            src="tests/e2e/myarea/src/my_kernel.cpp",
            output=os.path.join(compiler_build_dir, "my_kernel"),
            include_dirs=[include_dirs],
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
    ) -> str:
        """Compile *src* to *output* using hipcc (CPU-only operation).

        Exactly one of *hipcc* or *rocm_dir* must be supplied:
            - *hipcc*: Absolute path to the hipcc binary.
            - *rocm_dir*: Path to the ROCm/TheRock install tree; hipcc is
              derived as ``{rocm_dir}/bin/hipcc``.

        Args:
            src:                  Path to the C++ source file (relative to repo
                                  root or absolute).
            output:               Destination path for the compiled binary.
            hipcc:                Absolute path to the hipcc compiler binary.
                                  Mutually exclusive with *rocm_dir*.
            rocm_dir:             Path to a ROCm/TheRock installation that
                                  contains ``bin/hipcc``.  Mutually exclusive
                                  with *hipcc*.
            std:                  C++ standard passed as ``-std=<std>``
                                  (default ``"c++17"``).
            opt:                  Optimisation flag (default ``"-O2"``).
            arch:                 GPU architecture target passed as
                                  ``--offload-arch=<arch>`` (e.g. ``"gfx942"``).
                                  When ``None``, no arch flag is added and hipcc
                                  auto-detects from the installed ROCm device list.
            include_dirs:         Additional include paths; each becomes a ``-I``
                                  flag.
            extra_flags:          Any extra compiler flags appended verbatim
                                  before ``-o``.
            timeout:              Wall-clock seconds before the compiler is killed
                                  (default 7200 s / 2 h).  ``None`` disables the
                                  wall-clock limit (inactivity limit still applies).
            inactivity_timeout:   Seconds of silence (no compiler output) before
                                  the process is considered stalled and killed
                                  (default 600 s / 10 min).
            log_path:             File path for the build log.  If given, the
                                  merged stdout+stderr stream is written here
                                  simultaneously with the console.  Created (or
                                  overwritten) on each compile run; persists after
                                  the session so CI runners can archive it.

        Returns:
            Path to the compiled binary (same as *output*).

        Raises:
            ValueError:     Neither or both of *hipcc* / *rocm_dir* supplied.
            TimeoutError:   Compilation exceeded *timeout* or *inactivity_timeout*.
            AssertionError: Compilation exited non-zero.  Propagates through
                            pytest fixture machinery as test ``ERROR`` status.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        lock_path = output + ".lock"

        with open(lock_path, "w", encoding="utf-8") as lock_fh:
            # Acquire exclusive lock with timeout — blocks other xdist workers
            # until this worker either compiles the binary or confirms it is
            # current.  A non-blocking poll loop avoids hanging indefinitely if
            # the lock holder is killed by the OS while still running.
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
