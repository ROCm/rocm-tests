# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
background_process.py -- Background subprocess execution with live I/O streaming.

Provides the ``BackgroundProcess`` handle returned by every executor's
``start_background()`` method, and the ``NoOpBackgroundProcess`` stub used by
``DryRunExecutor``.

Design
------
Each ``BackgroundProcess`` owns exactly one OS subprocess and one daemon
``threading.Thread`` (the *reader thread*).  The reader thread runs a
``select``-based loop that:

    1. Forwards raw bytes from ``proc.stdout`` and ``proc.stderr`` to the live
       console (``sys.stdout.buffer`` / ``sys.stderr.buffer``) in real time so
       that long-running daemons produce visible output without buffering.
    2. Writes the same bytes to ``log_path`` (when given) so output survives
       the test session as a separate, attributable artifact.
    3. Accumulates all bytes into per-instance lists (``_stdout_chunks`` /
       ``_stderr_chunks``) so that ``stop()`` can assemble a complete
       ``ExecutionResult`` after the process exits.

Thread safety
-------------
There is **no shared mutable state** between two ``BackgroundProcess``
instances — every instance has its own subprocess, thread, event, and chunk
lists.  The ``stop()`` method follows a strict *signal → kill → join* protocol
before reading the accumulated chunks, so no explicit lock is needed:

    1. ``_stop_evt.set()``     — signals the reader thread to finish draining.
    2. ``_bg_graceful_kill()`` — terminates the OS process.
    3. ``thread.join()``       — **blocks until the reader thread exits**.
    4. Read ``_stdout_chunks`` / ``_stderr_chunks`` and build ``ExecutionResult``.

The ``join()`` at step 3 provides the memory visibility barrier that makes the
read at step 4 safe without a lock.

Concurrent ``run()`` and ``start_background()`` calls on the same executor
instance are also safe: ``run()`` creates its own ``Popen`` with its own local
pipe file-descriptors — it never shares state with the reader thread.

Log isolation
-------------
Each ``start_background()`` caller passes its own ``log_path``; multiple
concurrent background processes therefore write to distinct files.  Console
output may visually interleave (expected for any parallel subprocess setup);
``ExecutionResult.stdout`` / ``.stderr`` returned by ``stop()`` are always
per-process.

Usage (via executor.start_background() — not instantiated directly)::

    with cpu_executor.start_background(
        "rocm-smi --showmetrics --interval=2",
        log_path="output/artifacts/executor-logs/test_foo__monitor.log",
    ) as monitor:
        result = local_executor.run("./my_kernel")
        assert result.ok
        assert monitor.is_alive

    stopped = monitor.stop_result   # ExecutionResult with daemon output
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import select
import signal
import subprocess
import sys
import threading
import time

from framework.common.helpers import ExecutionResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers shared by the reader thread
# ---------------------------------------------------------------------------


def _bg_pipe_write(chunk: bytes, console_target, log_fh) -> None:
    """Write *chunk* to an optional console target and an optional log file.

    Args:
        chunk:          Raw bytes from a subprocess pipe.
        console_target: A writable binary stream (e.g. ``sys.stdout.buffer``),
                        or ``None`` to skip console output.
        log_fh:         An open binary file handle, or ``None`` to skip logging.
    """
    if console_target is not None:
        console_target.write(chunk)
        console_target.flush()
    if log_fh is not None:
        log_fh.write(chunk)
        log_fh.flush()


def _bg_graceful_kill(proc: subprocess.Popen, grace_secs: float = 30.0) -> None:
    """Send ``SIGTERM`` to *proc*, then ``SIGKILL`` if it survives *grace_secs*.

    Calls ``proc.kill()`` (``SIGKILL``) as a last resort.  Always calls
    ``proc.wait()`` to reap the process so no zombie is left.

    Args:
        proc:       ``subprocess.Popen`` instance to terminate.
        grace_secs: Seconds to wait after ``SIGTERM`` before escalating to
                    ``SIGKILL`` (default 30 s).
    """
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=grace_secs)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except ProcessLookupError:
        # Process already exited — nothing to kill.
        pass


def _reader_loop(
    proc: subprocess.Popen,
    stdout_chunks: list,
    stderr_chunks: list,
    stop_evt: threading.Event,
    log_fh,
    stdout_console,
    stderr_console,
) -> None:
    """Forward subprocess output to console and log in real time.

    Runs inside a daemon thread started by ``start_background()``.  Exits when
    *stop_evt* is set **and** the process has exited, then performs a final
    drain to capture any bytes still in the OS pipe buffers.

    Args:
        proc:           ``subprocess.Popen`` instance whose pipes to read.
        stdout_chunks:  List accumulating raw ``bytes`` from ``proc.stdout``.
        stderr_chunks:  List accumulating raw ``bytes`` from ``proc.stderr``.
        stop_evt:       ``threading.Event``; when set, the loop exits after
                        the process terminates and all data is drained.
        log_fh:         Open binary file handle for disk logging, or ``None``.
        stdout_console: Binary stream target for STDOUT console output, or
                        ``None`` to suppress.
        stderr_console: Binary stream target for STDERR console output, or
                        ``None`` to suppress.
    """
    while True:
        # Check for exit condition: stop requested AND process has terminated.
        if stop_evt.is_set() and proc.poll() is not None:
            break

        ready = select.select([s for s in [proc.stdout, proc.stderr] if s is not None], [], [], 0.5)[0]

        if proc.stdout in ready:
            chunk = proc.stdout.read()
            if chunk:
                _bg_pipe_write(chunk, stdout_console, log_fh)
                stdout_chunks.append(chunk)

        if proc.stderr in ready:
            chunk = proc.stderr.read()
            if chunk:
                _bg_pipe_write(chunk, stderr_console, log_fh)
                stderr_chunks.append(chunk)

    # Final drain: process has exited; read any bytes still in the OS buffers.
    for pipe, chunks, console in [
        (proc.stdout, stdout_chunks, stdout_console),
        (proc.stderr, stderr_chunks, stderr_console),
    ]:
        try:
            tail = pipe.read()  # type: ignore[union-attr]
            if tail:
                _bg_pipe_write(tail, console, log_fh)
                chunks.append(tail)
        except (OSError, ValueError):
            pass  # Pipe already closed — nothing to drain.


# ---------------------------------------------------------------------------
# BackgroundProcess — public handle
# ---------------------------------------------------------------------------


class BackgroundProcess:
    """Handle for a subprocess started in the background by an executor.

    Obtained via ``executor.start_background(command)`` — not constructed
    directly in test code.

    Every instance owns:
        - An OS subprocess (``subprocess.Popen``).
        - A daemon reader thread that forwards stdout/stderr to the console
          and an optional log file in real time.
        - Per-instance byte-accumulation lists for ``stop()`` to assemble into
          an ``ExecutionResult``.

    Thread safety:
        Multiple ``BackgroundProcess`` instances share no mutable state.
        ``stop()`` uses a *signal → kill → join* protocol so the chunk lists
        are safe to read without a lock once ``join()`` returns.

    Attributes:
        stop_result: ``ExecutionResult`` populated by ``stop()`` after the
                     process terminates.  ``None`` until ``stop()`` is called.

    Example (context manager)::

        with cpu_executor.start_background(
            "rocm-smi --showmetrics --interval=2",
            log_path="output/executor-logs/test__monitor.log",
        ) as monitor:
            result = local_executor.run("./my_kernel")
            assert result.ok
        # stop() called automatically; captured output in monitor.stop_result

    Example (explicit stop)::

        bg = cpu_executor.start_background("python3 -m http.server 9000")
        try:
            result = cpu_executor.run("curl http://localhost:9000/")
            assert result.ok
        finally:
            stopped = bg.stop(timeout=10.0)
            assert stopped.exit_code in (0, -15)
    """

    def __init__(
        self,
        proc: subprocess.Popen,
        reader_thread: threading.Thread,
        stop_evt: threading.Event,
        t0: float,
    ) -> None:
        """Initialise the handle.  Called by ``start_background()`` only.

        Args:
            proc:          Running ``subprocess.Popen`` instance.
            reader_thread: Daemon thread executing ``_reader_loop``.
            stop_evt:      Event used to signal the reader thread to stop.
            t0:            ``time.monotonic()`` timestamp from process start.
        """
        self._proc = proc
        self._thread = reader_thread
        self._stop_evt = stop_evt
        self._t0 = t0
        self.stop_result: ExecutionResult | None = None

        # These lists are written by the reader thread and read by stop()
        # after join() — no lock required (join provides the barrier).
        self._stdout_chunks: list = reader_thread._stdout_chunks  # type: ignore[attr-defined]
        self._stderr_chunks: list = reader_thread._stderr_chunks  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pid(self) -> int:
        """OS process ID of the background subprocess.

        Returns:
            Integer PID.
        """
        return self._proc.pid

    @property
    def is_alive(self) -> bool:
        """True while the background subprocess is still running.

        Returns:
            ``True`` if the process has not yet exited.
        """
        return self._proc.poll() is None

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def poll(self) -> int | None:
        """Non-blocking exit-code check.

        Returns:
            ``None`` if the process is still running, or the integer exit code
            if it has terminated.
        """
        return self._proc.poll()

    def stop(self, timeout: float = 30.0) -> ExecutionResult:
        """Terminate the background process and return its captured output.

        Sequence:
            1. Signal *_stop_evt* — reader thread will exit after the process
               terminates.
            2. Send ``SIGTERM``; if the process survives *timeout* seconds,
               escalate to ``SIGKILL``.
            3. Join the reader thread (up to *timeout* + 5 s) to drain the
               final bytes from the OS pipe buffers.
            4. Assemble and return an ``ExecutionResult`` from the accumulated
               chunks.

        Idempotent: calling ``stop()`` more than once returns the same cached
        ``ExecutionResult`` without re-terminating the process.

        Args:
            timeout: Grace period in seconds between ``SIGTERM`` and ``SIGKILL``
                     (default 30 s).  Also used as the reader-thread join timeout
                     (+ 5 s guard).

        Returns:
            ``ExecutionResult`` with ``exit_code``, ``stdout``, ``stderr``,
            and ``duration`` (wall-clock seconds from process start to stop).
        """
        if self.stop_result is not None:
            return self.stop_result

        # 1. Signal the reader thread to exit after the process terminates.
        self._stop_evt.set()

        # 2. Gracefully kill the process.
        _bg_graceful_kill(self._proc, grace_secs=timeout)

        # 3. Join the reader thread to ensure all bytes are drained.
        self._thread.join(timeout=timeout + 5.0)

        # 4. Collect exit code and build result.
        rc = self._proc.wait()
        duration = time.monotonic() - self._t0
        stdout = b"".join(self._stdout_chunks).decode(errors="replace").rstrip()
        stderr = b"".join(self._stderr_chunks).decode(errors="replace").rstrip()

        self.stop_result = ExecutionResult(
            exit_code=rc,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
        )
        return self.stop_result

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> BackgroundProcess:
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# NoOpBackgroundProcess — DryRun stub
# ---------------------------------------------------------------------------


class NoOpBackgroundProcess:
    """Stub ``BackgroundProcess`` returned by ``DryRunExecutor.start_background()``.

    Implements the same public interface as ``BackgroundProcess`` but never
    spawns a real subprocess.  All properties return inert values and ``stop()``
    returns a synthetic ``ExecutionResult`` consistent with ``DryRunExecutor``.

    Example::

        with dry_run_executor.start_background("would-be-a-daemon") as bg:
            result = dry_run_executor.run("echo RESULT_OK")
            assert result.ok
            assert not bg.is_alive   # NoOpBackgroundProcess is never alive
    """

    def __init__(self) -> None:
        self.stop_result: ExecutionResult | None = None

    @property
    def pid(self) -> int:
        """Synthetic PID — always ``-1`` for a no-op process.

        Returns:
            ``-1``
        """
        return -1

    @property
    def is_alive(self) -> bool:
        """Always ``False`` — a no-op process never runs.

        Returns:
            ``False``
        """
        return False

    def poll(self) -> int | None:
        """Always returns ``0`` (synthetic success, not running).

        Returns:
            ``0``
        """
        return 0

    def stop(self, _timeout: float = 30.0) -> ExecutionResult:
        """Return a synthetic ``ExecutionResult`` without terminating anything.

        Args:
            _timeout: Ignored.

        Returns:
            Synthetic ``ExecutionResult(exit_code=0, stdout="DRY_RUN=1\\n", ...)``.
        """
        if self.stop_result is None:
            self.stop_result = ExecutionResult(
                exit_code=0,
                stdout="DRY_RUN=1\nRESULT_OK",
                stderr="",
                duration=0.0,
            )
        return self.stop_result

    def __enter__(self) -> NoOpBackgroundProcess:
        return self

    def __exit__(self, *_) -> None:
        self.stop()


# ---------------------------------------------------------------------------
# Shared blocking stream runner — used by LocalExecutor and CpuExecutor.
# Not part of the public API.
# ---------------------------------------------------------------------------


def _blocking_stream_run(
    command: str,
    env: dict,
    cwd: str | None,
    timeout: float,
    stream_stdout: bool,
    stream_stderr: bool,
    log_path: str | None,
) -> ExecutionResult:
    """Execute *command* via Popen with streamed output and an optional log file.

    Shared implementation used by both ``LocalExecutor.run()`` and
    ``CpuExecutor.run()`` — the only difference between the two callers is
    the *env* dict and whether *cwd* is set.

    STDERR is written to ``sys.stderr`` when *stream_stderr* is True (default).
    STDOUT is written to ``sys.stdout`` only when *stream_stdout* is True.
    Both channels are appended to *log_path* when set.

    Args:
        command:       Shell command string to execute.
        env:           Full environment dict for the subprocess.
        cwd:           Working directory, or ``None`` to inherit.
        timeout:       Wall-clock seconds before forced termination.
        stream_stdout: When True, stream STDOUT to ``sys.stdout`` in real time.
        stream_stderr: When False, suppress live STDERR console output.
        log_path:      If given, all output is appended to this file.

    Returns:
        ExecutionResult with exit_code, stdout, stderr, and wall-clock duration.

    Raises:
        TimeoutError: If the command exceeds *timeout* seconds.
    """
    process = subprocess.Popen(
        command,
        shell=True,  # nosec B602 — shell=True required for pipeline/redirect support
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    os.set_blocking(process.stdout.fileno(), False)  # type: ignore[union-attr]
    os.set_blocking(process.stderr.fileno(), False)  # type: ignore[union-attr]

    stdout_buf, stderr_buf = b"", b""
    t0 = time.monotonic()
    stdout_console = sys.stdout.buffer if stream_stdout else None
    stderr_console = sys.stderr.buffer if stream_stderr else None

    with contextlib.ExitStack() as stack:
        log_fh = stack.enter_context(open(log_path, "ab")) if log_path else None

        while True:
            elapsed = time.monotonic() - t0
            if elapsed >= timeout:
                _bg_graceful_kill(process)
                raise TimeoutError(f"Command timed out after {timeout}s: {command}")

            ready = select.select(
                [s for s in [process.stdout, process.stderr] if s is not None],
                [],
                [],
                min(max(timeout - elapsed, 0.1), 5.0),
            )[0]

            if process.stdout in ready:
                chunk = process.stdout.read()
                if chunk:
                    _bg_pipe_write(chunk, stdout_console, log_fh)
                    stdout_buf += chunk

            if process.stderr in ready:
                chunk = process.stderr.read()
                if chunk:
                    _bg_pipe_write(chunk, stderr_console, log_fh)
                    stderr_buf += chunk

            if process.poll() is not None:
                for pipe, is_stdout in [
                    (process.stdout, True),
                    (process.stderr, False),
                ]:
                    tail = pipe.read()  # type: ignore[union-attr]
                    if tail:
                        target = stdout_console if is_stdout else stderr_console
                        _bg_pipe_write(tail, target, log_fh)
                        if is_stdout:
                            stdout_buf += tail
                        else:
                            stderr_buf += tail
                break

        if log_fh is not None:
            log_fh.flush()

    rc = process.wait()
    return ExecutionResult(
        exit_code=rc,
        stdout=stdout_buf.decode(errors="replace").rstrip(),
        stderr=stderr_buf.decode(errors="replace").rstrip(),
        duration=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# Internal factory — called by executor.start_background() implementations.
# Not part of the public API.
# ---------------------------------------------------------------------------


def _make_background_process(
    command: str,
    env: dict,
    cwd: str | None = None,
    log_path: str | None = None,
    stream_stdout: bool = False,
    stream_stderr: bool = True,
) -> BackgroundProcess:
    """Start *command* as a background subprocess and return a ``BackgroundProcess`` handle.

    This is the shared implementation used by ``LocalExecutor.start_background()``
    and ``CpuExecutor.start_background()``.  It is **not** part of the public
    API — callers should use the executor methods.

    Args:
        command:       Shell command string to execute.
        env:           Full environment dict for the subprocess.
        cwd:           Working directory, or ``None`` to inherit.
        log_path:      If given, all output (stdout+stderr) is appended to this
                       file in real time.  Use a distinct path per concurrent
                       background process for isolated log artifacts.
        stream_stdout: When True, STDOUT is forwarded to ``sys.stdout.buffer``
                       in real time.
        stream_stderr: When True (default), STDERR is forwarded to
                       ``sys.stderr.buffer`` in real time.

    Returns:
        ``BackgroundProcess`` handle with ``.pid``, ``.is_alive``, ``.poll()``,
        ``.stop()``, and context-manager support.
    """
    proc = subprocess.Popen(
        command,
        shell=True,  # nosec B602 — shell=True required for pipeline/redirect support
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    os.set_blocking(proc.stdout.fileno(), False)  # type: ignore[union-attr]
    os.set_blocking(proc.stderr.fileno(), False)  # type: ignore[union-attr]

    stop_evt = threading.Event()
    stdout_chunks: list = []
    stderr_chunks: list = []

    stdout_console = sys.stdout.buffer if stream_stdout else None
    stderr_console = sys.stderr.buffer if stream_stderr else None

    # Open the log file here (in the main thread) so that errors are surfaced
    # immediately rather than silently swallowed inside the reader thread.
    log_fh = None
    if log_path:
        pathlib.Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "ab")  # binary append; closed in thread

    def _thread_target() -> None:
        try:
            _reader_loop(
                proc=proc,
                stdout_chunks=stdout_chunks,
                stderr_chunks=stderr_chunks,
                stop_evt=stop_evt,
                log_fh=log_fh,
                stdout_console=stdout_console,
                stderr_console=stderr_console,
            )
        finally:
            if log_fh is not None:
                log_fh.close()

    thread = threading.Thread(target=_thread_target, daemon=True, name=f"bg-reader-{proc.pid}")
    # Attach the chunk lists so BackgroundProcess.__init__ can reference them.
    thread._stdout_chunks = stdout_chunks  # type: ignore[attr-defined]
    thread._stderr_chunks = stderr_chunks  # type: ignore[attr-defined]
    thread.start()

    return BackgroundProcess(proc=proc, reader_thread=thread, stop_evt=stop_evt, t0=time.monotonic())
