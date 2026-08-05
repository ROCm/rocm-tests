# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
abstract_executor.py -- Abstract base class for all command executors.

All executors share the same ``run(command, timeout)`` signature so that
test code, fixtures, and plugins never need to know which executor is active.
The executor in use is determined by configuration and CLI flags.

To add a new executor (e.g. SSH, container), subclass AbstractExecutor and
implement ``run()``. Register it in the executor factory if needed.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from framework.common.helpers import ExecutionResult

if TYPE_CHECKING:
    from framework.executors.background_process import AbstractBackgroundProcess


class AbstractExecutor(abc.ABC):
    """Contract for executing shell commands in a controlled environment.

    Subclasses:
        LocalExecutor       -- local subprocess with HIP_VISIBLE_DEVICES set
        DryRunExecutor      -- synthetic stub for GPU-less CI runs
    """

    @abc.abstractmethod
    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Execute *command* and return its result.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds to wait (None = no limit).

        Returns:
            ExecutionResult with exit_code, stdout, stderr, duration.

        Raises:
            TimeoutError:  If the command exceeds *timeout*.
            RuntimeError:  If the executor cannot start the command.
        """

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
        console_label: str | None = None,
        stream: bool = False,
    ) -> AbstractBackgroundProcess:
        """Start *command* in the background; return a handle immediately.

        The process runs concurrently while the test continues.  stdout and
        stderr are forwarded to the console and *log_path* (when given) in
        real time by a dedicated daemon ``threading.Thread``.

        Call ``handle.stop()`` (or use the handle as a context manager) to
        terminate the process and collect its full output as an
        ``ExecutionResult``.

        Thread safety:
            Each call creates a fully isolated subprocess and reader thread.
            Multiple concurrent calls to ``start_background()`` and ``run()``
            on the same executor instance are safe — no mutable state is shared
            between instances or between background and foreground calls.

        Log isolation:
            Pass a distinct *log_path* per process to keep captured output
            attributable.  Console output may interleave visually (expected
            for parallel subprocesses); ``ExecutionResult.stdout`` /
            ``.stderr`` returned by ``handle.stop()`` are always per-process.

        Supported by:
            ``LocalExecutor``, ``CpuExecutor``, ``DryRunExecutor``, and
            ``SshExecutor`` (via a detached ``SshBackgroundProcess``).  The SSH
            backend launches the command fully detached (``setsid``) with output
            redirected to node-side capture files; a daemon thread tails those
            files and forwards new output to the console and *log_path* live
            (a short poll interval behind the node), and ``stop()`` fetches the
            final captured stdout/stderr.  All of this uses brief, serialised
            control channels, so many concurrent background roles never exhaust
            the remote ``MaxSessions`` limit.

        For SSH, live streaming is **opt-in** (via *stream* and/or *log_path*): by
        default the detached process is only captured at ``stop()``.  The local
        executors always stream; they accept *console_label*/*stream* for API
        parity and ignore them (they already forward output live).

        Not yet supported by:
            ``ContainerExecutor`` — raises ``NotImplementedError``.

        Args:
            command:  Shell command to launch in the background.
            timeout:  Default stop-grace-period in seconds, forwarded to
                      ``BackgroundProcess.stop()`` (default 30 s if ``None``).
            log_path: If given, all subprocess output (stdout+stderr) is
                      appended to this file in real time.  Use a distinct path
                      per concurrent background process for isolated artifacts.
            console_label: Human-readable label for live output attribution (SSH:
                      ``[bg <console_label>]`` console prefix).  Ignored by local
                      executors.
            stream:   SSH only — emit live output to the ``rocm.test`` logger.
                      Ignored by local executors (which always stream).

        Returns:
            ``BackgroundProcess`` handle with ``.pid``, ``.is_alive``,
            ``.poll()``, ``.stop()``, and context-manager support.

        Raises:
            NotImplementedError: When the concrete executor does not support
                                 background execution.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support start_background(). "
            "Use LocalExecutor or CpuExecutor for local background processes."
        )
