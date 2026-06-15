# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
cpu_executor.py -- Real subprocess executor with no GPU environment injection.

Use for hw.cpu_only tests that need real shell commands. GPU env vars (ROCR_*)
are stripped. Streaming modes: stdout/stderr captured by default; set
stream_stdout=True for live output (useful for long-running build steps).
"""

from __future__ import annotations

import logging
import os

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import (
    BackgroundProcess,
    _blocking_stream_run,
    _make_background_process,
)
from framework.logging.test_logger import TestLogger

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT: float = 300.0


class CpuExecutor(AbstractExecutor):
    """Execute shell commands on the local host without modifying the GPU environment.

    Runs a real subprocess — unlike ``DryRunExecutor`` which returns synthetic
    results — and does not set ``HIP_VISIBLE_DEVICES`` — unlike ``LocalExecutor``
    which is bound to a specific GPU ordinal.

    Intended for tests marked ``@pytest.mark.hw.cpu_only``: ROCm tool version
    checks, compiler invocations, config validation, and health probes that do
    not need a GPU allocation.

    Attributes:
        working_dir:    Optional directory from which every command is launched.
        env_overrides:  Key-value pairs merged on top of the inherited environment
                        before each subprocess invocation.
        stream_stdout:  When True, subprocess STDOUT is written to ``sys.stdout``
                        in real time.
        stream_stderr:  When True (default), STDERR is written to ``sys.stderr``
                        in real time.  Set to False when ``test_logger`` handles
                        output routing.
        log_path:       If set, all subprocess output (STDOUT+STDERR) is appended
                        to this file.
    """

    def __init__(
        self,
        working_dir: str | None = None,
        env_overrides: dict | None = None,
        stream_stdout: bool = False,
        stream_stderr: bool = True,
        log_path: str | None = None,
        suppress_output_log: bool = False,
        test_logger: TestLogger | None = None,
    ) -> None:
        """Initialise a CPU-only subprocess executor.

        Args:
            working_dir:          Working directory for subprocess calls (default: inherited
                                  from the pytest process).
            env_overrides:        Extra environment variables applied to every ``run()`` call.
            stream_stdout:        When True, subprocess STDOUT is written to ``sys.stdout``
                                  in real time.
            stream_stderr:        When True (default), subprocess STDERR is written to
                                  ``sys.stderr`` in real time.  Pass False when
                                  ``test_logger`` handles output routing.
            log_path:             If given, all subprocess output (STDOUT+STDERR) is
                                  appended to this per-test file.
            suppress_output_log:  When True, skip emitting command output through the
                                  ``rocm.output`` logger.  Use for background pollers
                                  (e.g. ``--monitor-gpu``) whose output goes to a
                                  dedicated log file and must not flood the console.
            test_logger:          When provided, block-header logging protocol is applied:
                                  one header per command, verbatim output, and session.log
                                  event lines.
        """
        self.working_dir = working_dir
        self.env_overrides: dict = env_overrides or {}
        self.stream_stdout = stream_stdout
        self.stream_stderr = stream_stderr
        self.log_path = log_path
        self.suppress_output_log = suppress_output_log
        self.test_logger = test_logger

    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Execute *command* in a subprocess without modifying GPU-related variables.

        The full ``os.environ`` is inherited, then ``env_overrides`` are merged on top.
        Output is emitted through the ``rocm.output`` logger (console with timestamp)
        and written to ``log_path`` when set.  When ``test_logger`` is attached,
        block-header logging is used instead.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds before ``TimeoutError`` is raised (default: 300 s).

        Returns:
            ExecutionResult with exit_code, stdout, stderr, and wall-clock duration.

        Raises:
            TimeoutError: If the command exceeds *timeout*.
        """
        effective_timeout = timeout if timeout is not None else _DEFAULT_TIMEOUT
        proc_env = os.environ.copy()
        if self.env_overrides:
            proc_env.update(self.env_overrides)
        logger.debug("CpuExecutor running: %s", command)

        # When test_logger is provided, disable inner stderr streaming so that
        # TestLogger handles console/file output routing after the command returns.
        stream_stderr = self.stream_stderr if self.test_logger is None else False

        def _inner_run(cmd: str, t: float | None) -> ExecutionResult:
            return _blocking_stream_run(
                command=cmd,
                env=proc_env,
                cwd=self.working_dir,
                timeout=effective_timeout if t is None else t,
                stream_stdout=self.stream_stdout,
                stream_stderr=stream_stderr,
                log_path=self.log_path,
            )

        if self.test_logger is not None:
            start = self.test_logger.cmd_start(command)
            raw = _inner_run(command, timeout)
            self.test_logger.cmd_end(raw.stdout, raw.stderr, raw.exit_code, start)
            return raw

        raw = _inner_run(command, timeout)

        # Emit output lines via rocm.output logger (console + caplog → Allure).
        # Skipped when suppress_output_log=True (e.g. background GPU monitor pollers).
        if not self.suppress_output_log:
            _output_logger = logging.getLogger("rocm.output")
            combined = (raw.stdout or "") + "\n" + (raw.stderr or "")
            for line in combined.splitlines():
                if line.strip():
                    _output_logger.info("%s", line)

        return raw

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
    ) -> BackgroundProcess:
        """Start *command* in the background without modifying GPU-related variables.

        Applies ``env_overrides`` and ``working_dir`` (same as ``run()``).  No
        GPU environment variables are set or cleared.  stdout and stderr are
        forwarded to the live console and *log_path* (when given) in real time
        by a background reader thread.

        Thread safety:
            Each call creates a fully isolated ``subprocess.Popen``, reader
            thread, and chunk lists.  Concurrent ``run()`` and
            ``start_background()`` calls on the same executor instance are
            safe — they operate on independent pipe file-descriptors.

        Args:
            command:  Shell command to launch in the background.
            timeout:  Ignored here; passed through to ``BackgroundProcess``
                      as the default stop grace period.
            log_path: If given, all subprocess output is appended to this file
                      in real time.  Use a distinct path per concurrent process
                      to keep logs attributable.

        Returns:
            ``BackgroundProcess`` handle.
        """
        proc_env = os.environ.copy()
        if self.env_overrides:
            proc_env.update(self.env_overrides)
        logger.debug("CpuExecutor starting background: %s", command)
        return _make_background_process(
            command=command,
            env=proc_env,
            cwd=self.working_dir,
            log_path=log_path,
            stream_stdout=self.stream_stdout,
            stream_stderr=self.stream_stderr,
        )
