# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
cpu_executor.py -- Real subprocess executor with no GPU environment injection.

Use for hw.cpu_only tests that need real shell commands. GPU env vars (ROCR_*)
are stripped. Streaming modes: stdout/stderr captured by default; set
stream_stdout=True for live output (useful for long-running build steps).
"""

from __future__ import annotations

from datetime import datetime
import logging
import os
import pathlib

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import (
    BackgroundProcess,
    _blocking_stream_run,
    _make_background_process,
)

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
                        in real time.  Set to False when ``LogConfig`` handles
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
        session_log_path: str | None = None,
        suppress_output_log: bool = False,
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
                                  ``LogConfig`` handles output routing.
            log_path:             If given, all subprocess output (STDOUT+STDERR) is
                                  appended to this per-test file.
            session_log_path:     If given, all subprocess output is also appended to
                                  this session-wide aggregate log file.
            suppress_output_log:  When True, skip emitting command output through the
                                  ``rocm.output`` logger.  Use for background pollers
                                  (e.g. ``--monitor-gpu``) whose output goes to a
                                  dedicated log file and must not flood the console.
        """
        self.working_dir = working_dir
        self.env_overrides: dict = env_overrides or {}
        self.stream_stdout = stream_stdout
        self.stream_stderr = stream_stderr
        self.log_path = log_path
        self.session_log_path = session_log_path
        self.suppress_output_log = suppress_output_log

    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Execute *command* in a subprocess without modifying GPU-related variables.

        The full ``os.environ`` is inherited, then ``env_overrides`` are merged on top.
        Output is emitted through the ``rocm.output`` logger (console with timestamp)
        and written to ``log_path`` and ``session_log_path`` when set.

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
        raw = _blocking_stream_run(
            command=command,
            env=proc_env,
            cwd=self.working_dir,
            timeout=effective_timeout,
            stream_stdout=self.stream_stdout,
            stream_stderr=self.stream_stderr,
            log_path=self.log_path,
        )

        # Emit output lines via rocm.output logger (console + caplog → Allure).
        # Skipped when suppress_output_log=True (e.g. background GPU monitor pollers).
        if not self.suppress_output_log:
            _output_logger = logging.getLogger("rocm.output")
            combined = (raw.stdout or "") + "\n" + (raw.stderr or "")
            for line in combined.splitlines():
                if line.strip():
                    _output_logger.info("%s", line)

        # Write to session log with timestamp prefix when configured.
        if self.session_log_path:
            ts = datetime.now().strftime("%H:%M:%S.%f")[:12]
            all_lines = [
                command,
                *(raw.stdout or "").splitlines(),
                *(raw.stderr or "").splitlines(),
            ]
            prefixed = [f"{ts} [cpu_executor | localhost | CPU] : {ln}" for ln in all_lines if ln.strip()]
            if prefixed:
                try:
                    pathlib.Path(self.session_log_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(self.session_log_path, "a", encoding="utf-8") as fh:
                        fh.write("\n".join(prefixed) + "\n")
                except OSError as exc:
                    logger.warning("CpuExecutor: cannot write session log %s: %s", self.session_log_path, exc)

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
