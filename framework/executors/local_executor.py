# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
local_executor.py -- Local subprocess executor with GPU environment injection.

Runs commands on the local host using ``subprocess.Popen``.  The correct
``ROCR_VISIBLE_DEVICES`` value is injected automatically from the allocated
GPU index — test code must never set this variable directly.

``ROCR_VISIBLE_DEVICES`` is the framework-level standard for GPU isolation.
It operates at the ROCr/HSA layer, which all ROCm runtimes (HIP, HSA, OpenCL)
sit on top of — setting it once is sufficient for all ROCm workloads.

Three operating modes
---------------------
Explicit single (``gpu_index=N``):
    ``ROCR_VISIBLE_DEVICES`` is set to ``"N"`` on every ``run()`` call.
    This is the mode used for single-GPU ``hw.gpu`` tests.

Explicit multi (``gpu_index=[N, M, ...]``):
    ``ROCR_VISIBLE_DEVICES`` is set to ``"N,M,..."`` (comma-separated).
    This is the mode used for multi-GPU ``hw.multi_gpu`` tests on one node,
    parallel to ``SshExecutor(gpu_indices=[N, M, ...])``.

Ambient mode (``gpu_index=None``, the default):
    If ``ROCR_VISIBLE_DEVICES`` is already present in the process environment
    (e.g., set by a CI runner or an outer orchestrator), it is left untouched.
    If ``ROCR_VISIBLE_DEVICES`` is **not** set either, ``run()`` raises
    ``RuntimeError`` with a message directing the caller to use ``target_executor``.

Streaming modes
---------------
Default (``stream_stdout=False``):
    STDERR is written to ``sys.stderr`` in real time (visible in the console
    and captured by pytest for failure reports).
    STDOUT is buffered and returned in ``ExecutionResult.stdout`` only.
    Both STDOUT and STDERR are written to *log_path* (when set).

Verbose (``stream_stdout=True``, enabled by ``ROCM_TEST_FRAMEWORK_LOG_LEVEL=debug``):
    STDOUT is also written to ``sys.stdout`` in real time.
    Both channels still go to *log_path* (when set).

Usage (via NodeSlot.make_executor() — not directly in tests):
    executor = LocalExecutor(gpu_index=0)        # single explicit GPU
    executor = LocalExecutor(gpu_index=[0, 1])   # multi-GPU (ROCR_VISIBLE_DEVICES=0,1)
    executor = LocalExecutor()                   # ambient — reads ROCR_VISIBLE_DEVICES from env
    result = executor.run("python3 -c 'import torch; print(torch.cuda.is_available())'")
    assert result.ok
"""

from __future__ import annotations

import logging
import os
import re
import select
import shlex
import subprocess
import sys

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import (
    AbstractBackgroundProcess,
    _blocking_stream_run,
    _make_background_process,
)
from framework.logging.test_logger import TestLogger

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: float = 300.0  # 5 minutes

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


# ---------------------------------------------------------------------------
# LocalExecutor
# ---------------------------------------------------------------------------


class LocalExecutor(AbstractExecutor):
    """Execute commands locally with ROCR_VISIBLE_DEVICES set for the allocated GPU.

    Supports explicit single/multi-GPU and ambient modes — see module docstring.
    Streaming behavior controlled by stream_stdout and stream_stderr flags.
    """

    def __init__(
        self,
        gpu_index: int | list[int] | None = None,
        stream_stdout: bool = False,
        stream_stderr: bool = True,
        log_path: str | None = None,
        test_logger: TestLogger | None = None,
    ) -> None:
        """Initialize for a specific GPU ordinal, multiple ordinals, or ambient mode.

        Args:
            gpu_index:     GPU ordinal(s) assigned by ``GpuAllocator`` (0-based).
                           Pass an ``int`` for a single GPU, a ``list[int]`` for
                           multi-GPU (sets ``ROCR_VISIBLE_DEVICES=N,M,...``), or
                           ``None`` to operate in ambient mode and inherit
                           ``ROCR_VISIBLE_DEVICES`` from the process environment.
            stream_stdout: When True, subprocess STDOUT is written to
                           ``sys.stdout`` in real time.
            stream_stderr: When True (default), subprocess STDERR is written to
                           ``sys.stderr`` in real time.  Set to False when
                           *test_logger* is provided.
            log_path:      If given, all subprocess output (STDOUT+STDERR) is
                           appended to this file.
            test_logger:   When provided, the block-header logging protocol is
                           applied: one header per command, verbatim output, and
                           session.log event lines.
        """
        self.gpu_index = gpu_index
        self.stream_stdout = stream_stdout
        self.stream_stderr = stream_stderr
        self.log_path = log_path
        self.test_logger = test_logger

    def run(self, command: str, timeout: float | None = None, *, stream: bool = False) -> ExecutionResult:
        """Execute *command* in a subprocess with ROCR_VISIBLE_DEVICES configured.

        Env-priority logic:
            1. If ``ROCR_VISIBLE_DEVICES`` is already in the process environment
               it is left **untouched** — the caller (CI runner / allocator /
               outer test orchestrator) owns that assignment.
            2. If it is absent and ``gpu_index`` was supplied at construction,
               ``ROCR_VISIBLE_DEVICES`` is set to ``str(gpu_index)``.
            3. If it is absent and ``gpu_index`` is ``None``, a ``RuntimeError``
               is raised directing the caller to use the ``gpu_fixture`` or to
               set ``ROCR_VISIBLE_DEVICES`` explicitly.

        Args:
            command: Shell command string to execute.
            timeout: Seconds before a TimeoutError is raised (default 300 s).
            stream:  When True, stream stdout live for this command.

        Returns:
            ExecutionResult with exit_code, stdout, stderr, and wall-clock duration.

        Raises:
            RuntimeError: When no GPU is configured (ambient mode with no env var).
            TimeoutError: When the command exceeds *timeout*.
        """
        env = os.environ.copy()
        if "ROCR_VISIBLE_DEVICES" not in env:
            if self.gpu_index is None:
                raise RuntimeError(
                    "LocalExecutor: no gpu_index set and ROCR_VISIBLE_DEVICES is not in the "
                    "environment. Use target_executor (from remote_node_plugin) to allocate a GPU, or construct "
                    "LocalExecutor(gpu_index=N) explicitly."
                )
            if isinstance(self.gpu_index, list):
                env["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_index)
            else:
                env["ROCR_VISIBLE_DEVICES"] = str(self.gpu_index)

        effective_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT
        gpu_label = env.get("ROCR_VISIBLE_DEVICES", "?")
        logger.debug("LocalExecutor[ROCR_VISIBLE_DEVICES=%s] running: %s", gpu_label, command)

        # When test_logger is provided, disable inner stderr streaming so that
        # TestLogger handles console/file output routing after the command returns.
        stream_stderr = self.stream_stderr if self.test_logger is None else False

        def _inner_run(cmd: str, t: float | None) -> ExecutionResult:
            return _blocking_stream_run(
                command=cmd,
                env=env,
                cwd=None,
                timeout=effective_timeout if t is None else t,
                stream_stdout=stream or self.stream_stdout,
                stream_stderr=stream_stderr,
                log_path=self.log_path,
            )

        if self.test_logger is not None:
            start = self.test_logger.cmd_start(command)
            raw = _inner_run(command, timeout)
            self.test_logger.cmd_end(raw.stdout, raw.stderr, raw.exit_code, start)
            return raw
        return _inner_run(command, timeout)

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
        console_label: str | None = None,
        stream: bool = False,
    ) -> AbstractBackgroundProcess:
        """Start *command* in the background with ``ROCR_VISIBLE_DEVICES`` set.

        Applies the same GPU environment logic as ``run()``: explicit
        ``gpu_index`` takes priority; if ``gpu_index`` is ``None``, the
        existing ``ROCR_VISIBLE_DEVICES`` from the process environment is
        inherited; if neither is set, ``RuntimeError`` is raised.

        stdout and stderr are forwarded to the live console and *log_path*
        (when given) in real time by a background reader thread.  Call
        ``handle.stop()`` to terminate the process and collect its output.

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
                      in real time.  Use a distinct path per concurrent process.

        Returns:
            ``BackgroundProcess`` handle.

        Raises:
            RuntimeError: When no GPU is configured (ambient mode with no env var).
        """
        env = os.environ.copy()
        if "ROCR_VISIBLE_DEVICES" not in env:
            if self.gpu_index is None:
                raise RuntimeError(
                    "LocalExecutor: no gpu_index set and ROCR_VISIBLE_DEVICES is not in the "
                    "environment. Use target_executor (from remote_node_plugin) to allocate a GPU, or construct "
                    "LocalExecutor(gpu_index=N) explicitly."
                )
            if isinstance(self.gpu_index, list):
                env["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_index)
            else:
                env["ROCR_VISIBLE_DEVICES"] = str(self.gpu_index)

        gpu_label = env.get("ROCR_VISIBLE_DEVICES", "?")
        logger.debug(
            "LocalExecutor[ROCR_VISIBLE_DEVICES=%s] starting background: %s",
            gpu_label,
            command,
        )
        return _make_background_process(
            command=command,
            env=env,
            log_path=log_path,
            stream_stdout=self.stream_stdout,
            stream_stderr=self.stream_stderr,
        )


# ---------------------------------------------------------------------------
# Module-level functional API — for general-purpose subprocess execution.
#
# These complement LocalExecutor (which is GPU-bound and injects
# ROCR_VISIBLE_DEVICES).  Use these for non-GPU operations such as compiling
# test binaries or running any executable where GPU allocation is not needed.
# ---------------------------------------------------------------------------


def run_cmd_get_stdout_stderr(
    *cmd: str,
    cwd: str | None = None,
    env: dict | None = None,
    stdin: str | None = None,
    timeout: float = 1200.0,
    quiet: bool = False,
) -> tuple[int, str, str]:
    """Execute *cmd* and return ``(exit_code, stdout, stderr)`` with live output.

    Uses non-blocking ``select``-based I/O so that stdout and stderr are
    streamed to the console in real time — useful for long-running compilations
    or test binaries that produce output incrementally.

    Args:
        *cmd:    Command and its arguments as separate strings.
        cwd:     Working directory for the subprocess (default: inherited).
        env:     Extra environment variables merged on top of the current env.
        stdin:   Optional string written to the subprocess stdin.
        timeout: Seconds of inactivity before the process is killed (default 1200 s).
        quiet:   When True, suppress live console output (stdout/stderr still returned).

    Returns:
        Tuple of ``(exit_code, stdout_str, stderr_str)`` with ANSI escapes stripped.
    """
    env_str = " ".join(f'{k}="{v}"' for k, v in env.items()) if env else ""
    cwd_str = f"[{cwd}] " if cwd else ""
    logger.info("++Exec %s$ %s%s", cwd_str, env_str, shlex.join(cmd))

    cmd_env = os.environ.copy()
    if env:
        cmd_env.update({k: str(v) for k, v in env.items()})

    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=cmd_env,
        stdin=subprocess.PIPE if stdin else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
    )

    if stdin:
        with process.stdin:  # type: ignore[union-attr]
            data = stdin if isinstance(stdin, bytes) else stdin.encode()
            process.stdin.write(data)  # type: ignore[union-attr]

    os.set_blocking(process.stdout.fileno(), False)  # type: ignore[union-attr]
    os.set_blocking(process.stderr.fileno(), False)  # type: ignore[union-attr]

    def _read_stream(fd) -> bytes:
        chunk = fd.read()
        if chunk and not quiet:
            sys.stdout.write(chunk.decode(errors="replace"))
            sys.stdout.flush()
        return chunk or b""

    stdout_buf, stderr_buf = b"", b""
    chunk: bytes = b"x"  # non-empty sentinel to enter the loop
    while chunk != b"":
        ready = select.select([s for s in [process.stdout, process.stderr] if s is not None], [], [], timeout)[0]
        if not ready:
            msg = f"Reached timeout of {timeout}s — killing process."
            logger.warning(msg)
            stdout_buf += msg.encode()
            process.kill()
            break
        chunk = b""
        if process.stdout in ready:
            chunk = _read_stream(process.stdout)
            stdout_buf += chunk
        if process.stderr in ready:
            chunk = _read_stream(process.stderr)
            stderr_buf += chunk

    sys.stdout.write("\n")
    sys.stdout.flush()

    ret = process.wait()
    status = "success" if ret == 0 else "failed"
    logger.info("[%s] %s (exit=%d)", shlex.join(cmd), status, ret)
    return (
        ret,
        _ANSI_RE.sub("", stdout_buf.decode(errors="replace")),
        _ANSI_RE.sub("", stderr_buf.decode(errors="replace")),
    )


def run_cmd_get_output(*args, **kwargs) -> tuple[int, str]:
    """Execute a command; return ``(exit_code, combined stdout+stderr)``.

    All positional and keyword arguments are forwarded to
    :func:`run_cmd_get_stdout_stderr`.

    Returns:
        Tuple of ``(exit_code, output)`` where *output* is stdout and stderr
        concatenated.
    """
    ret, stdout, stderr = run_cmd_get_stdout_stderr(*args, **kwargs)
    return ret, stdout + stderr


def run_cmd(*args, **kwargs) -> int:
    """Execute a command; return the exit code.

    All positional and keyword arguments are forwarded to
    :func:`run_cmd_get_stdout_stderr`.

    Returns:
        Shell exit code (0 = success).
    """
    ret, _, _ = run_cmd_get_stdout_stderr(*args, **kwargs)
    return ret
