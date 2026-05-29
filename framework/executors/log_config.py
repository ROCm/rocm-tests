# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
log_config.py -- Per-test logging configuration and shared logging protocol.

``LogConfig`` is a dataclass that holds the context needed to produce structured,
prefixed log output for a single test invocation.  It is passed to
``LocalExecutor`` and ``SshExecutor`` so that logging is a first-class
executor capability rather than a formerly separate wrapper class.

The ``run_with_logging()`` helper implements the 7-step logging protocol:

    1.  Capture ``t_cmd`` timestamp (before the command runs).
    2.  Emit ``++Exec [node] $ command`` via the framework logger.
    3.  Write the command line to per-test and session log files with ``t_cmd``.
    4.  Delegate to the wrapped ``run()`` callable.
    5.  Capture ``t_out`` timestamp (after the command returns).
    6.  Emit stderr lines via the ``rocm.output`` logger (not streamed live).
    7.  Write output lines to log files; emit invocation summary.

``result.stdout`` / ``result.stderr`` are returned unmodified — assertions
work without prefix stripping.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
import pathlib

from framework.common.helpers import ExecutionResult

logger = logging.getLogger(__name__)

# Dedicated logger for subprocess output lines so they flow through pytest's
# ``caplog`` → Allure attachment pipeline with their own log name.
_output_logger = logging.getLogger("rocm.output")

# Fixed column widths for the bracketed context prefix.
_TEST_WIDTH = 20
_NODE_WIDTH = 12
_GPU_WIDTH = 10


@dataclass
class LogConfig:
    """Context needed to produce prefixed, timestamped log output for one test.

    Passed to ``LocalExecutor`` and ``SshExecutor`` so that logging is handled
    inside the executor via ``LogConfig``.

    Attributes:
        test_id:          Test function name (truncated to 20 chars in prefix).
        node_label:       Node identifier from ``NodeSpec.label``.
        gpu_label:        GPU identifier, e.g. ``"GPU-0"`` or ``"GPU-2,3"``.
        log_path:         Per-test log file (append mode), or ``None``.
        session_log_path: Session-wide aggregate log file (append mode), or ``None``.
    """

    test_id: str
    node_label: str
    gpu_label: str
    log_path: str | None = None
    session_log_path: str | None = None

    @property
    def prefix(self) -> str:
        """Bracketed context prefix, e.g. ``"[test_hip_runtime    | localhost    | GPU-0    ]"``."""
        t = self.test_id[:_TEST_WIDTH].ljust(_TEST_WIDTH)
        n = self.node_label[:_NODE_WIDTH].ljust(_NODE_WIDTH)
        g = self.gpu_label[:_GPU_WIDTH].ljust(_GPU_WIDTH)
        return f"[{t}| {n}| {g}]"


def _append_to_file(path: str | None, content: str) -> None:
    """Append *content* to *path* in UTF-8 text mode.

    Creates parent directories if needed.  Silently logs a warning on
    ``OSError`` so a log write failure never aborts a test.

    Args:
        path:    Destination file path, or ``None`` to skip.
        content: Text to append.
    """
    if not path or not content:
        return
    try:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(content)
            if not content.endswith("\n"):
                fh.write("\n")
    except OSError as exc:
        logger.warning("LogConfig: cannot write to %s: %s", path, exc)


def run_with_logging(
    log_cfg: LogConfig,
    command: str,
    timeout: float | None,
    inner_run: Callable[[str, float | None], ExecutionResult],
) -> ExecutionResult:
    """Execute *command* via *inner_run*, writing prefixed output to log files.

    Implements the 7-step logging protocol shared between ``LocalExecutor``
    and ``SshExecutor`` when a ``LogConfig`` is attached.

    Args:
        log_cfg:   Per-test logging context.
        command:   Shell command string.
        timeout:   Forwarded to *inner_run* unchanged.
        inner_run: Callable matching ``(command, timeout) -> ExecutionResult``.

    Returns:
        ``ExecutionResult`` with unmodified ``stdout`` / ``stderr``.
    """
    prefix = log_cfg.prefix
    log_path = log_cfg.log_path
    session_log = log_cfg.session_log_path

    # Step 1: capture issuance timestamp.
    t_cmd = datetime.now().strftime("%H:%M:%S.%f")[:12]

    # Step 2: pre-execution banner.
    cmd_display = command if len(command) <= 120 else command[:117] + "..."
    logger.info("++Exec %s$ %s", f"[{log_cfg.node_label}] " if log_cfg.node_label else "", cmd_display)

    # Step 3: write command line to log files *before* the command runs so the
    # timestamp reflects dispatch time, not completion time.
    cmd_line = f"{t_cmd} {prefix} : {command}\n"
    _append_to_file(log_path, cmd_line)
    _append_to_file(session_log, cmd_line)

    # Step 4: delegate to the real executor.
    raw = inner_run(command, timeout)

    # Step 5: capture completion timestamp.
    t_out = datetime.now().strftime("%H:%M:%S.%f")[:12]

    # Step 6: emit stderr lines via the output logger (not streamed live).
    for line in (raw.stderr or "").splitlines():
        if line.strip():
            _output_logger.info("%s", line)

    # Step 7: invocation summary + write output lines to log files.
    logger.info("%s %.2fs (exit=%d) %s", prefix, raw.duration, raw.exit_code, cmd_display)
    output_lines = []
    for line in (raw.stdout or "").splitlines():
        if line.strip():
            output_lines.append(f"{t_out} {prefix} : {line}")
    for line in (raw.stderr or "").splitlines():
        if line.strip():
            output_lines.append(f"{t_out} {prefix} : {line}")
    if output_lines:
        content = "\n".join(output_lines) + "\n"
        _append_to_file(log_path, content)
        _append_to_file(session_log, content)

    return raw
