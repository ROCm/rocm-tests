# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
labeled_executor.py -- Executor wrapper: context-tagged console output and timestamped log files.

When multiple GPU tests run in parallel across different nodes and GPUs,
``LabeledExecutor`` provides two complementary output streams:

**Console** (via pytest live logging, ``log_cli = true`` in ``pyproject.toml``):
    Each stdout/stderr line from the subprocess is emitted as a Python
    ``logging.INFO`` record through the ``rocm.output`` logger.  The
    wall-clock timestamp comes from ``log_cli_format`` (``%(asctime)s``)
    automatically — no manual timestamp needed::

        14:23:46 INFO  rocm.output  GPU 0: MI300A, VRAM: 32768 MiB
        14:23:46 INFO  rocm.output  Found 1 GPU(s)
        14:23:46 INFO  ...labeled_executor  [test_hip_runtime | localhost | GPU-0] 0.84s (exit=0) rocm-smi

    Output lines are captured by ``caplog`` → attached to Allure reports.

**Log files** (per-test and session aggregate):
    Every line — command and output — is written with a context prefix and
    timestamp so the files are both grep-friendly and self-identifying::

        14:23:46.001 [test_hip_runtime    | localhost    | GPU-0    ] : rocm-smi --showid
        14:23:46.001 [test_hip_runtime    | localhost    | GPU-0    ] : GPU 0: MI300A, VRAM: 32768 MiB
        14:23:46.002 [test_hip_runtime    | localhost    | GPU-0    ] : Found 1 GPU(s)

``result.stdout`` / ``result.stderr`` are returned unmodified so assertions
work without prefix stripping.

Usage (via ``NodeSlot.make_executor()`` — not constructed directly in tests)::

    inner = LocalExecutor(gpu_index=0, stream_stdout=False, stream_stderr=False)
    exec_ = LabeledExecutor(
        inner=inner,
        test_id="test_hip_runtime",
        node_label="localhost",
        gpu_label="GPU-0",
        log_path="output/artifacts/executor-logs/test_hip_runtime.log",
        session_log_path="output/artifacts/session.log",
    )
    result = exec_.run("rocm-smi --showid")
"""

from __future__ import annotations

from datetime import datetime
import logging
import pathlib

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

# Dedicated logger for subprocess output lines.  All stdout/stderr from
# executed commands flows through this logger so it appears on the console
# with pytest's %(asctime)s timestamp (log_cli_format in pyproject.toml)
# and is captured by caplog → Allure reports.  Framework diagnostic messages
# use the module-level `logger` and keep their own name/file/lineno context.
_output_logger = logging.getLogger("rocm.output")

# Column widths for the context prefix — wide enough to be readable without
# wrapping most terminal widths.
_TEST_WIDTH = 20
_NODE_WIDTH = 12
_GPU_WIDTH = 10


class LabeledExecutor(AbstractExecutor):
    """Wraps any executor: streams output to console via logging and writes prefixed log files.

    Each ``run()`` call:
    - Emits every stdout/stderr line through the ``rocm.output`` logger so it
      appears on the console with pytest's ``%(asctime)s`` timestamp.
    - Writes each line to ``log_path`` and ``session_log_path`` with the full
      ``HH:MM:SS.mmm [test_id | node | GPU] : content`` prefix.
    - Emits one invocation summary INFO line with duration and exit code.

    ``result.stdout`` and ``result.stderr`` are returned unmodified so plain
    substring assertions work without prefix stripping.

    Attributes:
        inner:            The backing executor (``LocalExecutor``, ``SshGpuExecutor``, etc.).
        test_id:          Test function name (truncated to 20 chars in the prefix).
        node_label:       Node identifier from ``NodeSpec.label``.
        gpu_label:        GPU identifier string, e.g. ``"GPU-0"`` or ``"GPU-2,3"``.
        log_path:         Per-test log file path (append mode), or ``None``.
        session_log_path: Session-wide log file path (append mode), or ``None``.
    """

    def __init__(
        self,
        inner: AbstractExecutor,
        test_id: str,
        node_label: str,
        gpu_label: str,
        log_path: str | None = None,
        session_log_path: str | None = None,
    ) -> None:
        self.inner = inner
        self.test_id = test_id
        self.node_label = node_label
        self.gpu_label = gpu_label
        self.log_path = log_path
        self.session_log_path = session_log_path

    # ------------------------------------------------------------------
    # Prefix helper
    # ------------------------------------------------------------------

    @property
    def prefix(self) -> str:
        """Context prefix used in log bookmarks.

        Returns:
            Bracketed context string, e.g.
            ``"[test_hip_runtime    | localhost    | GPU-0    ]"``
        """
        t = self.test_id[:_TEST_WIDTH].ljust(_TEST_WIDTH)
        n = self.node_label[:_NODE_WIDTH].ljust(_NODE_WIDTH)
        g = self.gpu_label[:_GPU_WIDTH].ljust(_GPU_WIDTH)
        return f"[{t}| {n}| {g}]"

    # ------------------------------------------------------------------
    # Log file helper
    # ------------------------------------------------------------------

    def _append_to_file(self, path: str | None, content: str) -> None:
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
            logger.warning("LabeledExecutor: cannot write to %s: %s", path, exc)

    def _write_prefixed_output(self, stdout: str, stderr: str, command: str, t_cmd: str, t_out: str) -> None:
        """Write command + output lines to log files with prefix and per-role timestamp.

        The command line uses *t_cmd* (captured before the command ran) so the
        log shows when the command was *issued*.  Output lines use *t_out*
        (captured after the command returned) so long-running commands show a
        meaningful gap between issuance and completion::

            09:00:00.001 [test_hip_runtime    | localhost    | GPU-0    ] : rocm-smi --showid
            09:00:00.843 [test_hip_runtime    | localhost    | GPU-0    ] : GPU 0: MI300A, VRAM: 32768 MiB

        Args:
            stdout:  Captured stdout text (may be empty).
            stderr:  Captured stderr text (may be empty).
            command: The shell command that was executed (written as first line).
            t_cmd:   Timestamp string (``HH:MM:SS.mmm``) from before ``inner.run()``.
            t_out:   Timestamp string (``HH:MM:SS.mmm``) from after ``inner.run()``.
        """
        lines = []
        if command.strip():
            lines.append(f"{t_cmd} {self.prefix} : {command}")
        for line in (stdout or "").splitlines():
            if line.strip():
                lines.append(f"{t_out} {self.prefix} : {line}")
        for line in (stderr or "").splitlines():
            if line.strip():
                lines.append(f"{t_out} {self.prefix} : {line}")
        if not lines:
            return
        content = "\n".join(lines) + "\n"
        self._append_to_file(self.log_path, content)
        self._append_to_file(self.session_log_path, content)

    # ------------------------------------------------------------------
    # AbstractExecutor contract
    # ------------------------------------------------------------------

    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Run *command* via the inner executor; stream output and write prefixed log files.

        Console output (via pytest live logging with ``log_cli = true``):
            Each stdout/stderr line is emitted through the ``rocm.output`` logger
            so it appears with pytest's ``%(asctime)s`` timestamp and is captured
            by ``caplog`` for Allure attachment::

                14:23:46 INFO  rocm.output  GPU 0: MI300A, VRAM: 32768 MiB
                14:23:46 INFO  rocm.output  Found 1 GPU(s)
                14:23:46 INFO  ...  [test_hip_runtime | localhost | GPU-0] 0.84s (exit=0) rocm-smi

        Log files (per-test and session):
            Each line is written with full context prefix and timestamp::

                14:23:46.001 [test_hip_runtime | localhost | GPU-0] : rocm-smi --showid
                14:23:46.001 [test_hip_runtime | localhost | GPU-0] : GPU 0: MI300A, VRAM: 32768 MiB

        ``result.stdout`` / ``result.stderr`` are returned unmodified.

        Args:
            command: Shell command string passed through to the inner executor.
            timeout: Forwarded to the inner executor unchanged.

        Returns:
            ``ExecutionResult`` with the original (unlabeled) ``stdout``,
            ``stderr``, ``exit_code``, and ``duration``.
        """
        # Capture issuance time before the command runs so the command line in the
        # log reflects when the test *dispatched* the command, not when it completed.
        t_cmd = datetime.now().strftime("%H:%M:%S.%f")[:12]

        # Pre-execution banner — mirrors the reference "++Exec $ <command>" pattern.
        # node_label is non-empty only for remote nodes; localhost simplifies to "++Exec $ ...".
        cmd_display = command if len(command) <= 120 else command[:117] + "..."
        logger.info("++Exec %s$ %s", f"[{self.node_label}] " if self.node_label else "", cmd_display)

        # Write command line to log files *before* inner.run() so the timestamp
        # reflects when the command was dispatched, not when it completed.
        cmd_line = f"{t_cmd} {self.prefix} : {command}\n"
        self._append_to_file(self.log_path, cmd_line)
        self._append_to_file(self.session_log_path, cmd_line)

        raw = self.inner.run(command, timeout=timeout)

        # Capture completion time after inner.run() returns — used for output lines.
        t_out = datetime.now().strftime("%H:%M:%S.%f")[:12]

        # Emit stderr lines via the output logger (stderr is not streamed live so
        # it needs the post-call path for console visibility + caplog/Allure capture).
        # stdout is omitted here — it streams live to sys.stdout.buffer via
        # stream_stdout=True in LocalExecutor, so logging it again would double-print.
        for line in (raw.stderr or "").splitlines():
            if line.strip():
                _output_logger.info("%s", line)

        # Invocation summary: context prefix + duration + exit code + command.
        logger.info("%s %.2fs (exit=%d) %s", self.prefix, raw.duration, raw.exit_code, cmd_display)

        # Write output lines to log files (command="" — already written pre-call above).
        self._write_prefixed_output(raw.stdout or "", raw.stderr or "", command="", t_cmd=t_cmd, t_out=t_out)

        return raw
