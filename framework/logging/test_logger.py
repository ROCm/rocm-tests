# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_logger.py -- Unified per-test logging for the rocm-tests framework.

``TestLogger`` provides:

- One ``FileHandler`` per test (opened at fixture setup, closed at teardown) — zero
  per-``run()`` ``open()`` or ``stat`` syscalls.
- Block-header format: one ``>> [ts] node | gpu | $ cmd`` line per command,
  followed by verbatim stdout/stderr — no per-line prefix construction.
- Compact structured event lines in ``session.log`` via per-call ``open("a")``
  (POSIX ``O_APPEND`` — cross-process-safe for xdist workers).
- Fixed ``logging.getLogger("rocm.test")`` base logger wrapped by ``LoggerAdapter``
  — zero logger-registry growth regardless of test count.

Session log event format (grep-able)::

    ACQR   HH:MM:SS.mmm  test_name                     detail ...
    CMD    HH:MM:SS.mmm  test_name                     node $ cmd
    END    HH:MM:SS.mmm  test_name                     exit=N
    OUT    HH:MM:SS.mmm  test_name
      <stdout line 1>
      <stdout line 2>
      ...
    ERR    HH:MM:SS.mmm  test_name
      <stderr line 1>
      ...
    REL    HH:MM:SS.mmm  test_name                     detail ...
    BKGD   HH:MM:SS.mmm  test_name                     detail ...
    SUMM   HH:MM:SS.mmm  total=N passed=N failed=N skipped=N

OUT/ERR blocks are capped at ``_SESSION_OUTPUT_CAP`` (3800) chars of content.
Each block is written as a single ``write()`` syscall so concurrent xdist
workers cannot interleave lines from different tests.  Full verbatim output is
always available in the per-test log file.

Per-test log format::

    # test=<name> node=<label> gpu=<label> started=<iso>
    >> [HH:MM:SS.mmm] node | gpu | $ command
    <verbatim stdout>
    <verbatim stderr (WARNING level)>
    >> [HH:MM:SS.mmm] node | gpu | $ next command
    ...
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys
import time

_RAW_FORMATTER = logging.Formatter("%(message)s")
_CONSOLE_FORMATTER = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

# Fixed base logger — one object for the entire process lifetime.
# LoggerAdapter injects per-test context without creating new logger objects.
# Zero registry growth across thousands of tests.
_BASE_LOGGER = logging.getLogger("rocm.test")
# Set once at import time — never inside TestLogger.__init__ (would re-fire per test).
# propagate=False: prevents pytest's root StreamHandler from double-printing with its
# "%(levelname)s %(name)s" prefix. Records reach only the handlers TestLogger adds.
_BASE_LOGGER.setLevel(logging.DEBUG)
_BASE_LOGGER.propagate = False

# Column width for test_id in session.log alignment
_SESSION_ID_WIDTH = 30

# Maximum bytes written per OUT/ERR line in session.log.  Kept within POSIX
# O_APPEND atomic-write guarantee (typically 4096 bytes) so concurrent xdist
# workers cannot interleave partial lines.  Full output is always in the per-test
# log file; session.log carries a compact, grep-able preview.
_SESSION_OUTPUT_CAP = 3800


def _ts() -> str:
    """Return a compact HH:MM:SS.mmm timestamp string."""
    return datetime.now().strftime("%H:%M:%S.%f")[:12]


class TestLogger:
    """Unified logger for a single test-executor pair.

    Lifecycle:
        - Created in fixture setup (``NodeSlot.make_executor`` or directly in
          ``target_executor``).
        - ``close()`` called in fixture teardown (``finally`` block) to remove
          handlers and flush file.

    Attributes:
        test_id:    Test function name.
        node_label: Node label from ``NodeSpec.label``.
        gpu_label:  GPU label, e.g. ``"GPU-0"`` or ``"GPU-2,3"``.
    """

    def __init__(
        self,
        test_id: str,
        node_label: str,
        gpu_label: str,
        log_path: str | None,
        session_log_path: str | None,
    ) -> None:
        """Open handlers for this test.

        Args:
            test_id:          Test function name.
            node_label:       Node label for the block header.
            gpu_label:        GPU label for the block header.
            log_path:         Per-test log file path (created fresh, mode ``"w"``).
            session_log_path: Session-wide aggregate log path (append mode per event).
        """
        self.test_id = test_id
        self.node_label = node_label
        self.gpu_label = gpu_label
        self._session_log_path = session_log_path
        self._fh: logging.FileHandler | None = None
        self._ch: logging.StreamHandler | None = None

        # Per-test file handler: opened once, raw "%(message)s" format.
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
            self._fh.setFormatter(_RAW_FORMATTER)
            _BASE_LOGGER.addHandler(self._fh)
            # Write self-contained metadata header as the first line.
            meta = (
                f"# test={test_id} node={node_label} gpu={gpu_label}"
                f" started={datetime.now().isoformat(timespec='seconds')}\n"
            )
            assert self._fh.stream is not None  # FileHandler opens stream in __init__
            self._fh.stream.write(meta)
            self._fh.stream.flush()

        # Console handler: scoped to this test to avoid double-timestamp noise.
        self._ch = logging.StreamHandler(sys.stdout)
        self._ch.setFormatter(_CONSOLE_FORMATTER)
        _BASE_LOGGER.addHandler(self._ch)

        self._adapter = logging.LoggerAdapter(
            _BASE_LOGGER,
            {"test_id": test_id, "node_label": node_label, "gpu_label": gpu_label},
        )

    # ------------------------------------------------------------------
    # Command block logging (called by executor run())
    # ------------------------------------------------------------------

    def cmd_start(self, cmd: str) -> float:
        """Emit block header to per-test log and console; return monotonic start time.

        The header is written *before* the subprocess runs so a crash or hang
        leaves a record of what was dispatched.

        Args:
            cmd: Shell command string about to be executed.

        Returns:
            ``time.monotonic()`` captured immediately after the header is written.
        """
        ts = _ts()
        # truncate the console logs to max 300 characters
        cmd_log_max = 300
        tail = f"… <{len(cmd) - cmd_log_max} chars truncated>"
        display_cmd = cmd if len(cmd) <= cmd_log_max else cmd[:cmd_log_max] + tail
        header = f">> [{ts}] {self.node_label} | {self.gpu_label} | $ {display_cmd}"
        _BASE_LOGGER.info(header)
        self._session_append(
            f"CMD   {ts} {self.test_id[: _SESSION_ID_WIDTH]:<{_SESSION_ID_WIDTH}}" f" {self.node_label} $ {cmd}\n"
        )
        return time.monotonic()

    def cmd_end(self, stdout: str, stderr: str, exit_code: int, start: float) -> None:
        """Write verbatim output to per-test log; record exit code and output in session.log.

        Per-test log receives the full verbatim stdout/stderr.  session.log
        receives an ``END`` event line followed by ``OUT`` and ``ERR`` preview
        lines (capped at ``_SESSION_OUTPUT_CAP`` bytes each so concurrent
        xdist workers cannot split a write across POSIX O_APPEND boundaries).
        Newlines are encoded as ``↵`` to keep session.log single-line-per-event
        and grep-able.

        Args:
            stdout:    Subprocess stdout string (verbatim, no prefix).
            stderr:    Subprocess stderr string (written at WARNING level so it
                       stands out in the console and log).
            exit_code: Process exit code.
            start:     Monotonic timestamp from ``cmd_start()`` (unused here but
                       accepted for a consistent call signature).
        """
        # Strip once — reused for logger and session.log to avoid redundant allocation.
        stdout_s = stdout.rstrip()
        stderr_s = stderr.rstrip()
        if stdout_s:
            _BASE_LOGGER.info(stdout_s)
        if stderr_s:
            _BASE_LOGGER.warning(stderr_s)
        self._session_append(
            f"END   {_ts()} {self.test_id[: _SESSION_ID_WIDTH]:<{_SESSION_ID_WIDTH}} exit={exit_code}\n"
        )
        if stdout_s:
            self._session_append_output("OUT", stdout_s)
        if stderr_s:
            self._session_append_output("ERR", stderr_s)

    # ------------------------------------------------------------------
    # General / event logging
    # ------------------------------------------------------------------

    def info(self, msg: str, *args) -> None:
        """Log an informational message through the test adapter (file + console).

        Args:
            msg:  Log message (``%``-style format string).
            *args: Format arguments.
        """
        self._adapter.info(msg, *args)

    def warning(self, msg: str, *args) -> None:
        """Log a warning through the test adapter (file + console).

        Args:
            msg:  Log message.
            *args: Format arguments.
        """
        self._adapter.warning(msg, *args)

    def event(self, kind: str, detail: str) -> None:
        """Write a structured event to session.log only (no per-test log write).

        Used for lifecycle events that belong in the master timeline but not in
        the per-test output file: GPU acquire/release, background process
        start/stop, session summary.

        Args:
            kind:   Event kind tag (e.g. ``"ACQR"``, ``"REL"``, ``"BKGD"``).
            detail: Free-form detail string appended after the test id column.
        """
        self._session_append(f"{kind:<6}{_ts()} {self.test_id[: _SESSION_ID_WIDTH]:<{_SESSION_ID_WIDTH}} {detail}\n")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and remove handlers added for this test.

        Safe to call multiple times (idempotent after the first call).
        ``removeHandler`` is called unconditionally before flush/close so a
        failed flush never leaves a stale handler on ``_BASE_LOGGER``.
        """
        for h in (self._fh, self._ch):
            if h is not None:
                _BASE_LOGGER.removeHandler(h)  # always remove, even if flush fails
                try:
                    h.flush()
                    h.close()
                except (OSError, ValueError):
                    # ValueError: I/O operation on closed file — fd already closed.
                    pass
        self._fh = self._ch = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_append(self, line: str) -> None:
        """Append one structured line to session.log using O_APPEND semantics.

        Each call opens the file independently so concurrent xdist workers can
        safely interleave single-line writes.  A line is ≤ 200 bytes — well
        within the POSIX ``O_APPEND`` write-atomicity guarantee.

        Args:
            line: Text to append (should end with ``"\\n"``).
        """
        if not self._session_log_path:
            return
        try:
            with open(self._session_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
        except OSError as exc:
            _BASE_LOGGER.warning("TestLogger: session_log write error: %s", exc)

    def _session_append_output(self, kind: str, text: str) -> None:
        """Append an OUT or ERR block to session.log with human-readable newlines.

        Writes a two-part block: a header line (grep-able by kind tag) followed
        by the output lines indented with two spaces.  Truncates *text* to
        ``_SESSION_OUTPUT_CAP`` chars to keep session.log compact — full output
        is always in the per-test log file.

        The entire block is passed to ``_session_append`` as a single string so
        it lands in one ``write()`` syscall, preserving atomicity under POSIX
        ``O_APPEND`` for concurrent xdist workers.

        Args:
            kind: ``"OUT"`` for stdout or ``"ERR"`` for stderr.
            text: Subprocess output string (caller should already have rstripped).
        """
        if not self._session_log_path or not text:
            return
        truncated = ""
        if len(text) > _SESSION_OUTPUT_CAP:
            omitted = len(text) - _SESSION_OUTPUT_CAP
            text = text[:_SESSION_OUTPUT_CAP]
            truncated = f"\n  [+{omitted} chars omitted]"
        indented = "\n".join(f"  {ln}" for ln in text.splitlines())
        block = (
            f"{kind:<6}{_ts()} {self.test_id[: _SESSION_ID_WIDTH]:<{_SESSION_ID_WIDTH}}\n" f"{indented}{truncated}\n"
        )
        self._session_append(block)
