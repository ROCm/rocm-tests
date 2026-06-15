# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
helpers.py -- Shared utility functions and data types.

Provides:
    ExecutionResult       -- Immutable result of running a shell command on a GPU.
    Outcome               -- Enum of all possible test outcomes.
    executor_log_path     -- Per-test executor log path helper.
    gpu_monitor_log_path  -- Per-test GPU monitor log path helper.

These are importable from both framework modules and test files:
    from framework.common.helpers import ExecutionResult, Outcome
"""

from __future__ import annotations

from dataclasses import dataclass
import enum
import logging
import os
import pathlib

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionResult:
    """Immutable result of a command executed on a GPU node.

    Attributes:
        exit_code: Shell exit code (0 = success).
        stdout:    Captured standard output, stripped of trailing whitespace.
        stderr:    Captured standard error, stripped of trailing whitespace.
        duration:  Wall-clock seconds the command took to complete.
    """

    exit_code: int
    stdout: str
    stderr: str
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        """True when exit_code is 0."""
        return self.exit_code == 0

    def __str__(self) -> str:
        """Human-readable multiline representation with stdout/stderr on separate lines."""
        lines = [f"ExecutionResult(exit_code={self.exit_code}, duration={self.duration:.3f}s)"]
        if self.stdout:
            lines.append("  stdout:")
            for line in self.stdout.splitlines():
                lines.append(f"    {line}")
        if self.stderr:
            lines.append("  stderr:")
            for line in self.stderr.splitlines():
                lines.append(f"    {line}")
        return "\n".join(lines)


def _executor_log_file(artifact_dir: str, test_name: str, nodeid: str | None = None) -> pathlib.Path:
    """Return the ``pathlib.Path`` for a per-test executor log — pure, no I/O side-effects.

    Shared by :func:`executor_log_path` (which creates + truncates) and by
    read-only callers (e.g. the Allure attachment fixture in teardown) that must
    locate the file without accidentally wiping it.

    Args:
        artifact_dir: Value of ``framework_config.framework.artifact_dir``.
        test_name:    ``request.node.name`` from the calling fixture.
        nodeid:       Full pytest node ID.  The immediate parent dir of the test
                      file becomes the log sub-directory when provided.

    Returns:
        ``pathlib.Path`` object (parent dir is NOT created by this function).
    """
    if nodeid and "::" in nodeid:
        test_file = nodeid.split("::")[0]
        test_subdir = pathlib.Path(test_file).parent.name
    else:
        test_subdir = "executor-logs"
    safe_func = test_name.replace("/", "_").replace("::", "__")
    return pathlib.Path(artifact_dir) / test_subdir / f"{safe_func}.log"


def executor_log_path(artifact_dir: str, test_name: str, nodeid: str | None = None) -> str:
    """Return a per-test executor log path, creating its directory and truncating the file.

    When *nodeid* is provided the immediate parent directory of the test file
    is used as the sub-directory under *artifact_dir*, so logs naturally group
    by test area (e.g. ``output/artifacts/compiler/test_2_llvm_stress.log``).
    Falls back to a flat ``executor-logs/`` sub-directory when *nodeid* is absent.

    Creates the directory on demand and truncates the log file so each session
    starts clean.  Use :func:`_executor_log_file` when you only need the path
    without the truncation side-effect (e.g. reading the file in teardown).

    This is the single source of truth used by all executor-providing fixtures
    (``target_executor``, ``multi_gpu_fixture``, ``cpu_executor``).

    Args:
        artifact_dir: Value of ``framework_config.framework.artifact_dir``.
        test_name:    ``request.node.name`` from the calling fixture.
        nodeid:       Full pytest node ID (``request.node.nodeid``).

    Returns:
        Path string ending in ``<safe_name>.log``.
    """
    log_file = _executor_log_file(artifact_dir, test_name, nodeid)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # Truncate to start fresh — avoids accumulating content from prior sessions.
    log_file.write_text("", encoding="utf-8")
    return str(log_file)


def gpu_monitor_log_path(artifact_dir: str, test_name: str) -> str:
    """Return a per-test GPU monitor log path under *artifact_dir*/executor-logs/.

    Same name sanitization as :func:`executor_log_path`; file is named
    ``<safe_name>_gpu_monitor.log``.  The directory is created on demand.
    The file is *not* pre-truncated — ``GpuBackgroundMonitor`` opens it fresh.

    Args:
        artifact_dir: Value of ``framework_config.framework.artifact_dir``.
        test_name:    ``request.node.name`` from the calling fixture.

    Returns:
        Path string ending in ``<safe_name>_gpu_monitor.log``.
    """
    safe = test_name.replace("/", "_").replace("::", "__")
    log_dir = os.path.join(artifact_dir, "executor-logs")
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(log_dir, f"{safe}_gpu_monitor.log")


# ---------------------------------------------------------------------------
# Test outcome classification
# ---------------------------------------------------------------------------


class Outcome(str, enum.Enum):
    """All possible test outcomes — each maps to a distinct root cause.

    Used by ``outcome_fixture`` and ``reports_plugin`` to classify test runs
    and attach outcome labels to Allure reports.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    TIMEOUT = "TIMEOUT"
    KILLED = "KILLED"
    ERROR = "ERROR"
    HEALTH_FAIL = "HEALTH_FAIL"
    PERF_DROP = "PERF_DROP"
    PERF_GAIN = "PERF_GAIN"
