# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
helpers.py -- Shared utility functions and data types.

Provides:
    ExecutionResult  -- Immutable result of running a shell command on a GPU.
    parse_metric     -- Extract a named KEY=value float from command output.
    retry            -- Simple retry decorator for flaky operations.

These are importable from both framework modules and test files:
    from framework.common.helpers import ExecutionResult, parse_metric
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import enum
import functools
import logging
import os
import pathlib
import time
from typing import TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


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


def parse_metric(output: str, key: str) -> float | None:
    """Extract a ``KEY=<float>`` value from multi-line command output.

    Tests emit metrics in the form ``KEY=<value>`` on their own line so that
    the baseline fixture can compare them automatically.

    Args:
        output: Multi-line stdout from an ExecutionResult.
        key:    Exact key prefix, e.g. ``"THROUGHPUT_TFLOPS"``.

    Returns:
        Parsed float value, or None if the key is not present.

    Example:
        >>> result = gpu_fixture.run("python3 -c 'print(\"THROUGHPUT_TFLOPS=1.23\")'")
        >>> value = parse_metric(result.stdout, "THROUGHPUT_TFLOPS")
        >>> assert value == 1.23
    """
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(f"{key}="):
            raw = line.split("=", 1)[1].strip()
            try:
                return float(raw)
            except ValueError:
                logger.warning("Could not parse metric %s value: %r", key, raw)
    return None


def executor_log_path(artifact_dir: str, test_name: str, nodeid: str | None = None) -> str:
    """Return a per-test executor log path mirroring the test directory structure.

    When *nodeid* is provided the immediate parent directory of the test file
    is used as the sub-directory under *artifact_dir*, so logs naturally group
    by test area (e.g. ``output/artifacts/compiler/test_2_llvm_stress.log``).
    Falls back to a flat ``executor-logs/`` sub-directory when *nodeid* is absent.

    Creates the directory on demand and truncates the log file so each session
    starts clean.

    This is the single source of truth used by all executor-providing fixtures
    (``target_executor``, ``multi_gpu_fixture``, ``cpu_executor``,
    ``session_executor``) to avoid duplicating the path-construction logic.

    Args:
        artifact_dir: Value of ``framework_config.framework.artifact_dir``.
        test_name:    ``request.node.name`` from the calling fixture.
        nodeid:       Full pytest node ID (``request.node.nodeid``).  When
                      provided the immediate parent dir of the test file is used
                      as the log sub-directory.

    Returns:
        Path string ending in ``<safe_name>.log``.
    """
    if nodeid and "::" in nodeid:
        # "tests/e2e/compiler/test_2_llvm.py::test_func" → "compiler"
        test_file = nodeid.split("::")[0]
        test_subdir = pathlib.Path(test_file).parent.name
    else:
        test_subdir = "executor-logs"

    log_dir = pathlib.Path(artifact_dir) / test_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    safe_func = test_name.replace("/", "_").replace("::", "__")
    log_file = log_dir / f"{safe_func}.log"
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
# Test outcome classification (merged from framework/results/classifier.py)
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


def classify(
    result: ExecutionResult | None = None,
    *,
    exit_code: int | None = None,
    timed_out: bool = False,
    killed: bool = False,
    framework_error: str | None = None,
    health_passed: bool = True,
    baseline_comparisons: list | None = None,
) -> Outcome:
    """Classify a test execution into a single Outcome.

    Args:
        result:               ExecutionResult from gpu_fixture.run(), or None.
        exit_code:            Override exit code (used when result is None).
        timed_out:            True if the command was killed by timeout.
        killed:               True if the framework killed the process (OOM/watchdog).
        framework_error:      Non-None if a framework-level error occurred.
        health_passed:        False if pre- or post-execution health check failed.
        baseline_comparisons: List of BaselineComparison from baseline_fixture.compare().

    Returns:
        The most specific Outcome for the test run.
    """
    if framework_error:
        return Outcome.ERROR
    if not health_passed:
        return Outcome.HEALTH_FAIL
    if killed:
        return Outcome.KILLED
    if timed_out:
        return Outcome.TIMEOUT

    effective_exit = exit_code if exit_code is not None else (result.exit_code if result else 0)
    if effective_exit != 0:
        return Outcome.FAIL

    if baseline_comparisons:
        regressions = [c for c in baseline_comparisons if not c.passed and (c.delta_pct or 0) < 0]
        gains = [c for c in baseline_comparisons if not c.passed and (c.delta_pct or 0) > 0]
        if regressions:
            return Outcome.PERF_DROP
        if gains:
            return Outcome.PERF_GAIN

    return Outcome.PASS


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator: retry a function up to *max_attempts* times on exception.

    Args:
        max_attempts: Total number of attempts before re-raising.
        delay:        Seconds to wait between attempts.
        exceptions:   Exception types that trigger a retry.

    Returns:
        Decorated function that retries on the specified exceptions.

    Example:
        @retry(max_attempts=3, delay=2.0, exceptions=(OSError,))
        def flaky_io():
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        logger.warning(
                            "Attempt %d/%d failed (%s). Retrying in %.1fs…",
                            attempt,
                            max_attempts,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
            raise last_exc

        return wrapper  # type: ignore[return-value]

    return decorator
