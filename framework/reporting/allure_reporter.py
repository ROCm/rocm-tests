# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
allure_reporter.py -- Allure step-level reporting helpers.

Provides thin wrappers around the ``allure`` library so that:
  - Framework code and tests emit structured steps visible in the Allure HTML report.
  - Metrics (float KEY=value) are attached as parameters for trend comparison.
  - Execution artifacts (stdout, stderr, state dumps) are attached per step.
  - All calls degrade gracefully when ``allure-pytest`` is not installed
    (e.g. unit test runs without the Allure pytest plugin active).

Usage in tests:
    from framework.reporting.allure_reporter import step, attach_text, report_metric

    def test_hip_runtime(target_executor):
        with step("Compile HIP kernel"):
            result = target_executor.run("hipcc kernel.cpp -o kernel")
        with step("Execute kernel"):
            result = target_executor.run("./kernel")
            attach_text(result.stdout, name="kernel_stdout")
            report_metric("THROUGHPUT_TFLOPS", 1.23)

Usage via fixture (preferred — injected automatically):
    def test_foo(target_executor, allure_reporter):
        allure_reporter.step("Run benchmark", target_executor.run, "python3 bench.py")

The ``allure_reporter`` pytest fixture is registered in artifacts_plugin.py and
injected automatically into any test that declares it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
import contextlib
import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal — graceful import of allure
# ---------------------------------------------------------------------------

try:
    import allure as _allure

    _ALLURE_AVAILABLE = True
except ImportError:
    _allure = None  # type: ignore[assignment]
    _ALLURE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def step(title: str) -> Iterator[None]:
    """Context manager that wraps a block of test code as a named Allure step.

    When allure-pytest is not active, the block still executes normally.

    Args:
        title: Human-readable step name shown in the Allure report.

    Example:
        with step("Install ROCm packages"):
            result = target_executor.run("apt-get install -y rocm-hip")
            assert result.ok
    """
    if _ALLURE_AVAILABLE:
        with _allure.step(title):
            yield
    else:
        logger.debug("[step] %s", title)
        yield


def attach_text(content: str, name: str = "output") -> None:
    """Attach a plain-text string to the current Allure test/step.

    Args:
        content: Text to attach (e.g. stdout, log excerpt).
        name:    Label shown in the Allure attachment panel.
    """
    if _ALLURE_AVAILABLE and content:
        _allure.attach(
            content,
            name=name,
            attachment_type=_allure.attachment_type.TEXT,
        )


def attach_json(content: str, name: str = "data") -> None:
    """Attach a JSON string to the current Allure test/step.

    Args:
        content: JSON-encoded string.
        name:    Label shown in the Allure attachment panel.
    """
    if _ALLURE_AVAILABLE and content:
        _allure.attach(
            content,
            name=name,
            attachment_type=_allure.attachment_type.JSON,
        )


def report_metric(key: str, value: float, unit: str = "") -> None:
    """Record a numeric metric as an Allure dynamic parameter.

    The metric is visible in the Allure report's Parameters tab and is
    used by the baseline_plugin for regression detection.

    Args:
        key:   Metric name matching the KEY= prefix in command stdout.
        value: Float metric value.
        unit:  Optional unit string (e.g. "TFLOPS", "ms", "GB/s").
    """
    display = f"{value:.4g} {unit}".strip() if unit else f"{value:.4g}"
    if _ALLURE_AVAILABLE:
        _allure.dynamic.parameter(key, display)
    logger.info("METRIC %s = %s", key, display)


def allure_step(title: str | None = None) -> Callable:
    """Decorator: wrap a function as a named Allure step.

    Args:
        title: Step title. Defaults to the function's ``__name__``.

    Returns:
        Decorated function that appears as a step in Allure reports.

    Example:
        @allure_step("Verify GPU health")
        def verify_gpu_health(result):
            assert result.ok
    """

    def decorator(func: Callable) -> Callable:
        label = title or func.__name__.replace("_", " ").title()

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with step(label):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# AllureReporter — injectable fixture object
# ---------------------------------------------------------------------------


class AllureReporter:
    """Stateful reporter bound to a single test's Allure context.

    Injected via the ``allure_reporter`` fixture (registered in artifacts_plugin).
    Provides a fluent API for structured step-level reporting inside test functions.

    Attributes:
        test_name: Name of the owning test function.
        _step_count: Running counter of steps executed so far.
    """

    def __init__(self, test_name: str) -> None:
        """Initialize the reporter for a given test.

        Args:
            test_name: pytest node ID or function name for log correlation.
        """
        self.test_name = test_name
        self._step_count = 0

    def step(  # pylint: disable=keyword-arg-before-vararg
        self,
        title: str,
        func: Callable | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute *func* inside a named Allure step, or use as context manager.

        Args:
            title:  Human-readable step name.
            func:   Optional callable to execute within the step.
            *args:  Positional arguments forwarded to *func*.
            **kwargs: Keyword arguments forwarded to *func*.

        Returns:
            Return value of *func* when called, or a context manager otherwise.

        Example (inline call):
            result = reporter.step("Run benchmark", target_executor.run, "python3 bench.py")

        Example (context manager):
            with reporter.step("Compile kernel"):
                result = target_executor.run("hipcc kernel.cpp -o kernel")
        """
        self._step_count += 1
        if func is not None:
            with step(title):
                return func(*args, **kwargs)
        return step(title)

    def attach(self, content: str, name: str = "output") -> None:
        """Attach text content to the current Allure context.

        Args:
            content: Text to attach.
            name:    Attachment label.
        """
        attach_text(content, name=name)

    def metric(self, key: str, value: float, unit: str = "") -> None:
        """Record a numeric metric as an Allure dynamic parameter.

        Args:
            key:   Metric name.
            value: Float value.
            unit:  Optional unit string.
        """
        report_metric(key, value, unit)
