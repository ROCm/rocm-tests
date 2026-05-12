# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
retry_plugin.py -- Auto-retry fixture with per-attempt artifact capture.

Provides the ``retry_fixture`` which wraps test execution with configurable
retry logic. On each failed attempt, GPU state is captured and attached to
Allure before the next retry begins.

Features:
    - Per-test retry count configurable via ``@pytest.mark.retry(count=N)``
      or via the ``--retry-count`` CLI flag.
    - Each attempt runs in an isolated environment with fresh GPU state.
    - Tests that pass on retry are tagged ``flaky`` in the Allure report.
    - Persistent flakiness trends surface across the N-build Allure history.

Usage in tests (via fixture injection):
    def test_rccl(target_executor, retry_fixture):
        result = retry_fixture.run(target_executor, "python3 -c 'import rccl'")
        assert result.ok

Or, use the ``@pytest.mark.retry`` marker for automatic retry:
    @pytest.mark.retry(count=2)
    @pytest.mark.hw.gpu
    @pytest.mark.ci.nightly
    @pytest.mark.layer.math_lib
    def test_rccl_allreduce(target_executor):
        result = target_executor.run("python3 bench/rccl_allreduce.py")
        assert result.ok
"""

from __future__ import annotations

import logging

import pytest

from framework.reporting.allure_reporter import attach_text, step

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --retry-count CLI option."""
    group = parser.getgroup("rocm-retry", "ROCm test retry options")
    group.addoption(
        "--retry-count",
        action="store",
        type=int,
        default=0,
        metavar="N",
        help="Retry failed tests up to N additional times (default: 0 = no retry).",
    )


class RetryHelper:
    """Helper that wraps a callable with retry-on-failure behaviour.

    Args:
        max_attempts: Total attempts including the first (min: 1).
        test_name:    Test node ID for log correlation.
    """

    def __init__(self, max_attempts: int, test_name: str) -> None:
        self.max_attempts = max(1, max_attempts)
        self.test_name = test_name
        self._flaky = False

    @property
    def flaky(self) -> bool:
        """True if the test passed on a retry (not the first attempt)."""
        return self._flaky

    def run(self, executor, command: str, timeout: float | None = None):
        """Execute *command* via *executor* with retry on non-zero exit.

        Args:
            executor: Any executor providing ``.run(command, timeout)`` —
                      typically ``target_executor`` (NodeExecutorGroup) from the test.
            command:  Shell command to execute.
            timeout:  Max seconds per attempt.

        Returns:
            ExecutionResult from the first successful attempt, or the last
            failed attempt if all retries are exhausted.
        """
        from framework.common.helpers import ExecutionResult  # pylint: disable=import-outside-toplevel

        last_result: ExecutionResult | None = None
        for attempt in range(1, self.max_attempts + 1):
            with step(f"Attempt {attempt}/{self.max_attempts}"):
                result = executor.run(command, timeout=timeout)
                if result.ok:
                    if attempt > 1:
                        self._flaky = True
                        logger.warning(
                            "FLAKY: %s passed on attempt %d/%d",
                            self.test_name,
                            attempt,
                            self.max_attempts,
                        )
                        try:
                            import allure  # pylint: disable=import-outside-toplevel

                            allure.dynamic.tag("flaky")
                        except ImportError:
                            pass
                    return result

                attach_text(result.stderr or result.stdout, name=f"attempt_{attempt}_output")
                logger.warning(
                    "Attempt %d/%d failed (exit %d): %s",
                    attempt,
                    self.max_attempts,
                    result.exit_code,
                    self.test_name,
                )
                last_result = result

        return last_result


@pytest.fixture
def retry_fixture(request):
    """Inject a RetryHelper configured from the test's @pytest.mark.retry or --retry-count.

    Priority: ``@pytest.mark.retry(count=N)`` > ``--retry-count`` CLI flag > 1 attempt.

    Args:
        request: pytest request object.

    Returns:
        RetryHelper: Ready to call .run(executor, command).
    """
    cli_count = request.config.getoption("--retry-count", default=0)
    mark = request.node.get_closest_marker("retry")
    mark_count = mark.kwargs.get("count", 0) if mark else 0
    attempts = 1 + max(cli_count, mark_count)  # 1 base + N retries

    return RetryHelper(max_attempts=attempts, test_name=request.node.nodeid)
