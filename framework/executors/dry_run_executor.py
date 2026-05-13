# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
dry_run_executor.py -- Stub executor that returns synthetic success without GPU hardware.

Activated by the ``--no-gpu`` pytest flag (added by gpu_plugin.py).  All
``run()`` calls return a successful ExecutionResult with synthetic stdout so
that framework logic (marker validation, baseline comparison, artifact capture)
can be exercised in CI environments without AMD GPU hardware.

Tests marked ``@pytest.mark.hw.gpu`` are skipped automatically when ``--no-gpu``
is active — DryRunExecutor is therefore only reached by tests marked ``hw.cpu_only``
or by the gpu_fixture itself in mock mode.
"""

from __future__ import annotations

import logging

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import BackgroundProcess, NoOpBackgroundProcess

logger = logging.getLogger(__name__)


class DryRunExecutor(AbstractExecutor):
    """Return a synthetic successful ExecutionResult without running any command.

    Used for:
    - ``--no-gpu`` runs in CI (DryRun mode)
    - Unit tests that need an executor without real hardware
    - Mock testing of framework fixture logic
    """

    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Return a synthetic ExecutionResult without executing *command*.

        Args:
            command: Command that would have been executed (logged for traceability).
            timeout: Ignored in DryRun mode.

        Returns:
            ExecutionResult with exit_code=0, synthetic stdout, empty stderr.
        """
        logger.info("[DryRun] Skipping execution of: %s", command)
        synthetic_stdout = "DRY_RUN=1\nRESULT_OK\nTHROUGHPUT_TFLOPS=0.0\n"
        return ExecutionResult(
            exit_code=0,
            stdout=synthetic_stdout,
            stderr="",
            duration=0.0,
        )

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
    ) -> BackgroundProcess:
        """Return a ``NoOpBackgroundProcess`` without starting a real subprocess.

        Logs the command for traceability so dry-run sessions remain auditable.

        Args:
            command:  Command that would have been started (logged only).
            timeout:  Ignored in DryRun mode.
            log_path: Ignored in DryRun mode.

        Returns:
            ``NoOpBackgroundProcess`` with a synthetic ``ExecutionResult``.
        """
        logger.info("[DryRun] Skipping background start of: %s", command)
        return NoOpBackgroundProcess()  # type: ignore[return-value]
