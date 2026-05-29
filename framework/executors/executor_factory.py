# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
executor_factory.py -- Static executor factory for CLI-flag-driven backend selection.

ExecutorFactory.resolve() returns DryRunExecutor, ContainerExecutor, or LocalExecutor
based on --no-gpu / --container-mode / default priority. Used internally by
executor_plugin and remote_node_plugin — tests never call it directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.container_executor import ContainerExecutor
from framework.executors.cpu_executor import CpuExecutor
from framework.executors.dry_run_executor import DryRunExecutor
from framework.executors.local_executor import LocalExecutor
from framework.executors.ssh_executor import SshExecutor

if TYPE_CHECKING:
    import pytest

logger = logging.getLogger(__name__)


class ExecutorFactory:
    """Select and instantiate the correct executor from the active pytest config.

    This class is never instantiated — use its classmethods from a fixture.
    All executor selection logic lives here so that adding a new execution
    backend requires changing only this file and the matching CLI option
    registration in ``executor_plugin``.

    Classmethods:
        resolve()  Auto-select executor from ``--no-gpu`` / ``--container-mode``
                   CLI flags.
        cpu()      Return a ``CpuExecutor`` for CPU-only operations.
        remote()   Return a new ``SshExecutor`` for a named remote host.
    """

    @classmethod
    def resolve(
        cls,
        request: pytest.FixtureRequest,
        gpu_index: int = 0,
        log_path: str | None = None,
    ) -> AbstractExecutor:
        """Return the executor appropriate for the current pytest session.

        Reads ``--no-gpu``, ``--container-mode``, ``--container-image``, and
        ``--container-runtime`` from the active pytest config to decide which
        backend to build.

        Args:
            request:   Pytest fixture request — provides access to CLI options.
            gpu_index: AMD GPU ordinal used when building ``LocalExecutor`` or
                       ``ContainerExecutor`` (ignored for ``DryRunExecutor``).
            log_path:  Per-test log file path forwarded to ``LocalExecutor``.
                       ``None`` disables disk logging.

        Returns:
            One of ``LocalExecutor``, ``ContainerExecutor``, or ``DryRunExecutor``
            depending on active CLI flags.
        """
        config = request.config

        if config.getoption("--no-gpu", default=False):
            logger.debug("ExecutorFactory: --no-gpu active → DryRunExecutor")
            return DryRunExecutor()

        if config.getoption("--container-mode", default=False):
            image = config.getoption("--container-image", default="rocm/pytorch:latest")
            runtime = config.getoption("--container-runtime", default="docker")
            logger.debug(
                "ExecutorFactory: --container-mode active → ContainerExecutor(image=%s, gpu_index=%d, runtime=%s)",
                image,
                gpu_index,
                runtime,
            )
            return ContainerExecutor(
                image=image,
                gpu_index=gpu_index,
                runtime=runtime,
            )

        logger.debug("ExecutorFactory: default → LocalExecutor(gpu_index=%d)", gpu_index)
        return LocalExecutor(
            gpu_index=gpu_index,
            stream_stdout=False,
            log_path=log_path,
        )

    @classmethod
    def cpu(cls) -> CpuExecutor:
        """Return a ``CpuExecutor`` for commands that need no GPU environment.

        Convenience shortcut for non-fixture contexts (e.g. inside
        ``ContainerExecutor._host``).  In test code, prefer the
        ``cpu_executor`` fixture.

        Returns:
            A new ``CpuExecutor`` with default settings.
        """
        return CpuExecutor()

    @classmethod
    def remote(
        cls,
        host: str,
        user: str | None = None,
        key_path: str | None = None,
        password: str | None = None,
        port: int = 22,
    ) -> SshExecutor:
        """Return a new ``SshExecutor`` for the given remote host.

        Useful in non-fixture contexts (e.g. framework health checks or
        prerequisite probes) where fixture injection is not available.

        Args:
            host:     Remote hostname or IP address.
            user:     SSH login name (default: ``$USER``).
            key_path: Path to SSH private key file (``~`` expanded).
            password: SSH password — prefer *key_path* for CI environments.
            port:     SSH server port (default 22).

        Returns:
            A new ``SshExecutor`` whose connection opens lazily on first
            ``run()`` call.
        """
        return SshExecutor(
            host=host,
            user=user,
            key_path=key_path,
            password=password,
            port=port,
        )
