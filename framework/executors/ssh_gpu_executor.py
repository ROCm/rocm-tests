# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ssh_gpu_executor.py -- SSH executor with automatic ROCR_VISIBLE_DEVICES injection.

Wraps ``SshExecutor`` and injects ``ROCR_VISIBLE_DEVICES`` into every command
via ``env_overrides`` so that GPU tests targeting a remote node never need to
set the variable themselves — the same guarantee that ``LocalExecutor`` provides
for local runs.

``ROCR_VISIBLE_DEVICES`` is the framework-level standard for GPU isolation.
It operates at the ROCr/HSA layer, which all ROCm runtimes (HIP, HSA, OpenCL)
sit on top of — setting it once is sufficient for all ROCm workloads.

Supports single-GPU (``gpu_indices=[2]``) and multi-GPU
(``gpu_indices=[0, 1, 2, 3]``) assignments transparently.

Usage (via ``NodeSlot.make_executor()`` — not instantiated directly in tests)::

    from framework.executors import SshGpuExecutor, SshExecutor

    ssh = SshExecutor(host="gpu-node-01", user="ci", key_path="~/.ssh/ci_rsa")
    exec_ = SshGpuExecutor(ssh=ssh, gpu_indices=[0, 1])
    result = exec_.run("python3 allreduce.py")  # ROCR_VISIBLE_DEVICES=0,1
    assert result.ok
"""

from __future__ import annotations

import logging

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.ssh_executor import SshExecutor

logger = logging.getLogger(__name__)


class SshGpuExecutor(AbstractExecutor):
    """Remote executor that injects ``ROCR_VISIBLE_DEVICES`` on the remote shell.

    Delegates all command execution to an inner ``SshExecutor`` and prepends
    ``ROCR_VISIBLE_DEVICES=<indices>`` as a shell environment override so the
    remote subprocess sees the correct GPU assignment without requiring SSH
    ``SendEnv`` or remote shell profile edits.

    Attributes:
        ssh:         Underlying ``SshExecutor`` that owns the Paramiko connection.
        gpu_indices: List of GPU ordinals to expose on the remote node.
    """

    def __init__(
        self,
        ssh: SshExecutor,
        gpu_indices: list[int],
    ) -> None:
        """Wrap *ssh* with automatic ``ROCR_VISIBLE_DEVICES`` injection.

        Args:
            ssh:         Active or lazy-connect ``SshExecutor`` for the target node.
            gpu_indices: One or more GPU ordinals (0-based) to assign via
                         ``ROCR_VISIBLE_DEVICES``.  Order is preserved.
        """
        self.ssh = ssh
        self.gpu_indices = list(gpu_indices)

    @property
    def rvd_value(self) -> str:
        """Comma-joined GPU index string for ``ROCR_VISIBLE_DEVICES``.

        Returns:
            E.g. ``"0"`` for a single GPU or ``"2,3"`` for two GPUs.
        """
        return ",".join(str(i) for i in self.gpu_indices)

    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Execute *command* on the remote host with ``ROCR_VISIBLE_DEVICES`` set.

        Delegates to ``SshExecutor.run()`` with ``env_overrides`` carrying
        ``ROCR_VISIBLE_DEVICES``.  The remote shell effectively sees::

            export ROCR_VISIBLE_DEVICES=<indices>; <command>

        Args:
            command: Shell command string to execute on the remote host.
            timeout: Maximum seconds to wait (default 300 s via SshExecutor).

        Returns:
            ``ExecutionResult`` from the remote subprocess.

        Raises:
            RuntimeError: If the SSH connection or authentication fails.
            TimeoutError: If the command exceeds *timeout*.
        """
        logger.debug(
            "SshGpuExecutor[%s RVD=%s] running: %s",
            self.ssh.session_key,
            self.rvd_value,
            command,
        )
        return self.ssh.run(
            command,
            timeout=timeout,
            env_overrides={"ROCR_VISIBLE_DEVICES": self.rvd_value},
        )
