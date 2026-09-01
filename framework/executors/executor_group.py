# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
executor_group.py -- Uniform executor container returned by ``target_executor``.

``NodeExecutorGroup`` wraps a list of executors and provides a uniform API
across all GPU modes so test code never changes based on whether a test uses
one GPU, multiple GPUs on one node, or GPUs across multiple nodes:

    # single-GPU (hw.gpu) — one executor in the group
    def test_hip(target_executor):
        result = target_executor.run("rocm-smi --showid")
        assert result.ok

    # multi-GPU same node (hw.multi_gpu) — one multi-index executor in the group
    @pytest.mark.hw.multi_gpu
    @pytest.mark.gpu_count(2)
    def test_rccl(target_executor):
        result = target_executor.run("python3 allreduce.py")
        assert result.ok

    # multi-node (e2e.multinode) — one executor per node, iterate to dispatch
    @pytest.mark.e2e.multinode
    @pytest.mark.gpu_count(1)
    def test_multinode(target_executor):
        for exec_ in target_executor:
            exec_.run("torchrun --nproc_per_node=1 allreduce.py")

``.run()`` and ``.start_background()`` delegate to the **first** executor in
the group — correct for single-GPU and multi-GPU-same-node modes.  For
multi-node tests, iterate over the group directly (``for exec_ in group:``).
"""

from __future__ import annotations

from collections.abc import Iterator

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import AbstractBackgroundProcess


class NodeExecutorGroup:
    """Uniform container for one or more executors returned by ``target_executor``.

    Always returned by ``target_executor``, ``multi_gpu_fixture``, and
    ``multi_node_fixture``.  The number of executors inside depends on the
    test's markers:

    - ``hw.gpu``       → 1 executor (single GPU, local or remote)
    - ``hw.multi_gpu`` → 1 executor (multiple GPUs via ROCR_VISIBLE_DEVICES)
    - ``e2e.multinode`` → N executors, one per node in the fleet

    For ``--no-gpu`` and ``--container-mode``, a single ``DryRunExecutor`` or
    ``ContainerExecutor`` is wrapped so the group API is unchanged.

    Attributes:
        _executors: Ordered list of executors in this group.
    """

    def __init__(self, executors: list[AbstractExecutor]) -> None:
        """Wrap *executors* in a group.

        Args:
            executors: One or more executor instances.  Must be non-empty.

        Raises:
            ValueError: If *executors* is empty.
        """
        if not executors:
            raise ValueError("NodeExecutorGroup requires at least one executor")
        self._executors = list(executors)

    # ------------------------------------------------------------------
    # Sequence-like access
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[AbstractExecutor]:
        """Iterate over executors in the group.

        Returns:
            Iterator over ``AbstractExecutor`` instances.
        """
        return iter(self._executors)

    def __len__(self) -> int:
        """Return the number of executors in the group.

        Returns:
            Executor count.
        """
        return len(self._executors)

    @property
    def count(self) -> int:
        """Number of executors (nodes) in the group.

        Returns:
            1 for single-GPU / multi-GPU same-node; N for multi-node.
        """
        return len(self._executors)

    @property
    def visible_gpu_count(self) -> int:
        """Number of GPU ordinals exposed to the first executor.

        Multi-GPU same-node tests use a single executor with multiple GPU
        indices injected through ``ROCR_VISIBLE_DEVICES``. This property reports
        that visible GPU count so tests can pass matching ``-g``/``--ngpus``
        values without parsing environment variables or reaching into plugins.

        Returns:
            Number of visible GPUs for the first executor, or ``1`` for
            executors that do not expose GPU-index metadata (DryRun/container).
        """
        executor = self._executors[0]
        gpu_indices = getattr(executor, "gpu_indices", None)
        if gpu_indices:
            return len(gpu_indices)
        gpu_index = getattr(executor, "gpu_index", None)
        if isinstance(gpu_index, list):
            return len(gpu_index)
        if gpu_index is not None:
            return 1
        return 1

    # ------------------------------------------------------------------
    # Executor delegation — forwards to the first executor
    # ------------------------------------------------------------------

    def run(self, command: str, timeout: float | None = None, *, stream: bool = False) -> ExecutionResult:
        """Execute *command* via the first executor in the group.

        Correct for single-GPU and multi-GPU-same-node modes where the group
        holds exactly one executor.  For multi-node tests, iterate and call
        ``exec_.run()`` on each executor individually.

        Args:
            command: Shell command string to execute.
            timeout: Maximum seconds to wait (forwarded to the inner executor).
            stream:  When True, request live output/progress from the inner
                     executor when supported.

        Returns:
            ``ExecutionResult`` from the first executor.
        """
        return self._executors[0].run(command, timeout=timeout, stream=stream)

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
        console_label: str | None = None,
        stream: bool = False,
    ) -> AbstractBackgroundProcess:
        """Start *command* in the background via the first executor.

        See ``AbstractExecutor.start_background()`` for full semantics.

        Args:
            command:  Shell command to launch.
            timeout:  Stop-grace-period (forwarded to the inner executor).
            log_path: If given, subprocess output is appended to this file.
            console_label: Human-readable label for live output attribution.
            stream:   SSH only — emit live output to the ``rocm.test`` logger.

        Returns:
            ``BackgroundProcess`` handle.
        """
        return self._executors[0].start_background(
            command, timeout=timeout, log_path=log_path, console_label=console_label, stream=stream
        )
