# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.executors -- Command executor hierarchy.

All executors share the ``AbstractExecutor.run(command, timeout)`` contract so
that test code never needs to know which backend is active.

Executor / fixture decision guide
----------------------------------

Choose the fixture that matches the *hw.* + *e2e.* marker combination on
the test.  Never import executor classes directly in test files.

Fixture decision guide:

+---------------------+----------------------------------+--------------------------------+
| Fixture             | Markers                          | Yields / backend               |
+=====================+==================================+================================+
| target_executor     | hw.gpu                           | NodeExecutorGroup(1 executor): |
|                     | hw.multi_gpu + gpu_count(N)      | LocalExecutor or SshGpuExecutor|
|                     | e2e.multinode + gpu_count(N)     | (one executor per node).       |
|                     | --no-gpu / --container-mode      | DryRunExecutor/ContainerExecutor|
+---------------------+----------------------------------+--------------------------------+
| multi_gpu_fixture   | hw.multi_gpu + gpu_count(N)      | NodeExecutorGroup(1 executor): |
|                     | (same-node, explicit; prefer     | N GPUs from ONE node.          |
|                     |  target_executor instead)        |                                |
+---------------------+----------------------------------+--------------------------------+
| multi_node_fixture  | e2e.multinode + gpu_count(N)     | NodeExecutorGroup(N executors):|
|                     | (multi-host, explicit; prefer    | one per node in the fleet.     |
|                     |  target_executor instead)        |                                |
+---------------------+----------------------------------+--------------------------------+
| cpu_executor        | hw.cpu_only                      | CpuExecutor — real subprocess, |
|                     |                                  | no GPU env modifications.      |
+---------------------+----------------------------------+--------------------------------+
| dry_run_executor    | hw.cpu_only + ci.pr              | DryRunExecutor — synthetic     |
|                     |                                  | success, no subprocess.        |
+---------------------+----------------------------------+--------------------------------+
| container_executor  | direct container control         | ContainerExecutor — docker/    |
|                     |                                  | podman with AMD KFD+DRI.       |
+---------------------+----------------------------------+--------------------------------+
| remote_pool         | manual SSH multi-node            | SshExecutor per node.          |
|                     |                                  | Persistent Paramiko session.   |
+---------------------+----------------------------------+--------------------------------+

LocalExecutor operates in three modes:
    Explicit single  (gpu_index=N):       pins ROCR_VISIBLE_DEVICES to "N".
    Explicit multi   (gpu_index=[N,M,...]): pins ROCR_VISIBLE_DEVICES to "N,M,...".
    Ambient          (gpu_index=None):    inherits ROCR_VISIBLE_DEVICES from the
        process environment; raises RuntimeError if neither source is set.

``ExecutorFactory`` backs session_executor and is the single place where
factory executor selection logic lives.
"""

from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import BackgroundProcess, NoOpBackgroundProcess
from framework.executors.container_executor import ContainerExecutor, ContainerStatus
from framework.executors.cpu_executor import CpuExecutor
from framework.executors.dry_run_executor import DryRunExecutor
from framework.executors.executor_factory import ExecutorFactory
from framework.executors.executor_group import NodeExecutorGroup
from framework.executors.labeled_executor import LabeledExecutor
from framework.executors.local_executor import LocalExecutor
from framework.executors.ssh_executor import SshExecutor
from framework.executors.ssh_gpu_executor import SshGpuExecutor

__all__ = [
    "AbstractExecutor",
    "BackgroundProcess",
    "ContainerExecutor",
    "ContainerStatus",
    "CpuExecutor",
    "DryRunExecutor",
    "ExecutorFactory",
    "LabeledExecutor",
    "LocalExecutor",
    "NoOpBackgroundProcess",
    "NodeExecutorGroup",
    "SshExecutor",
    "SshGpuExecutor",
]
