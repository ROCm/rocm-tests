# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.common -- Shared utilities accessible by both framework and test code.

Public API (import from here, not from sub-modules)::

    from framework.common import ExecutionResult, Outcome, executor_log_path, gpu_monitor_log_path
"""

from framework.common.helpers import (
    ExecutionResult,
    Outcome,
    executor_log_path,
    gpu_monitor_log_path,
)

__all__ = ["ExecutionResult", "Outcome", "executor_log_path", "gpu_monitor_log_path"]
