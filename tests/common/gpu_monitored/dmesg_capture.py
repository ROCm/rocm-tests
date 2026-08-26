# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared dmesg capture helpers for orchestrator and pretest health probe."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.cpu_executor import CpuExecutor

from tests.common.gpu_monitored.privilege import run_priv

DMESG_SNAPSHOT_UNAVAILABLE = "# [dmesg-capture] snapshot unavailable\n"


def dmesg_result(res) -> tuple[bool, str]:
    """Return ``(available, output)`` for a dmesg subprocess result.

    Some kernels emit a permission warning to stderr and exit non-zero while
    still printing a complete ring buffer on stdout. Treat empty stdout (or
    stdout that is only the permission diagnostic) as capture failure.
    """
    output = res.stdout or ""
    rc = getattr(res, "returncode", getattr(res, "exit_code", 1))
    if rc == 0:
        return True, output
    diagnostic_only = bool(re.search(
        r"(?:read kernel buffer failed|Operation not permitted|"
        r"Permission denied)",
        output,
        re.IGNORECASE,
    ))
    return bool(output and not diagnostic_only), (
        output if output and not diagnostic_only else ""
    )


def capture_dmesg_text(cpu_executor: CpuExecutor | None = None) -> tuple[bool, str]:
    """Read the kernel ring buffer, trying framework, local, then privileged paths."""
    if cpu_executor is not None:
        try:
            res = cpu_executor.run("dmesg -T")
            available, output = dmesg_result(res)
            if available:
                return available, output
        except Exception:
            pass
    try:
        res = subprocess.run(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=15,
        )
        available, output = dmesg_result(res)
        if available:
            return available, output
    except Exception:
        pass
    try:
        proc = run_priv(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=15,
        )
        return dmesg_result(proc)
    except Exception:
        return False, ""
