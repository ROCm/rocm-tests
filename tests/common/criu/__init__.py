# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared CRIU checkpoint/restore machinery.

Generic across ROCm workloads that need CRIU; each workload suite provides its own build and
test functions and reuses this package.

Public surface:
    - ``ensure_criu_runtime`` (host/SSH) / ``ensure_criu_runtime_target`` (inside target_executor)
      / ``CRIU`` — runtime readiness helpers and command prefix.
    - ``criu_dump`` / ``criu_restore`` / ``attach_criu_log`` / ``kill_pid`` — CRIU actions.
    - ``build_and_install`` — build/install CRIU + the amdgpu plugin from a checkout.
"""

from __future__ import annotations

from .fixtures import CRIU, ensure_criu_runtime, ensure_criu_runtime_target
from .installer import build_and_install
from .steps import attach_criu_log, criu_dump, criu_restore, kill_pid

__all__ = [
    "CRIU",
    "attach_criu_log",
    "build_and_install",
    "criu_dump",
    "criu_restore",
    "ensure_criu_runtime",
    "ensure_criu_runtime_target",
    "kill_pid",
]
