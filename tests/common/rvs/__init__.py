# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared RVS (ROCm Validation Suite) test-suite helpers.

Mirrors the ``tests/common/criu/`` pattern. Import from here in any conftest
that needs to locate, build, or invoke RVS.

Public surface:
    - :func:`find_rvs_binary` — locate a pre-installed binary.
    - :func:`build_rvs_from_source` — clone and CMake-build when absent.
    - :func:`ensure_rvs_runtime` — top-level entry point (prefer installed, build if missing).
    - :func:`run_rvs` — execute one RVS config, return ``(output, exit_code)``.
    - :func:`resolve_conf_dir` — look up the qualification-matrix conf directory for a module.
    - :func:`detect_rvs_device_key` — detect the GPU key used by the matrix.
    - :func:`binary_conf_root` — derive the ``conf`` tree path from the binary location.
    - :func:`build_rvs_env` — construct the ``env VAR=...`` prefix for ``rvs`` invocations.
"""

from __future__ import annotations

from .installer import build_rvs_from_source, ensure_rvs_runtime, find_rvs_binary
from .runner import binary_conf_root, build_rvs_env, detect_rvs_device_key, resolve_conf_dir, run_rvs

__all__ = [
    "binary_conf_root",
    "build_rvs_env",
    "build_rvs_from_source",
    "detect_rvs_device_key",
    "ensure_rvs_runtime",
    "find_rvs_binary",
    "resolve_conf_dir",
    "run_rvs",
]
