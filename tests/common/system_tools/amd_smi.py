# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""amd-smi binary discovery and subcommand availability helpers.

Provides executor-transparent helpers for locating the ``amd-smi`` binary
and verifying that required subcommands are available.  Shared across
``system_tools/amd_smi/`` and ``system_tools/amd_smi/events/`` test suites.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
from typing import Any

import pytest

logger = logging.getLogger(__name__)


def resolve_amd_smi(rock_dir: str, executor=None) -> str | None:
    """Return the ``amd-smi`` binary path or ``None`` when absent.

    Prefers ``<rock_dir>/bin/amd-smi``; falls back to the system PATH.
    Transparent for local and remote executors.

    Args:
        rock_dir:  TheRock/ROCm install root.
        executor:  Remote executor; ``None`` means local.
    """
    if rock_dir:
        candidate = f"{rock_dir}/bin/amd-smi"
        if executor is not None:
            probe = executor.run(f"test -x {shlex.quote(candidate)}", timeout=15.0)
            if probe.ok:
                return candidate
        elif os.access(candidate, os.X_OK):
            return candidate
    if executor is not None:
        which = executor.run("command -v amd-smi", timeout=15.0)
        if which.ok and (which.stdout or "").strip():
            return which.stdout.strip().splitlines()[-1].strip()
        return None
    return shutil.which("amd-smi")


def require_amd_smi(
    rock_dir: str,
    executor,
    subcommand: str | None = None,
) -> str:
    """Fail when ``amd-smi`` is absent or the requested subcommand is unavailable.

    Args:
        rock_dir:    TheRock/ROCm install root.
        executor:    Any executor with a ``.run()`` method.
        subcommand:  Optional subcommand to verify (e.g. ``"metric"``, ``"event"``).

    Returns:
        Absolute path to the ``amd-smi`` binary.
    """
    path = resolve_amd_smi(rock_dir, executor)
    if not path:
        pytest.fail(
            "amd-smi not found under --rock-dir or on PATH — "
            "ensure ROCm is installed and --rock-dir points to the correct path."
        )

    if subcommand is not None:
        check = executor.run(
            f"{shlex.quote(path)} {shlex.quote(subcommand)} --help",
            timeout=30.0,
        )
        if not check.ok:
            pytest.fail(
                f"amd-smi '{subcommand}' subcommand not available — "
                f"upgrade amd-smi to a version that supports it.\n"
                f"stderr: {check.stderr[:500]}"
            )

    return path


def parse_amd_smi_json(data: Any) -> list[dict]:
    """Normalize amd-smi JSON output across wrapper styles.

    amd-smi may wrap per-sample data under ``"gpu_data"``, ``"data"``,
    or return a plain list.  Returns a flat list of per-sample dicts.

    Args:
        data: Parsed JSON value from an amd-smi JSON output field.

    Returns:
        List of per-sample dicts; empty list when ``data`` has no recognised shape.
    """
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for wrapper in ("gpu_data", "data"):
        inner = data.get(wrapper)
        if isinstance(inner, list):
            return [item for item in inner if isinstance(item, dict)]
    return []
