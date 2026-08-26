# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Privileged command execution helper (replaces shell `run_priv`)."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Sequence


def is_root() -> bool:
    return os.geteuid() == 0


def run_priv(cmd: Sequence, **kwargs) -> subprocess.CompletedProcess:
    """Run `cmd` as root (or with sudo -n if available).

    If already root: run directly.
    Else if sudo is available: prepend `sudo -n`.
    Otherwise: run as-is (may still work if binary has suid or doesn't need root).
    """
    cmd = [str(c) for c in cmd]
    if is_root():
        return subprocess.run(cmd, **kwargs)
    if shutil.which("sudo"):
        return subprocess.run(["sudo", "-n"] + cmd, **kwargs)
    return subprocess.run(cmd, **kwargs)


def run_priv_silent(cmd: Sequence) -> int:
    """Run privileged command, swallow output, return exit code."""
    try:
        proc = run_priv(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return proc.returncode
    except Exception:
        return 1
