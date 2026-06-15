# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
hip.py -- HIP runtime and ROCm version helpers.

Normalises flag changes across ROCm versions so test code never encodes
``hipconfig`` flag history.  Also provides ``require_rocm_version()`` for
lightweight ``pytest.skip()`` gating — tests skip cleanly instead of failing
with obscure CLI errors.

Known flag changes handled:

    ``hipconfig --version``  →  ``hipconfig -v``          (pre-5.0 fallback)
    ``hipconfig --rocmpath`` →  ``hipconfig --path``       (ROCm 6.x rename)

Usage::

    from framework.rocm.libs.hip import hip_version, rocm_path, require_rocm_version
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)


def hip_version(executor: AbstractExecutor) -> str | None:
    """Return the HIP runtime version string from ``hipconfig``.

    Tries flags in the order they existed across ROCm versions:
    ``--version`` (current), ``-v`` (legacy alias), ``--full`` (verbose).

    Args:
        executor: Any executor with a ``.run(command)`` method.

    Returns:
        Version string (e.g. ``"60300000"``), or None if unavailable.
    """
    for flag in ("--version", "-v", "--full"):
        result = executor.run(f"hipconfig {flag} 2>/dev/null")
        if result.ok and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    return None


def rocm_path(executor: AbstractExecutor) -> str | None:
    """Return the ROCm installation path from ``hipconfig``.

    Handles the ``--rocmpath`` → ``--path`` flag rename introduced in ROCm 6.x.

    Args:
        executor: Any executor with a ``.run(command)`` method.

    Returns:
        Path string (e.g. ``"/opt/rocm"``), or None if unavailable.
    """
    for flag in ("--rocmpath", "--path"):
        result = executor.run(f"hipconfig {flag} 2>/dev/null")
        if result.ok and result.stdout.strip():
            return result.stdout.strip()
    return None


def get_device_count(executor: AbstractExecutor) -> int:
    """Return the number of AMD GPU devices visible to the executor.

    Uses a brief Python snippet executed via the executor so it works in local,
    SSH, and container contexts without any extra tooling on the runner.

    Args:
        executor: Any executor with a ``.run(command)`` method.

    Returns:
        Device count (0 if HIP is unavailable or no GPUs found).
    """
    script = (
        "import ctypes, ctypes.util, json; "
        "lib = ctypes.util.find_library('amdhip64'); "
        "hip = ctypes.CDLL(lib) if lib else None; "
        "c = ctypes.c_int(0); "
        "hip.hipGetDeviceCount(ctypes.byref(c)) if hip else None; "
        "print(c.value)"
    )
    result = executor.run(f"python3 -c {script!r}")
    if result.ok and result.stdout.strip().isdigit():
        return int(result.stdout.strip())
    return 0


def get_device_arch(executor: AbstractExecutor, _device: int = 0) -> str:
    """Return the GFX architecture string for *_device*.

    Args:
        executor: Any executor with a ``.run()`` method.
        _device:  GPU ordinal (reserved for future use; current implementation
                  queries the active platform context).

    Returns:
        Architecture string (e.g. ``"gfx942"``), or ``"unknown"`` if unavailable.
    """
    script = (
        "import subprocess, re; "
        "r = subprocess.run(['hipconfig', '--current-platform'], capture_output=True, text=True); "
        "m = re.search(r'gfx\\\\d+[a-z]?', r.stdout + r.stderr); "
        "print(m.group(0) if m else 'unknown')"
    )
    result = executor.run(f"python3 -c {script!r}")
    if result.ok and result.stdout.strip():
        return result.stdout.strip().splitlines()[-1]
    return "unknown"


def require_rocm_version(executor: AbstractExecutor, major: int, minor: int = 0) -> None:
    """Fail the current test if the installed ROCm version is below ``major.minor``.

    Reads ``/opt/rocm/.info/version`` first (most reliable), then falls back to
    ``hipconfig --version`` parsing.

    Args:
        executor: Any executor with a ``.run()`` method.
        major:    Minimum required major version.
        minor:    Minimum required minor version (default 0).

    Raises:
        pytest.fail.Exception: When the installed version is below the requirement
                               or cannot be detected — missing ROCm is a prerequisite
                               failure, not a resource shortage.
    """
    import pytest  # pylint: disable=import-outside-toplevel

    version_str = _detect_rocm_version(executor)
    if version_str is None:
        pytest.fail(f"ROCm version not detectable — cannot assert >= {major}.{minor}")

    parts = version_str.split(".")
    try:
        installed = (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        pytest.fail(f"Could not parse ROCm version: {version_str!r}")

    if installed < (major, minor):
        pytest.fail(f"ROCm {major}.{minor}+ required; installed {version_str}")


def _detect_rocm_version(executor: AbstractExecutor) -> str | None:
    """Detect the installed ROCm release version string.

    Tries ``/opt/rocm/.info/version`` first, then ``rocminfo``, then
    ``hipconfig``.

    Args:
        executor: Any executor with a ``.run()`` method.

    Returns:
        Version string (e.g. ``"6.3.0"``), or None if not detectable.
    """
    # Version file (most reliable, present in ROCm >= 5.0)
    result = executor.run("cat /opt/rocm/.info/version 2>/dev/null")
    if result.ok and result.stdout.strip():
        return result.stdout.strip()

    # rocminfo header
    result = executor.run("rocminfo 2>/dev/null | grep -i 'ROCm Runtime Version'")
    if result.ok and result.stdout.strip():
        m = re.search(r"(\d+\.\d+[\.\d]*)", result.stdout)
        if m:
            return m.group(1)

    # hipconfig fallback
    result = executor.run("hipconfig --version 2>/dev/null")
    if result.ok and result.stdout.strip():
        m = re.search(r"(\d+\.\d+[\.\d]*)", result.stdout)
        if m:
            return m.group(1)

    return None
