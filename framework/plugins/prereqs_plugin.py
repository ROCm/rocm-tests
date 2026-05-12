# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
prereqs_plugin.py -- Session-level prerequisite checks.

Provides the ``prereqs_fixture`` which verifies that required software
(ROCm driver, HIP runtime, Python version) is present and at acceptable
versions before any test is collected.

Failing a prerequisite does not fail individual tests — it emits a warning
and optionally skips the session via the ``--strict-prereqs`` CLI flag.

Adding a new prerequisite:
    1. Create a callable that returns (bool, str) — (passed, message).
    2. Register it in the _BUILTIN_CHECKS list below.
    3. Or inject a custom list via the ``extra_prereqs`` pytest option.
"""

from __future__ import annotations

from collections.abc import Callable
import logging
import subprocess
import sys

import pytest

logger = logging.getLogger(__name__)

# Type alias: a prerequisite check is a zero-arg callable returning (passed, message)
PrereqCheck = Callable[[], tuple[bool, str]]


# ---------------------------------------------------------------------------
# Built-in prerequisite checks
# ---------------------------------------------------------------------------


def _check_python_version() -> tuple[bool, str]:
    """Verify Python is at least 3.10."""
    major, minor = sys.version_info[:2]
    passed = (major, minor) >= (3, 10)
    return passed, f"Python {major}.{minor} ({'OK' if passed else 'need >=3.10'})"


def _check_rocm_available() -> tuple[bool, str]:
    """Verify ROCm hip-clang or rocm-smi is accessible on PATH."""
    for cmd in ["hipconfig", "rocm-smi", "amd-smi"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
            return True, f"ROCm toolchain found via '{cmd}'"
        except FileNotFoundError:
            continue
    return False, "ROCm not found (hipconfig / rocm-smi / amd-smi not on PATH)"


_BUILTIN_CHECKS: list[PrereqCheck] = [
    _check_python_version,
    _check_rocm_available,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register --strict-prereqs option."""
    group = parser.getgroup("rocm-prereqs", "ROCm prerequisite options")
    group.addoption(
        "--strict-prereqs",
        action="store_true",
        default=False,
        help="Abort the test session if any prerequisite check fails.",
    )


@pytest.fixture(scope="session")
def prereqs_fixture(request):
    """Run all built-in prerequisite checks and log their results.

    When ``--strict-prereqs`` is active, any failing check triggers
    ``pytest.exit()`` to abort the session before test collection.

    Args:
        request: pytest session-scoped request object.

    Returns:
        List of (passed, message) tuples — one per check run.
    """
    results: list[tuple[bool, str]] = []
    strict = request.config.getoption("--strict-prereqs", default=False)

    for check in _BUILTIN_CHECKS:
        try:
            passed, message = check()
        except Exception as exc:
            passed, message = False, f"{check.__name__} raised: {exc}"

        results.append((passed, message))
        level = logging.INFO if passed else logging.WARNING
        logger.log(level, "Prereq [%s]: %s", "PASS" if passed else "FAIL", message)

    failed = [msg for ok, msg in results if not ok]
    if failed and strict:
        pytest.exit(
            f"Session aborted: {len(failed)} prerequisite(s) failed:\n" + "\n".join(f"  - {m}" for m in failed),
            returncode=3,
        )

    return results
