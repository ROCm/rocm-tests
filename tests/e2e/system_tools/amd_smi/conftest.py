# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Fixtures for AMD SMI gtest suite.

Provides: amdsmitst_binary (session-scoped fixture that locates the pre-installed
amdsmitst gtest binary from the ROCm install tree).
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


def _locate_amdsmitst_binary(executor, rock_dir: str) -> str:
    """Locate the amdsmitst gtest binary from rock_dir.

    The amdsmitst binary is pre-installed by the AMD SMI library (amd-smi-lib)
    into the ROCm share directory. We check two canonical locations in order:
        1. <rock_dir>/share/amd_smi/tests/amdsmitst
        2. <rock_dir>/share/amd_smi/amdsmitst

    Args:
        executor: NodeExecutorGroup or compatible executor to run commands on the test node.
        rock_dir: Resolved path to the TheRock/ROCm installation.

    Returns:
        Absolute path to the amdsmitst binary.

    Raises:
        pytest.fail.Exception: When amdsmitst is not found at either location.
    """
    candidates = [
        f"{rock_dir}/share/amd_smi/tests/amdsmitst",
        f"{rock_dir}/share/amd_smi/amdsmitst",
    ]

    for candidate in candidates:
        probe = executor.run(f"test -x {candidate} && echo FOUND")
        if (probe.stdout or "").strip() == "FOUND":
            logger.info("amdsmitst_binary: located at %s", candidate)
            return candidate

    pytest.fail(
        f"amdsmitst gtest binary not found in rock_dir={rock_dir} — "
        f"checked: {', '.join(candidates)}. "
        f"Ensure amd-smi-lib is installed with TheRock."
    )


@pytest.fixture(scope="session")
def amdsmitst_binary(target_executor, rock_dir: str) -> str:
    """Resolve the amdsmitst gtest binary from the ROCm install.

    The amdsmitst binary is a gtest suite that validates the amd-smi library's
    core functionality across all supported AMD GPU types. It exits 0 when all
    test cases pass and prints gtest PASSED/FAILED markers to stdout.

    Args:
        target_executor: NodeExecutorGroup for running commands on the test node.
        rock_dir: Resolved ROCm installation path from builder_plugin.

    Returns:
        Absolute path to the amdsmitst binary on the test node.
    """
    return _locate_amdsmitst_binary(target_executor, rock_dir)
