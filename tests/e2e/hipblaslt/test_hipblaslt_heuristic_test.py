# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_heuristic_test.py -- hipBLASLt heuristic workspace constraint.

Validates that hipblasLtMatmulAlgoGetHeuristic() returns only algorithms that
respect the specified 4 MB workspace limit (M=2304, N=4096, K=768, 128 candidates).
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

# Fatal library error patterns written to stderr when runtime files cannot load.
_FATAL_STDERR_PATTERNS = [
    "hipModuleLoad failed",
    "rocblaslt error:",
    "Cannot read",
    "Could not load",
]


@pytest.mark.runtime.fast
def test_hipblaslt_heuristic_workspace_constraint(
    target_executor,
    ld_path: dict,
    tensile_lib_path: str,
    hipblaslt_heuristic_test_binary: str,
):
    """Validate hipBLASLt heuristic returns algorithms that respect the workspace limit."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={tensile_lib_path}"
        f" {hipblaslt_heuristic_test_binary}"
    )
    logger.info("hipblaslt_heuristic_test stdout:\n%s", result.stdout)
    if result.stderr:
        logger.warning("hipblaslt_heuristic_test stderr:\n%s", result.stderr)
    assert result.ok, (
        f"hipblaslt_heuristic_test failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASS" in result.stdout, f"hipblaslt_heuristic_test did not report PASS:\n{result.stdout[:2000]}"
    for pat in _FATAL_STDERR_PATTERNS:
        assert pat not in result.stderr, (
            f"Fatal library error detected in stderr (pattern: {pat!r}):\n" f"stderr: {result.stderr[:2000]}"
        )
