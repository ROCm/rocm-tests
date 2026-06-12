# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_heuristic_workspace.py -- hipBLASLt heuristic workspace constraint.

Verifies hipblasLtMatmulAlgoGetHeuristic respects a 4 MB workspace budget
(M=2304, N=4096, K=768, 128 candidates).
Binary: tests/e2e/hipblaslt/src/hipblaslt_heuristic_workspace/hipblaslt_heuristic_workspace.hip
Checks: no "FAIL" in stdout, no fatal library errors in stderr.
"""

from __future__ import annotations

import pytest

# Fatal error patterns the hipBLASLt / rocBLAS-lt runtime writes to stderr when
# library files are missing or cannot be loaded.  Any of these in stderr indicates
# a real runtime failure even when the process exits 0.
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
    hipblaslt_heuristic_workspace_binary: str,
):
    """Validate hipBLASLt heuristic workspace constraint is respected."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={tensile_lib_path}"
        f" {hipblaslt_heuristic_workspace_binary}"
    )
    assert result.ok, (
        f"hipblaslt_heuristic_workspace failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "FAIL" not in result.stdout, f"Workspace constraint violation detected:\n{result.stdout[:2000]}"
    for pat in _FATAL_STDERR_PATTERNS:
        assert pat not in result.stderr, (
            f"Fatal library error detected in stderr (pattern: {pat!r}):\n" f"stderr: {result.stderr[:2000]}"
        )
