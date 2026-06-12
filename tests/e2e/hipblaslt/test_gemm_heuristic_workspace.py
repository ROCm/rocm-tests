# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_gemm_heuristic_workspace.py -- hipBLASLt workspace-constrained GEMM smoke.

Validates algorithm selection across workspace budget tiers (M=2048).
Binary: tests/e2e/hipblaslt/src/gemm_heuristic_workspace_budget.cpp
Checks: "High Memory Baseline: Pass" in stdout.
"""

import pytest


@pytest.mark.runtime.fast
def test_gemm_heuristic_workspace(
    target_executor,
    ld_path: dict,
    tensile_lib_path: str,
    gemm_heuristic_workspace_budget_binary: str,
):
    """Validate hipBLASLt workspace budget TFLOPS consistency (M=2048 smoke)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={tensile_lib_path}"
        f" {gemm_heuristic_workspace_budget_binary} 2048",
        timeout=600.0,
    )
    assert result.ok, (
        f"gemm_heuristic_workspace failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "High Memory Baseline: Pass" in result.stdout
    ), f"Expected 'High Memory Baseline: Pass' in stdout:\n{result.stdout[:3000]}"
