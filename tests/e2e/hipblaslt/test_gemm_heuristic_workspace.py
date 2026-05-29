# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_gemm_heuristic_workspace.py -- hipBLASLt workspace-constrained algorithm selection.

Sweeps workspace memory budgets from 512 MB down to 1 MB, queries the hipBLASLt
heuristic for a compliant algorithm at each budget, allocates workspace, runs
warmup + timed GEMM, and measures TFLOPS consistency (CV <= 3% within tier).
Validates that at least one budget >= 32 MB produces a working algorithm.

Binary compiled from:
    tests/e2e/hipblaslt/src/gemm_heuristic_workspace_budget.cpp

Smoke args (< 2 min): M=2048 -> A[2048x2048] x B[2048x4096]

runtime.fast is declared explicitly.
"""

import pathlib

import pytest


@pytest.mark.runtime.fast
def test_gemm_heuristic_workspace(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    gpu_arch: str | None,
    arch_lib_path,
    gemm_heuristic_workspace_budget_binary: str,
):
    """Validate hipBLASLt workspace budget TFLOPS consistency (M=2048 smoke)."""
    library_base = pathlib.Path(rock_dir) / "lib" / "hipblaslt" / "library"
    # Tensile kernels live under a per-arch subdirectory:
    #   <rock_dir>/lib/hipblaslt/library/<arch>/TensileLibrary_lazy_<arch>.dat
    tensile_lib = arch_lib_path(library_base)
    if gpu_arch:
        tensile_dat = pathlib.Path(tensile_lib) / f"TensileLibrary_lazy_{gpu_arch}.dat"
        if not tensile_dat.exists():
            pytest.fail(
                f"hipBLASLt Tensile kernels missing for arch {gpu_arch!r}: {tensile_dat}\n"
                "Install the BLAS artifact package (pass --blas to install_rocm_from_artifacts.py)."
            )

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={tensile_lib}"
        f" {gemm_heuristic_workspace_budget_binary} 2048"
    )
    assert result.ok, (
        f"gemm_heuristic_workspace failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "High Memory Baseline: Pass" in result.stdout
    ), f"Expected 'High Memory Baseline: Pass' in stdout:\n{result.stdout[:3000]}"
