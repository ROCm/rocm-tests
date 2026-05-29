# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_mini_residual_app.py -- hipBLASLt GEMM beta-path zero invariant validation.

Validates that when input matrix A=0 and C=0, the output D is exactly zero
and finite for all selected heuristic algorithms. Exercises the full hipBLASLt
stack: heuristic selection → workspace allocation → FP8/BF16 kernel → device-side
NaN/Inf scan → host validation.

Binary compiled from:
    tests/e2e/hipblaslt/src/mini_residual_app.cpp

Smoke args (< 10 s): --M 512 --N 512 --K 256 --iters 5

runtime.fast is declared explicitly.
"""

import pathlib

import pytest


@pytest.mark.runtime.fast
def test_mini_residual_app(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    gpu_arch: str | None,
    arch_lib_path,
    mini_residual_app_binary: str,
):
    """Validate hipBLASLt GEMM beta-path zero invariant (FP8/BF16)."""
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
        f" {mini_residual_app_binary}"
        " --M 512 --N 512 --K 256 --iters 5"
    )
    assert result.ok, (
        f"mini_residual_app failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, f"Expected '[PASS]' in stdout:\n{result.stdout[:2000]}"
