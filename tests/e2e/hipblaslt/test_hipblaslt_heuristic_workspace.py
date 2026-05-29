# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_heuristic_workspace.py -- hipBLASLt workspace constraint heuristic validation.

Verifies that hipblasLtMatmulAlgoGetHeuristic respects the workspace size limit
passed by the caller. Queries 128 algorithm candidates for M=2304, N=4096, K=768
with a 4 MB workspace budget and validates no candidate returns a workspace
requirement exceeding that budget. Catches the class of defect where the heuristic
over-allocates workspace.

Binary compiled via CMake from:
    tests/e2e/hipblaslt/src/hipblaslt_heuristic_workspace/hipblaslt_heuristic_workspace.hip

This test exercises the hipBLASLt heuristic API integration path (not a full
GEMM workload), so e2e.stack is inherited from the directory profile — the
auto-injected profile is retained; override only if reclassifying this test.

runtime.fast is declared explicitly.
"""

from __future__ import annotations

import pathlib

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
    arch_lib_path,
    hipblaslt_heuristic_workspace_binary: str,
    rock_dir: str,
    gpu_arch: str | None,
):
    """Validate hipBLASLt heuristic workspace constraint is respected."""
    ld = ld_path["LD_LIBRARY_PATH"]
    library_base = pathlib.Path(rock_dir) / "lib" / "hipblaslt" / "library"
    # hipBLASLt lays out its tensile library files under a per-arch subdirectory:
    #   <rock_dir>/lib/hipblaslt/library/<arch>/TensileLibrary_lazy_<arch>.dat
    #   <rock_dir>/lib/hipblaslt/library/<arch>/Kernels.so-000-<arch>.hsaco
    # arch_lib_path() appends /<arch> when --gpu-arch is supplied, or returns
    # the base path unchanged when the option is absent.
    tensile_lib = arch_lib_path(library_base)

    # Assertive preflight: Tensile kernel blobs must be present before running.
    # A missing file is a prerequisite failure — fail loudly so CI surfaces the
    # environment defect rather than an opaque error from inside the binary.
    if gpu_arch:
        tensile_dat = pathlib.Path(tensile_lib) / f"TensileLibrary_lazy_{gpu_arch}.dat"
        if not tensile_dat.exists():
            pytest.fail(
                f"hipBLASLt Tensile kernels missing for arch {gpu_arch!r}: {tensile_dat}\n"
                "Install the BLAS artifact package (pass --blas to install_rocm_from_artifacts.py)."
            )

    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={tensile_lib}"
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
