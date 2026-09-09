# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_mini_residual_app.py -- hipBLASLt GEMM beta-path zero-invariant validation.

Description:
    System-level test for hipBLASLt GEMM on AMD GPUs. Validates the full stack
    (application -> hipBLASLt -> HIP -> ROCm driver -> GPU) using FP8/BF8/BF16
    matrix multiply and the beta-path invariant: when A=0 and C=0, output D must
    be zero and finite.

    The binary auto-selects FP8 (E4M3/E5M2) when the local hipBLASLt heuristic
    supports it; otherwise falls back to BF16. All heuristic candidates are
    validated before the best one is selected for the training loop.

    Supported GPU architectures: gfx942, gfx950.

Binary: tests/e2e/hipblaslt/mini_residual_app/src/mini_residual_app.cpp
Exit codes:
    0  = all iterations passed
    1  = HIP runtime error
    2  = hipBLASLt API error
    3  = CLI argument error
    5  = no heuristic algorithm found
    11 = GEMM or validation failed during training loop
    12 = no candidate algorithm passed initial validation

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hipblaslt/:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.medium  (1000-iteration run at LLM-like sizes)
    runtime.fast    (small-shape smoke run)
"""

from __future__ import annotations

import pathlib

import pytest

# Architectures where hipBLASLt has no FP8 GEMM kernels.
# Listed as unsupported prefixes so new architectures run by default.
_NO_FP8_ARCH_PREFIXES = ("gfx90", "gfx100", "gfx101", "gfx103", "gfx110", "gfx115")

# Fatal library error patterns that indicate a real failure even when exit code is 0.
_FATAL_STDERR_PATTERNS = [
    "hipModuleLoad failed",
    "rocblaslt error:",
    "Cannot read",
    "Could not load",
]


def _resolve_tensile_lib_path(tensile_lib_path: str, rock_dir: str) -> str:
    """Resolve hipBLASLt Tensile library path, checking arch-specific subdirectories.

    hipBLASLt organizes kernel libraries in GPU architecture subdirectories
    (e.g., gfx942/). If the base tensile_lib_path is a directory with no loose
    kernel files, search for arch-specific subdirectories.

    Args:
        tensile_lib_path: Path to hipblaslt/library/ (from builder_plugin)
        rock_dir:         Path to ROCm/TheRock install root.

    Returns:
        Resolved path: either tensile_lib_path or an arch-specific subdirectory.
    """
    base = pathlib.Path(tensile_lib_path)
    if not base.exists():
        return tensile_lib_path

    has_loose_libs = any(base.glob("*.dat*")) or any(base.glob("Kernels.so*"))
    if has_loose_libs:
        return tensile_lib_path

    arch_dirs = sorted(base.glob("gfx*"))
    if arch_dirs:
        return str(arch_dirs[0])

    return tensile_lib_path


def _check_fatal_stderr(result, label: str) -> None:
    """Assert no fatal hipBLASLt library error patterns appear in stderr.

    Args:
        result: ``ExecutionResult`` from ``target_executor.run()``.
        label:  Short test identifier for the assertion message.
    """
    for pat in _FATAL_STDERR_PATTERNS:
        assert pat not in result.stderr, (
            f"{label}: fatal library error in stderr (pattern: {pat!r}):\n" f"stderr: {result.stderr[:2000]}"
        )


@pytest.mark.runtime.medium
def test_mini_residual_app_default(
    target_executor,
    ld_path: dict,
    tensile_lib_path: str,
    mini_residual_app_binary: str,
    gpu_arch: str | None,
    rock_dir: str,
):
    """LLM-like shape (M=8192, N=32768, K=1024, 1000 iterations).

    Runs the full 1000-iteration training loop at production GEMM sizes.
    Validates the beta-path invariant (A=0, C=0 -> D=0) on every iteration
    using the best heuristic algorithm selected from up to 32 candidates.
    FP8 (E4M3/E5M2) is used when supported; BF16 otherwise.

    Args:
        target_executor:         GPU executor from ``remote_node_plugin``.
        ld_path:                 LD_LIBRARY_PATH dict from ``builder_plugin``.
        tensile_lib_path:        hipBLASLt Tensile kernel directory path.
        mini_residual_app_binary: Compiled binary path from this conftest.
        gpu_arch:                Target GPU architecture string, or ``None``.
        rock_dir:                Resolved ROCm install path.
    """
    if gpu_arch and gpu_arch.startswith(_NO_FP8_ARCH_PREFIXES):
        pytest.skip(f"hipBLASLt has no FP8 GEMM kernels on {gpu_arch}; " "mini_residual_app FP8 path does not apply")
    ld = ld_path["LD_LIBRARY_PATH"]
    lib_path = _resolve_tensile_lib_path(tensile_lib_path, rock_dir)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}" f" HIPBLASLT_TENSILE_LIBPATH={lib_path}" f" {mini_residual_app_binary}",
        timeout=1800.0,
    )
    assert result.ok, (
        f"mini_residual_app (default) failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, (
        f"mini_residual_app (default): '[PASS]' not found in stdout:\n" f"{result.stdout[:2000]}"
    )
    _check_fatal_stderr(result, "mini_residual_app_default")


@pytest.mark.runtime.fast
def test_mini_residual_app_smoke(
    target_executor,
    ld_path: dict,
    tensile_lib_path: str,
    mini_residual_app_binary: str,
    gpu_arch: str | None,
    rock_dir: str,
):
    """Small-shape smoke run (M=1024, N=1024, K=512, 10 iterations).

    Fast-feedback check that the binary initialises, allocates buffers, selects
    a heuristic algorithm, and completes 10 iterations successfully. Suitable for
    ci.nightly pre-flight verification before the heavier test runs.

    Args:
        target_executor:         GPU executor from ``remote_node_plugin``.
        ld_path:                 LD_LIBRARY_PATH dict from ``builder_plugin``.
        tensile_lib_path:        hipBLASLt Tensile kernel directory path.
        mini_residual_app_binary: Compiled binary path from this conftest.
        gpu_arch:                Target GPU architecture string, or ``None``.
        rock_dir:                Resolved ROCm install path.
    """
    if gpu_arch and gpu_arch.startswith(_NO_FP8_ARCH_PREFIXES):
        pytest.skip(f"hipBLASLt has no FP8 GEMM kernels on {gpu_arch}; " "mini_residual_app FP8 path does not apply")
    ld = ld_path["LD_LIBRARY_PATH"]
    lib_path = _resolve_tensile_lib_path(tensile_lib_path, rock_dir)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={lib_path}"
        f" {mini_residual_app_binary}"
        " --M 1024 --N 1024 --K 512 --iters 10",
        timeout=300.0,
    )
    assert result.ok, (
        f"mini_residual_app (smoke) failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, (
        f"mini_residual_app (smoke): '[PASS]' not found in stdout:\n" f"{result.stdout[:2000]}"
    )
    _check_fatal_stderr(result, "mini_residual_app_smoke")


@pytest.mark.runtime.medium
def test_mini_residual_app_two_phase(
    target_executor,
    ld_path: dict,
    tensile_lib_path: str,
    mini_residual_app_binary: str,
    gpu_arch: str | None,
    rock_dir: str,
):
    """Two-phase training loop (M=2048, N=2048, K=1024, 600 iters, 100 nonzero).

    Exercises the two-phase training loop: the first 100 iterations use a
    non-zero A and a randomly initialised C (non-zero phase), then iterations
    101-600 zero both A and C and verify D=0 (zero phase). This validates that
    the beta-path invariant holds even after repeated non-zero GEMM computations
    on the same heuristic algorithm instance.

    Args:
        target_executor:         GPU executor from ``remote_node_plugin``.
        ld_path:                 LD_LIBRARY_PATH dict from ``builder_plugin``.
        tensile_lib_path:        hipBLASLt Tensile kernel directory path.
        mini_residual_app_binary: Compiled binary path from this conftest.
        gpu_arch:                Target GPU architecture string, or ``None``.
        rock_dir:                Resolved ROCm install path.
    """
    if gpu_arch and gpu_arch.startswith(_NO_FP8_ARCH_PREFIXES):
        pytest.skip(f"hipBLASLt has no FP8 GEMM kernels on {gpu_arch}; " "mini_residual_app FP8 path does not apply")
    ld = ld_path["LD_LIBRARY_PATH"]
    lib_path = _resolve_tensile_lib_path(tensile_lib_path, rock_dir)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" HIPBLASLT_TENSILE_LIBPATH={lib_path}"
        f" {mini_residual_app_binary}"
        " --M 2048 --N 2048 --K 1024 --iters 600 --nonzero_iters 100",
        timeout=1200.0,
    )
    assert result.ok, (
        f"mini_residual_app (two-phase) failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "[PASS]" in result.stdout, (
        f"mini_residual_app (two-phase): '[PASS]' not found in stdout:\n" f"{result.stdout[:2000]}"
    )
    _check_fatal_stderr(result, "mini_residual_app_two_phase")
