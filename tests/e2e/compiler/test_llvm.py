# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_llvm.py -- LLVM memory-intrinsic stress test for AMD GPU.

Validates that the LLVM lowering of ``__builtin_memset``, ``__builtin_memcpy``,
and ``__builtin_memmove`` produces correct results under multi-stream,
large-buffer conditions on real AMD GPU hardware.

The test binary is compiled from:
    tests/e2e/compiler/src/llvm_memIntrinsic_stress.cpp

Compiled binary is cached at:
    tests/e2e/compiler/build/llvm_mem_intrinsic_stress

Prerequisites:
    - ``--rock-dir`` CLI flag or ``ROCK_DIR`` env var pointing to a TheRock/ROCm
      installation that provides ``bin/hipcc`` and ``lib/``.
    - Real AMD GPU hardware.

Explicit marker (not in profile):
    runtime.medium  -- Expected duration < 30 minutes.
"""

import pytest


@pytest.mark.runtime.medium
def test_llvm_mem_intrinsic_stress(
    target_executor,
    ld_path: dict,
    llvm_mem_intrinsic_stress_binary: str,
):
    """Run the LLVM memory-intrinsic stress binary on an AMD GPU.

    Exercises multi-stream large-buffer memset/memcpy/memmove kernels and
    validates results against a host-side golden replay.  The binary exits
    non-zero on any mismatch or HIP API error.
    """
    ld_library_path = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld_library_path} {llvm_mem_intrinsic_stress_binary}")
    assert result.ok, (
        f"llvm_mem_intrinsic_stress failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\n"
        f"stderr: {result.stderr[:500]}"
    )
