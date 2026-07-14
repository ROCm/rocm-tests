# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared helpers for SPIR-V offload-bundle validation."""

from __future__ import annotations

import os
import shlex

import pytest

# llvm-objdump lives under different subpaths depending on the ROCm packaging.
_OBJDUMP_CANDIDATES = (
    ("lib", "llvm", "bin", "llvm-objdump"),  # TheRock
    ("llvm", "bin", "llvm-objdump"),  # standard ROCm
    ("bin", "llvm-objdump"),  # packaging variants
)


def assert_spirv_offload_bundle(target_executor, rock_dir: str, binary: str, label: str) -> None:
    """Assert that *binary* carries an amdgcnspirv offload bundle.

    The lookup and objdump invocation run through ``target_executor`` instead of
    local filesystem checks, so the helper works for local, container, and SSH
    execution backends.
    """
    candidates = [os.path.join(rock_dir, *parts) for parts in _OBJDUMP_CANDIDATES]
    candidate_args = " ".join(shlex.quote(path) for path in candidates)
    find_result = target_executor.run(
        f'for p in {candidate_args}; do [ -x "$p" ] && printf "%s" "$p" && exit 0; done; exit 1'
    )
    if not find_result.ok or not find_result.stdout.strip():
        pytest.skip(f"llvm-objdump not found under {rock_dir}; cannot verify SPIR-V offload bundle")

    objdump = shlex.quote(find_result.stdout.strip().splitlines()[-1])
    dump = target_executor.run(f"{objdump} --offloading {shlex.quote(binary)}")
    assert dump.ok, (
        f"llvm-objdump failed for {label} (exit={dump.exit_code}):\n"
        f"stdout: {dump.stdout[:2000]}\nstderr: {dump.stderr[:500]}"
    )
    assert (
        "amdgcnspirv" in dump.stdout
    ), f"{label} was not compiled to SPIR-V (no amdgcnspirv bundle):\n{dump.stdout[:2000]}"
