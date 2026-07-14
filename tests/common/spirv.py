# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""SPIR-V offload-bundle validation helpers."""

from __future__ import annotations

import os
import shlex

import pytest

_OBJDUMP_CANDIDATES = (
    ("lib", "llvm", "bin", "llvm-objdump"),
    ("llvm", "bin", "llvm-objdump"),
    ("bin", "llvm-objdump"),
)


def assert_spirv_offload_bundle(target_executor, rock_dir: str, binary: str, label: str) -> None:
    """Assert that *binary* carries an amdgcnspirv offload bundle."""
    candidates = [os.path.join(rock_dir, *parts) for parts in _OBJDUMP_CANDIDATES]
    find_cmd = "for p in " + " ".join(shlex.quote(path) for path in candidates)
    find_cmd += '; do [ -x "$p" ] && printf "%s" "$p" && exit 0; done; exit 1'
    find_result = target_executor.run(find_cmd)
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
