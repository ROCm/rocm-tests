# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build a basic HIP app for SPIR-V and verify the SPIR-V offload bundle + execution.

Ports the legacy ``hip_basic_spirv`` testcase: compile a HIP program with
``--offload-arch=amdgcnspirv``, confirm the produced binary carries an
``amdgcnspirv`` offload bundle (via ``llvm-objdump --offloading``), then run it
and check it still prints its success marker (HIP JIT-compiles the SPIR-V at
load time).
"""

import os

import pytest

# llvm-objdump lives under different subpaths depending on the ROCm packaging.
_OBJDUMP_CANDIDATES = (
    ("lib", "llvm", "bin", "llvm-objdump"),  # TheRock
    ("llvm", "bin", "llvm-objdump"),  # standard ROCm
    ("bin", "llvm-objdump"),  # packaging variants
)


def _find_llvm_objdump(rock_dir: str) -> str | None:
    """Return the first existing llvm-objdump path under *rock_dir*, or None."""
    for parts in _OBJDUMP_CANDIDATES:
        candidate = os.path.join(rock_dir, *parts)
        if os.path.isfile(candidate):
            return candidate
    return None


@pytest.mark.runtime.fast
def test_hip_basic_spirv(target_executor, cpu_executor, ld_path: dict, rock_dir: str, hip_basic_spirv_binary: str):
    """Verify a SPIR-V-targeted HIP app emits a SPIR-V bundle and runs correctly."""
    binary = hip_basic_spirv_binary

    objdump = _find_llvm_objdump(rock_dir)
    if objdump is None:
        pytest.skip(f"llvm-objdump not found under {rock_dir}; cannot verify SPIR-V offload bundle")

    # A SPIR-V build must carry an 'amdgcnspirv' offload bundle rather than a
    # native gfx code object. 'No kernel section found' is accepted for the
    # rare host-only object, mirroring the legacy check.
    dump = cpu_executor.run(f"{objdump} --offloading {binary}")
    assert (
        "amdgcnspirv" in dump.stdout or "No kernel section found" in dump.stdout
    ), f"binary {binary} was not compiled to SPIR-V (no amdgcnspirv bundle):\n{dump.stdout[:2000]}"

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary}")
    assert result.ok, (
        f"SPIR-V HIP app run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "PASSED" in result.stdout or "Passed" in result.stdout
    ), f"SPIR-V HIP app ran but did not report success:\n{result.stdout[:2000]}"
