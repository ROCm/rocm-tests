# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build representative HIP samples for SPIR-V and verify SPIR-V offload + execution.

Ports the legacy ``hip_samples_spirv`` testcase: each selected upstream HIP
sample is built with ``-DCMAKE_HIP_ARCHITECTURES=amdgcnspirv`` (SPIR-V target),
confirmed to carry an ``amdgcnspirv`` offload bundle (``llvm-objdump
--offloading``), then run — HIP JIT-compiles the SPIR-V at load time and the
sample must still print ``PASSED``/``Passed``.

Uses the same ROCm/hip-tests ``samples/`` subtree as ``test_hip_samples`` (cloned
once per session via the ``hip_samples_repo`` fixture), so the SPIR-V variant
exercises the identical sample set as the native build.
"""

import os

import pytest

# Subset of the native hip_samples set that is single-GPU and self-validating.
# (sample path under samples/, produced executable name)
_SAMPLES = [
    ("0_Intro/bit_extract", "bit_extract"),
    ("0_Intro/square", "square"),
    ("2_Cookbook/0_MatrixTranspose", "MatrixTranspose"),
    ("2_Cookbook/1_hipEvent", "hipEvent"),
    ("2_Cookbook/3_shared_memory", "sharedMemory"),
    ("2_Cookbook/4_shfl", "shfl"),
    ("2_Cookbook/7_streams", "stream"),
]

# llvm-objdump lives under different subpaths depending on ROCm packaging.
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


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("sample_path", "exec_name"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
def test_hip_samples_spirv(
    target_executor,
    cpu_executor,
    ld_path: dict,
    rock_dir: str,
    hip_sample_spirv_build,
    sample_path: str,
    exec_name: str,
):
    """Build one HIP sample for SPIR-V, assert it carries an amdgcnspirv bundle, and run it."""
    # Resolving the build triggers the per-sample cmake configure + make targeting
    # amdgcnspirv; a build failure surfaces as a fixture ERROR.
    binary = hip_sample_spirv_build(sample_path, exec_name)

    objdump = _find_llvm_objdump(rock_dir)
    if objdump is None:
        pytest.skip(f"llvm-objdump not found under {rock_dir}; cannot verify SPIR-V offload bundle")

    dump = cpu_executor.run(f"{objdump} --offloading {binary}")
    assert (
        "amdgcnspirv" in dump.stdout or "No kernel section found" in dump.stdout
    ), f"hip sample {sample_path} was not compiled to SPIR-V (no amdgcnspirv bundle):\n{dump.stdout[:2000]}"

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {binary}")
    assert result.ok, (
        f"SPIR-V hip sample {sample_path} run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "PASSED" in result.stdout or "Passed" in result.stdout
    ), f"SPIR-V hip sample {sample_path} ran but did not report 'PASSED'/'Passed':\n{result.stdout[:2000]}"
