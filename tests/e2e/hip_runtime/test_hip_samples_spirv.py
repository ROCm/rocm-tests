# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the representative HIP samples for SPIR-V and verify bundle + execution.

Reuses the exact single-GPU sample list from ``test_hip_samples`` (same
ROCm/hip-tests sources); the only differences are the SPIR-V build fixture
(``-DCMAKE_HIP_ARCHITECTURES=amdgcnspirv``) and the added amdgcnspirv
offload-bundle assertion. Multi-GPU / toolchain-special / arch-gated samples are
intentionally left out of this representative set (mirrors ``test_hip_samples``);
the multi-GPU case is covered separately below.
"""

import pytest

from tests.common.spirv import assert_spirv_offload_bundle
from tests.e2e.hip_runtime.test_hip_samples import _SAMPLES


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("sample_path", "exec_name"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
def test_hip_samples_spirv(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    hip_sample_spirv_build,
    sample_path: str,
    exec_name: str,
):
    """Build and run one HIP sample targeting SPIR-V."""
    binary = hip_sample_spirv_build(sample_path, exec_name)

    assert_spirv_offload_bundle(target_executor, rock_dir, binary, f"hip sample {sample_path}")

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {binary}")
    assert result.ok, (
        f"SPIR-V hip sample {sample_path} run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "PASSED" in result.stdout or "Passed" in result.stdout
    ), f"SPIR-V hip sample {sample_path} ran but did not report PASSED/Passed:\n{result.stdout[:2000]}"


# Multi-GPU HIP sample: marked hw.multi_gpu + gpu_count(2) so the framework
# auto-skips where fewer than 2 GPUs are available and runs across the allocated
# GPUs otherwise (mirrors the multi-GPU exclusion in the single-GPU set above).
_MULTI_GPU_SAMPLES = [
    ("2_Cookbook/8_peer2peer", "peer2peer"),
]


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
@pytest.mark.parametrize(("sample_path", "exec_name"), _MULTI_GPU_SAMPLES, ids=[s[0] for s in _MULTI_GPU_SAMPLES])
def test_hip_samples_spirv_multi_gpu(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    hip_sample_spirv_build,
    sample_path: str,
    exec_name: str,
):
    """Build and run one multi-GPU HIP sample targeting SPIR-V across >=2 GPUs."""
    binary = hip_sample_spirv_build(sample_path, exec_name)

    assert_spirv_offload_bundle(target_executor, rock_dir, binary, f"hip sample {sample_path}")

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {binary}")
    assert result.ok, (
        f"SPIR-V multi-GPU hip sample {sample_path} run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        "PASSED" in result.stdout or "Passed" in result.stdout
    ), f"SPIR-V multi-GPU hip sample {sample_path} did not report PASSED:\n{result.stdout[:2000]}"
