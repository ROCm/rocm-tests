# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build representative HIP samples for SPIR-V and verify bundle + execution."""

import pytest

from tests.common.spirv import assert_spirv_offload_bundle

# (sample path under samples/, produced executable name)
_SAMPLES = [
    ("0_Intro/bit_extract", "bit_extract"),
    ("0_Intro/square", "square"),
    ("2_Cookbook/0_MatrixTranspose", "MatrixTranspose"),
    ("2_Cookbook/1_hipEvent", "hipEvent"),
    ("2_Cookbook/3_shared_memory", "sharedMemory"),
    ("2_Cookbook/4_shfl", "shfl"),
    ("2_Cookbook/5_2dshfl", "2dshfl"),
    ("2_Cookbook/6_dynamic_shared", "dynamic_shared"),
    ("2_Cookbook/7_streams", "stream"),
    ("2_Cookbook/9_unroll", "unroll"),
    ("2_Cookbook/13_occupancy", "occupancy"),
]


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
    ), f"SPIR-V hip sample {sample_path} ran but did not report 'PASSED'/'Passed':\n{result.stdout[:2000]}"
