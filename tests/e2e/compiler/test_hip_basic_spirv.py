# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build ROCm/rocm-examples HIP-Basic samples for SPIR-V and run them."""

import pytest

from tests.common.spirv import assert_spirv_offload_bundle

_SAMPLES = [
    ("bit_extract", "hip_bit_extract", "Validation passed."),
    ("dynamic_shared", "hip_dynamic_shared", "Validation passed."),
    ("events", "hip_events", "Validation passed."),
    ("shared_memory", "hip_shared_memory", "Validation passed."),
    ("streams", "hip_streams", "streams completed!"),
]


@pytest.mark.runtime.fast
@pytest.mark.parametrize(("sample_path", "exec_name", "marker"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
def test_hip_basic_spirv(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    hip_basic_spirv_build,
    sample_path: str,
    exec_name: str,
    marker: str,
):
    """Verify a HIP-Basic sample emits a SPIR-V bundle and runs correctly."""
    binary = hip_basic_spirv_build(sample_path, exec_name)

    assert_spirv_offload_bundle(target_executor, rock_dir, binary, f"hip_basic_spirv/{sample_path}")

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary}")
    assert result.ok, (
        f"SPIR-V HIP app run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert marker in result.stdout, f"SPIR-V HIP-Basic sample {sample_path} missed {marker!r}:\n{result.stdout[:2000]}"
