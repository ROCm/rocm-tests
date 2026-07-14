# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run representative public ROCm/rocm-examples HIP-Basic samples."""

import pytest

_EXAMPLES = [
    ("HIP-Basic/bit_extract", "hip_bit_extract", "Validation passed."),
    ("HIP-Basic/dynamic_shared", "hip_dynamic_shared", "Validation passed."),
    ("HIP-Basic/events", "hip_events", "Validation passed."),
    ("HIP-Basic/shared_memory", "hip_shared_memory", "Validation passed."),
    ("HIP-Basic/streams", "hip_streams", "streams completed!"),
]


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("example_path", "exec_name", "marker"), _EXAMPLES, ids=[e[0] for e in _EXAMPLES])
def test_rocm_examples_hip_basic(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    rocm_example_build,
    example_path: str,
    exec_name: str,
    marker: str,
):
    """Build one ROCm/rocm-examples HIP-Basic sample and verify its success marker."""
    binary = rocm_example_build(example_path, exec_name)
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {binary}")
    assert result.ok, (
        f"rocm-examples sample {example_path} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert marker in result.stdout, f"rocm-examples sample {example_path} missed {marker!r}:\n{result.stdout[:2000]}"
