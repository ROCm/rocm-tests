# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run representative upstream HIP samples.

Generic ``hip_samples``: each selected sample
builds with its own CMake project, runs, and prints ``PASSED`` or ``Passed``.
Special legacy cases that need extra toolchains, multi-GPU setup, or custom
parsers are intentionally left out of this representative single-GPU set.
"""

import pytest

# (sample path under samples/, produced executable name) — executable names are
# copied verbatim from the legacy hip_samples testcases table.
_SAMPLES = [
    ("0_Intro/bit_extract", "bit_extract"),
    ("0_Intro/square", "square"),
    ("2_Cookbook/0_MatrixTranspose", "MatrixTranspose"),
    ("2_Cookbook/1_hipEvent", "hipEvent"),
    ("2_Cookbook/3_shared_memory", "sharedMemory"),
    ("2_Cookbook/4_shfl", "shfl"),
    ("2_Cookbook/7_streams", "stream"),
    ("2_Cookbook/9_unroll", "unroll"),
    ("2_Cookbook/13_occupancy", "occupancy"),
]


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("sample_path", "exec_name"), _SAMPLES, ids=[s[0] for s in _SAMPLES])
def test_hip_samples(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    hip_sample_build,
    sample_path: str,
    exec_name: str,
):
    """Build one HIP sample via its CMake project, run it, and assert it PASSED."""
    # Resolving the build triggers the per-sample cmake configure + make; a build
    # failure surfaces as a fixture ERROR (mirrors the legacy per-sample build step).
    binary = hip_sample_build(sample_path, exec_name)

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {binary}")
    assert result.ok, (
        f"hip sample {sample_path} run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Legacy generic criterion: 'PASSED' or 'Passed' must appear in the output.
    assert (
        "PASSED" in result.stdout or "Passed" in result.stdout
    ), f"hip sample {sample_path} did not report 'PASSED'/'Passed':\n{result.stdout[:2000]}"
