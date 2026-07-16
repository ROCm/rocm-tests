# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run the public ROCm/rocm-examples CTest suite."""

import pytest


@pytest.mark.runtime.medium
def test_rocm_examples(target_executor, ld_path: dict, rock_dir: str, rocm_examples_build_dir: str):
    """Build and run the full public ROCm/rocm-examples CTest suite (no exclusions)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    reqs = f"{rock_dir}/libexec/rocprofiler-compute/requirements.txt"
    target_executor.run(f"test ! -f {reqs} || python -m pip install -q -r {reqs}", timeout=600)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
        f"ctest --test-dir {rocm_examples_build_dir} --output-on-failure",
        timeout=7200,
    )
    assert result.ok, (
        f"rocm-examples CTest failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-1000:]}"
    )
    assert "100% tests passed" in result.stdout, f"rocm-examples CTest was not fully green:\n{result.stdout[-4000:]}"
