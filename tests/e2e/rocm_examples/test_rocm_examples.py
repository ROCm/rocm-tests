# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run the public ROCm/rocm-examples CTest suite."""

import pytest

# Full amd-mainline CTest is green on gfx942 except callback/debug/profiler samples
# that need extra runtime/debug-profiler environment beyond the examples app set.
_KNOWN_FAILING = "hipfft_callback|rocfft_callback|rocgdb-.*|rocprof.*"


@pytest.mark.runtime.medium
def test_rocm_examples(target_executor, ld_path: dict, rock_dir: str, rocm_examples_build_dir: str):
    """Run ROCm/rocm-examples CTest, excluding known failing callback samples."""
    ld = ld_path["LD_LIBRARY_PATH"]
    reqs = f"{rock_dir}/libexec/rocprofiler-compute/requirements.txt"
    target_executor.run(f"test ! -f {reqs} || python -m pip install -q -r {reqs}", timeout=600)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
        f"ctest --test-dir {rocm_examples_build_dir} --output-on-failure -E '{_KNOWN_FAILING}'",
        timeout=7200,
    )
    assert result.ok, (
        f"rocm-examples CTest failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-1000:]}"
    )
    assert "100% tests passed" in result.stdout, f"rocm-examples CTest was not fully green:\n{result.stdout[-4000:]}"
