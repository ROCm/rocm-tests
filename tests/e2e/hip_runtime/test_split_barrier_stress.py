# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""HIP cooperative-groups split-barrier stress test.

The vendored workload builds with its own CMake project, then exercises
``barrier_arrive`` / ``barrier_wait`` under workload imbalance and interleaved
rocSOLVER operations.  The legacy pass criterion is preserved: successful exit
plus ``PASSED`` in stdout.
"""

import pytest


@pytest.mark.runtime.medium
def test_split_barrier_stress(
    target_executor,
    ld_path: dict,
    split_barrier_stress_binary: str,
    rock_dir: str,
):
    """Run the split-barrier stress sample on an AMD GPU and assert it PASSED."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {split_barrier_stress_binary}")
    assert result.ok, (
        f"split_barrier_stress failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Legacy criterion: 'PASSED' must appear in the output.
    assert "PASSED" in result.stdout, f"split_barrier_stress did not report 'PASSED':\n{result.stdout[:2000]}"
