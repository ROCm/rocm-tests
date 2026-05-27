# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocprim_stress.py -- rocPRIM aggressive stress / longevity tests.

Exercises the ``SystemStressTests`` GTest suite from the shared ``rocprim_tests``
binary.  These tests are intentionally excluded from nightly CI (they can run
for many minutes) and are gated to the weekly CI gate instead.

Binary compiled via CMake from:
    tests/e2e/rocprim/src/system/stress/test_system_performance_stress.cpp
    (along with the other system sources; see conftest.py)

Binary output location:
    output/test-binaries/rocprim/build/rocprim_tests

Markers declared explicitly (override CATEGORY_PROFILES ci.nightly injection):
    hw.gpu, layer.math_lib, ci.weekly, e2e.stack, os.linux, runtime.soak

GTest cases in SystemStressTests:
    MaximumMemoryPressure_100Percent
    MassiveConcurrency_128Streams
    MixedWorkload_ConcurrentDifferentOperations
    LinearScaling_DataSize
"""

import pytest

# GPU memory fault patterns written to stderr when a kernel accesses an invalid
# address. Detected before the generic result.ok check for a more actionable message.
_GPU_FAULT_PATTERNS = [
    "Memory Fault Error",
    "GPU core dump",
]


@pytest.mark.hw.gpu
@pytest.mark.layer.math_lib
@pytest.mark.ci.weekly
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.soak
def test_rocprim_stress(
    target_executor,
    ld_path: dict,
    rocprim_tests_binary: str,
):
    """rocPRIM stress: memory pressure, massive concurrency, and mixed workloads.

    Runs the ``SystemStressTests`` GTest filter, which exercises rocPRIM under
    aggressive conditions:

    - ``MaximumMemoryPressure_100Percent``: allocates 100 % of available VRAM
      and runs rocPRIM sort/reduce under memory pressure.
    - ``MassiveConcurrency_128Streams``: launches rocPRIM kernels on 128 HIP
      streams simultaneously and validates result correctness.
    - ``MixedWorkload_ConcurrentDifferentOperations``: concurrent sort, reduce,
      and scan on independent streams; validates no cross-stream corruption.
    - ``LinearScaling_DataSize``: sweeps data size from small to large and
      asserts linear scaling of wall time.

    Excluded from nightly (``ci.nightly``) because individual cases can exceed
    2 minutes; scheduled in the weekly CI gate (``ci.weekly``, ``runtime.soak``).

    Args:
        target_executor:      Executor bound to the allocated GPU.
        ld_path:              ``LD_LIBRARY_PATH`` dict for ROCm libs.
        rocprim_tests_binary: Path to the compiled GTest binary.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} " f"{rocprim_tests_binary} --gtest_filter=SystemStressTests.*"
    )
    for pat in _GPU_FAULT_PATTERNS:
        assert pat not in result.stderr, (
            f"GPU memory fault in rocPRIM stress test (pattern: {pat!r}).\n" f"Faulting stderr:\n{result.stderr[:1000]}"
        )
    assert result.ok, (
        f"rocprim_stress failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "FAILED" not in result.stdout, f"rocprim_stress: GTest reported test failures:\n{result.stdout[:2000]}"
    assert (
        "PASSED" in result.stdout or "All tests passed!" in result.stdout
    ), f"rocprim_stress: GTest pass token not found in stdout:\n{result.stdout[:2000]}"
