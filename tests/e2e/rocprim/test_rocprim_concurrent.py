# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocprim_concurrent.py -- rocPRIM multi-stream concurrency system tests.

Validates rocPRIM correctness and stream-safety under concurrent usage patterns:
multiple independent workloads (reduce, radix_sort, inclusive_scan) running in
parallel on different HIP streams, and multi-stage pipelines composed from
rocPRIM primitives.

Binary compiled via CMake from:
    tests/e2e/rocprim/src/system/concurrency/test_system_multistream.cpp
    tests/e2e/rocprim/src/system/reliability/test_system_multigpu_hmm.cpp
    tests/e2e/rocprim/src/system/stress/test_system_performance_stress.cpp

Binary output location:
    output/test-binaries/rocprim/build/rocprim_tests

``runtime.medium`` is declared explicitly (< 2 min for concurrency filter).

GTest suites in the binary:
    SystemMultiStreamTests  — single-GPU concurrency (reduce, sort, scan, event deps)
    MultiGPUHMMTests        — HMM managed-memory across 2 GPUs (requires hw.multi_gpu)
    SystemStressTests       — longevity/stress (ci.weekly, runtime.soak; see test_rocprim_stress.py)
"""

import pytest

# GPU memory fault patterns written to stderr when a kernel accesses an invalid
# address. These indicate a real kernel bug (e.g. rocRAND philox engine under
# concurrent HIP streams) and must be detected before the generic result.ok check
# so CI gets an actionable error message pointing at the rocRAND/rocPRIM issue.
_GPU_FAULT_PATTERNS = [
    "Memory Fault Error",
    "GPU core dump",
]


@pytest.mark.runtime.medium
@pytest.mark.retry(count=1)
def test_rocprim_concurrent(
    target_executor,
    ld_path: dict,
    rocprim_tests_binary: str,
):
    """rocPRIM concurrency: multi-stream reduce/sort/scan correctness.

    Runs the ``SystemMultiStreamTests`` GTest filter, which exercises rocPRIM
    device algorithms (reduce, radix sort, inclusive scan) concurrently across
    multiple HIP streams with separate temporary storage.  Validates:

    - Stream correctness: rocPRIM launches work on the provided stream.
    - Isolation: concurrent operations on different streams do not corrupt
      each other's inputs, outputs, or temporary buffers.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} " f"{rocprim_tests_binary} --gtest_filter=SystemMultiStreamTests.*"
    )
    # Detect GPU memory faults before the generic result.ok check — they produce a
    # more actionable error pointing at the rocRAND/rocPRIM kernel issue rather than
    # a bare "exit code 1" message.
    for pat in _GPU_FAULT_PATTERNS:
        assert pat not in result.stderr, (
            f"GPU memory fault in rocPRIM concurrent test (pattern: {pat!r}).\n"
            f"The rocRAND philox4x32_10_engine kernel accessed an invalid address "
            f"under concurrent HIP streams. Check the rocRAND and rocPRIM versions "
            f"in the TheRock artifact for this run.\n"
            f"faulting stderr:\n{result.stderr[:1000]}"
        )
    assert result.ok, (
        f"rocprim_concurrent failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "FAILED" not in result.stdout, (
        f"rocprim_concurrent: GTest reported test failures:\n" f"{result.stdout[:2000]}"
    )
    assert "PASSED" in result.stdout or "All tests passed!" in result.stdout, (
        f"rocprim_concurrent: GTest pass token not found in stdout:\n" f"{result.stdout[:2000]}"
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.layer.math_lib
@pytest.mark.ci.nightly
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.medium
def test_rocprim_multigpu_hmm(
    target_executor,
    ld_path: dict,
    rocprim_tests_binary: str,
):
    """rocPRIM HMM managed-memory correctness across 2 GPUs.

    Runs the ``MultiGPUHMMTests`` GTest filter, which exercises rocPRIM under
    hipMallocManaged (HMM) scenarios across two GPUs:

    - Memory migration: alternating GPU access triggers page migration;
      validates rocPRIM sort/reduce correctness after migration.
    - Memory coherence: partitioned concurrent sorts via managed memory on two
      GPUs simultaneously; validates no cross-GPU data corruption.

    Requires 2 GPUs (``hw.multi_gpu`` + ``gpu_count(2)``).
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} " f"{rocprim_tests_binary} --gtest_filter=MultiGPUHMMTests.*"
    )
    assert result.ok, (
        f"rocprim_multigpu_hmm failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "FAILED" not in result.stdout, (
        f"rocprim_multigpu_hmm: GTest reported test failures:\n" f"{result.stdout[:2000]}"
    )
    assert "PASSED" in result.stdout or "All tests passed!" in result.stdout, (
        f"rocprim_multigpu_hmm: GTest pass token not found in stdout:\n" f"{result.stdout[:2000]}"
    )
