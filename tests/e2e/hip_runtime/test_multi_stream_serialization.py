# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_multi_stream_serialization.py -- HIP multi-stream kernel serialization correctness.

Validates that AMD_SERIALIZE_KERNEL=1 (set internally by the binary via setenv)
correctly enforces serialized kernel dispatch across multiple HIP streams. Each
round: N streams submit M kernels concurrently; the serialization layer must
ensure correct ordering. Verifies device-side and host-side ordering invariants.

Regression guard for kernel scheduling correctness under AMD_SERIALIZE_KERNEL.

Binary compiled via CMake from:
    tests/e2e/hip_runtime/src/multi_stream_serialization.cpp

Environment variables used (set by binary internally, no pre-export needed):
    AMD_SERIALIZE_KERNEL=1    (set via setenv inside the binary)
    KS_STREAMS=4              (default; overridden by env before launch)
    KS_KERNELS=8              (smoke; default=40)
    KS_ELEMENTS=65536         (smoke; default=262144)
    KS_ROUNDS=8               (smoke; default=900)

Markers auto-injected by CATEGORY_PROFILES (tests/e2e/hip_runtime):
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

runtime.fast is declared explicitly.
"""

import pytest


@pytest.mark.runtime.fast
def test_multi_stream_serialization(
    target_executor,
    ld_path: dict,
    multi_stream_serialization_binary: str,
):
    """Validate HIP multi-stream serialization correctness (smoke: 8 rounds).

    Args:
        target_executor:                    Location-transparent GPU executor.
        ld_path:                            LD_LIBRARY_PATH dict for ROCm libs.
        multi_stream_serialization_binary: Path to compiled binary.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    # Smoke parameters: KS_ROUNDS=8, KS_KERNELS=8, KS_ELEMENTS=65536
    # AMD_SERIALIZE_KERNEL is set internally by the binary via setenv().
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld}"
        f" KS_ROUNDS=8 KS_KERNELS=8 KS_ELEMENTS=65536"
        f" {multi_stream_serialization_binary}"
    )
    assert result.ok, (
        f"multi_stream_serialization failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "OVERALL: PASSED" in result.stdout, f"Expected 'OVERALL: PASSED' in stdout:\n{result.stdout[:2000]}"
    assert "OVERALL: FAILED" not in result.stdout, f"OVERALL: FAILED found in stdout:\n{result.stdout[:2000]}"
    assert "FAIL" not in result.stderr, f"Failure/error detected in stderr:\n{result.stderr[:1000]}"
