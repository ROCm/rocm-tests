# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_deviceside_malloc.py -- HIP device-side dynamic allocation correctness.

Builds ``device_side_alloc`` from ``tests/e2e/hip_runtime/src/device_side_alloc.cpp``
and runs each in-kernel allocation scenario as its own parametrized case so a
failure isolates to one allocation path:

    malloc / new / per_thread / per_block / across_kernels

Each scenario allocates inside the kernel, writes verifiable values back to a
host-checked buffer, and prints ``device_side_alloc <scenario>: PASSED``. Pass
requires that sentinel with no allocation fault (``HSA_STATUS_ERROR_EXCEPTION`` /
``Memory access fault``).

runtime.fast is declared explicitly.
"""

import pytest

_SCENARIOS = ["malloc", "new", "per_thread", "per_block", "across_kernels"]
_FAULT_MARKERS = ("FAILED", "HSA_STATUS_ERROR_EXCEPTION", "Memory access fault", "core dumped")


@pytest.mark.runtime.fast
@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_deviceside_malloc(target_executor, ld_path: dict, device_side_alloc_binary: str, scenario: str):
    """Validate one HIP device-side allocation scenario end-to-end."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {device_side_alloc_binary} {scenario}")
    assert result.ok, (
        f"device_side_alloc {scenario!r} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert (
        f"device_side_alloc {scenario}: PASSED" in result.stdout
    ), f"missing PASSED sentinel for {scenario!r}:\n{result.stdout[:2000]}"
    combined = result.stdout + result.stderr
    for marker in _FAULT_MARKERS:
        assert marker not in combined, f"allocation fault marker {marker!r} for {scenario!r}:\n{combined[:2000]}"
