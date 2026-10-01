# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Assert the HIP stream-ordered memory pool gives back everything it reserved.

Runs the upstream ``Performance_MempoolManager_hipMallocAsync_hipFreeAsync``
catch2 scenario, which reserves 30 GB through ``hipMallocAsync``, releases it
through ``hipFreeAsync``, and prints memory counters after each step. Upstream
only checks the allocations returned non-null, so the leak assertion lives here.
"""

import re

import pytest

_TEST_CASE = "Performance_MempoolManager_hipMallocAsync_hipFreeAsync"

# Selects the memory-manager pool (upstream SWDEV-497841) over the legacy
# allocator. Currently the runtime default, but pinned because the paths differ
# wildly: on gfx942/ROCm 10.2 the legacy one still holds 14 GB after all 30 GB
# is freed, where this one holds 768 MB.
_VMHEAP_ENV = "DEBUG_HIP_MEM_POOL_VMHEAP=1"

# Printed by HIP_SKIP_TEST (hip_test_common.hh) when the device has under 30 GB
# free. Without it that case parses zero snapshots and fails as a format change.
_SKIP_SENTINEL = "HIP_SKIP_THIS_TEST"

# One block per step: start, after reserving 30 GB, after releasing 10, after
# releasing the rest.
_SNAPSHOT_RE = re.compile(
    r"Total device memory \(GB\)\s*:\s*(\d+)\s+"
    r"Free device memory \(GB\)\s*:\s*(\d+)\s+"
    r"Pool Reserved current\(GB\)\s*:\s*(\d+)\s+"
    r"Pool Used current \(GB\)\s*:\s*(\d+)"
)
_EXPECTED_SNAPSHOTS = 4

# The scenario runs in a couple of seconds; this bounds a wedged GPU or stalled
# connection rather than budgeting for slow hardware.
_RUN_TIMEOUT = 300.0


@pytest.mark.runtime.fast
def test_hip_mem_pool(target_executor, ld_path: dict, rock_dir: str, hip_perf_memory_binary: str):
    """Reserve and release 30 GB via the mempool, then assert nothing was retained."""
    # MemoryPerformance links the ROCm-bundled librocm_sysdeps_numa.so.1.
    ld = f"{rock_dir}/lib/rocm_sysdeps/lib:" + ld_path["LD_LIBRARY_PATH"]
    env = f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {_VMHEAP_ENV}"
    result = target_executor.run(f"{env} {hip_perf_memory_binary} {_TEST_CASE!r}", timeout=_RUN_TIMEOUT)
    output = result.stdout + result.stderr
    if _SKIP_SENTINEL in output:
        pytest.skip(f"{_TEST_CASE}: device has under 30 GB free")
    assert result.ok, f"{_TEST_CASE} failed (exit={result.exit_code}):\n{output[-1500:]}"

    snapshots = [
        {"total": int(total), "free": int(free), "reserved": int(reserved), "used": int(used)}
        for total, free, reserved, used in _SNAPSHOT_RE.findall(output)
    ]
    assert len(snapshots) == _EXPECTED_SNAPSHOTS, (
        f"expected {_EXPECTED_SNAPSHOTS} memory snapshots, parsed {len(snapshots)} — "
        f"upstream may have changed its output format:\n{output[-1500:]}"
    )
    assert snapshots[0] == snapshots[-1], (
        "mempool did not return to its starting state after releasing all 30 GB "
        f"(start={snapshots[0]}, end={snapshots[-1]})"
    )
