# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_concurrent_collectives.py -- RCCL concurrent-collectives multi-GPU stress test.

Validates RCCL correctness under concurrent collective operations on independent
HIP streams without ``ncclGroupStart``/``ncclGroupEnd`` around the collectives
(grouping is used only for ``ncclCommInitRank``).  Targets race conditions,
deadlocks, and resource issues when multiple overlapping collective calls share
the same communicator.

Binary compiled from:
    tests/e2e/concurrent_collectives/src/concurrent_collectives.cpp

Binary output location:
    output/test-binaries/concurrent_collectives/concurrent_collectives

Markers auto-injected by CATEGORY_PROFILES in taxonomy.py (for this directory):
    hw.multi_gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

``runtime.*`` is declared explicitly per function (absent from all profiles).
``ci.weekly`` on the weekly variant overrides the profile-injected ``ci.nightly``.

Prerequisites:
    - ``--rock-dir`` or ``ROCK_DIR`` env var pointing to a ROCm/TheRock install
      that provides ``bin/hipcc``, ``lib/librccl.so``, and ``lib/libamdhip64.so``.
    - At least 2 AMD GPUs visible to the test runner.
"""

import pytest


@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
def test_concurrent_collectives_sanity(
    target_executor,
    ld_path: dict,
    concurrent_collectives_binary: str,
):
    """Run concurrent_collectives in sanity mode on 2+ GPUs.

    Exercises three concurrent collectives per dtype (AllReduce, AllGather,
    Broadcast) on separate HIP streams with default iteration/size settings
    (100 iterations, 16 MB).  Verifies correctness with host-side checks.

    Args:
        target_executor:                   Multi-GPU executor bound to ≥2 GPUs
                                           (``ROCR_VISIBLE_DEVICES=0,1,...``).
        ld_path:                           ``LD_LIBRARY_PATH`` dict for RCCL libs.
        concurrent_collectives_binary:     Path to the compiled binary.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {concurrent_collectives_binary} sanity")
    assert result.ok, (
        f"concurrent_collectives sanity failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )


@pytest.mark.ci.weekly
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.soak
def test_concurrent_collectives_weekly(
    target_executor,
    ld_path: dict,
    concurrent_collectives_binary: str,
):
    """Run concurrent_collectives in weekly (soak) mode on 2+ GPUs.

    Exercises six concurrent collectives per dtype (adds Reduce, ReduceScatter,
    AllToAll) with heavier defaults (1000 iterations, 256 MB).  Extended soak
    run designed to surface intermittent race conditions and resource leaks.

    Args:
        target_executor:                   Multi-GPU executor bound to ≥2 GPUs.
        ld_path:                           ``LD_LIBRARY_PATH`` dict for RCCL libs.
        concurrent_collectives_binary:     Path to the compiled binary.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {concurrent_collectives_binary} weekly",
        timeout=7200.0,
    )
    assert result.ok, (
        f"concurrent_collectives weekly failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
