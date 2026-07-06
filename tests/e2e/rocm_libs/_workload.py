# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""_workload.py -- problem-size profiles for the rocm_libs solver workloads.

Each solver binary self-validates its numeric result (``Total Errors: 0`` etc.)
regardless of problem size, so the size is a *gate-tuning* knob, not part of the
pass criterion.  Centralising the per-workload sizes here (keyed by the
``workload_scale`` fixture) lets heavier CI gates drive larger problems without
editing test bodies, and keeps each test file focused on its objective.

Profiles:
    ``smoke`` -- fast, PR/nightly-friendly sizing (the default).
    ``full``  -- larger stress sizing for opt-in heavier runs
                 (``ROCM_TEST_WORKLOAD_SCALE=full``).  These are deliberately
                 bigger problem instances, not a claim of legacy-exact sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

# Environment assignment that routes the HIP stream-ordered memory pool off the
# virtual-memory heap.
#
# Some amdgpu driver/kernel stacks do not support HIP virtual-memory management
# (``hsa_amd_vmem_address_reserve`` fails with status 4097). On those stacks the
# stream-ordered memory pool backing ``hipMallocAsync`` cannot reserve its VM
# heap, so allocations inside hipSPARSE/rocSOLVER fail with
# ``HIPSPARSE_STATUS_ALLOC_FAILED`` / ``rocblas_status_memory_error``.
#
# This is NOT applied unconditionally: the ``hip_mempool_env`` fixture (see
# conftest.py) probes the target host once via ``hip_mempool_probe`` and prepends
# this assignment ONLY when the default VM-backed pool is shown not to work.
# It never alters the workload, its sizes, or its pass criterion, and on
# VMM-capable hosts the default allocation path is left untouched.
HIP_MEM_POOL_ENV = "DEBUG_HIP_MEM_POOL_VMHEAP=0"


@dataclass(frozen=True)
class WorkloadProfile:
    """A sized invocation of a solver workload.

    Attributes:
        args:    Command-line argument string appended after the binary path.
        timeout: Wall-clock timeout (seconds) appropriate for this size.
    """

    args: str
    timeout: float


# name -> scale -> profile.  Smoke values match the original ported invocations.
_PROFILES: dict[str, dict[str, WorkloadProfile]] = {
    "equilibration_batch_kalman": {
        "smoke": WorkloadProfile("--size 512 --nrhs 256 --batch 16 --streams 8 --m 64", 900.0),
        "full": WorkloadProfile("--size 4096 --nrhs 1024 --batch 64 --streams 8 --m 256", 3600.0),
    },
    "async_mixed_precision_workflow": {
        "smoke": WorkloadProfile("--size 2048 --matrices 32 --streams 8", 900.0),
        "full": WorkloadProfile("--size 8192 --matrices 128 --streams 8", 3600.0),
    },
    "sparse_csrrf_analysis_reuse": {
        "smoke": WorkloadProfile("-m 5000 -n 15000 -i 5", 900.0),
        "full": WorkloadProfile("-m 50000 -n 150000 -i 20", 3600.0),
    },
}


def solver_workload(name: str, scale: str) -> WorkloadProfile:
    """Return the sized invocation for solver *name* at problem *scale*.

    Args:
        name:  Workload key (matches the binary's output_name).
        scale: ``"smoke"`` or ``"full"`` (from the ``workload_scale`` fixture).

    Returns:
        The matching ``WorkloadProfile``.

    Raises:
        KeyError: If *name* or *scale* is unknown.
    """
    return _PROFILES[name][scale]
