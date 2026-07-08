# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""_workload.py -- shared run helpers for the RCCL unroll-factor test family.

The ``rccl_unroll_test`` binaries (dual-kernel build, perf matrix, and any
future unroll sub-test) are self-validating harnesses that drive ``all_reduce_perf``
under a common environment and print a ``Results: N passed, M failed`` summary.
Centralising the environment builder and the pass-summary pattern here keeps the
per-test files to their unique objective and makes adding another unroll sub-test
a one-liner.
"""

from __future__ import annotations

import re

# Pass summary printed by every self-validating rccl_unroll_test binary.
RESULTS_PASS_RE = re.compile(r"Results:\s*\d+ passed,\s*0 failed")


def all_reduce_perf_env(
    *,
    rock_dir: str,
    ld_library_path: str,
    all_reduce_perf: str,
    rccl_lib: str | None = None,
) -> str:
    """Build the ``env ...`` command prefix the rccl_unroll_test binaries expect.

    Reproduces the legacy ``_build_run_command`` environment.  ``rccl_lib`` is
    only required by binaries that ``dlopen`` librccl directly (the dual-kernel
    symbol probe); perf-only binaries omit it.

    Args:
        rock_dir:        ROCm/TheRock install path (``ROCM_PATH``).
        ld_library_path: Value for ``LD_LIBRARY_PATH`` (from the ``ld_path`` fixture).
        all_reduce_perf: Absolute path to the ``all_reduce_perf`` client.
        rccl_lib:        Optional path to ``librccl.so`` (``RCCL_LIB_PATH``).

    Returns:
        A shell ``env K=V ...`` prefix string (no trailing space) to prepend to
        the binary invocation.
    """
    parts = [
        "env",
        "HIP_PLATFORM=amd",
        f"ROCM_PATH={rock_dir}",
        f"LD_LIBRARY_PATH={ld_library_path}",
        "NCCL_DEBUG=INFO",
        "RCCL_TEST_VERBOSE=1",
        f"ALL_REDUCE_PERF_PATH={all_reduce_perf}",
    ]
    if rccl_lib is not None:
        parts.append(f"RCCL_LIB_PATH={rccl_lib}")
    return " ".join(parts)
