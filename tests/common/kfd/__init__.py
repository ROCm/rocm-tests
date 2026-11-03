# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared KFD (Kernel Fusion Driver / kfdtest) test-suite helpers.

Mirrors the ``tests/common/criu/`` pattern. Import from here in any conftest
that needs to build or run the kfdtest GTest suite.

Public surface:
    - :func:`build_kfdtest_binary` — sparse-clone libhsakmt and CMake-build kfdtest.
    - :func:`assert_thunk_headers_present` — verify HSAkmt headers exist before building.
    - :func:`ensure_kfd_device` — skip when ``/dev/kfd`` or ``amdgpu`` is absent.
    - :func:`run_kfdtest` — invoke kfdtest with a GTest filter; return the result.
    - :func:`assert_kfd_no_crash_markers` — fail on GPU memory-access faults or segfaults.
"""

from __future__ import annotations

from .installer import assert_thunk_headers_present, build_kfdtest_binary, find_kfdtest_binary
from .runner import assert_kfd_no_crash_markers, ensure_kfd_device, run_kfdtest

__all__ = [
    "assert_kfd_no_crash_markers",
    "assert_thunk_headers_present",
    "build_kfdtest_binary",
    "ensure_kfd_device",
    "find_kfdtest_binary",
    "run_kfdtest",
]
