# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RCCL test result assertion helpers shared across rccl test suites.

Provides a pass-summary assertion for self-validating ``rccl_unroll_test``
binaries that print ``Results: N passed, 0 failed``.  Shared across
``rccl/test_rccl_dual_kernel_build.py`` and ``rccl/test_rccl_unroll_perf_matrix.py``.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Pass summary printed by every self-validating rccl_unroll_test binary.
_RESULTS_PASS_RE = re.compile(r"Results:\s*\d+\s+passed,\s*0\s+failed")


def assert_rccl_results_pass(output: str, label: str = "") -> None:
    """Assert ``"Results: N passed, 0 failed"`` appears in rccl-tests output.

    Self-validating ``rccl_unroll_test`` binaries print this summary line on
    success.  Its absence or a non-zero failed count indicates a collective
    failure that may not be reflected in the process exit code alone.

    Args:
        output: Combined stdout + stderr from the rccl-tests run.
        label:  Short test name used in failure messages.
    """
    prefix = f"{label}: " if label else ""
    assert _RESULTS_PASS_RE.search(output), (
        f"{prefix}rccl-tests did not report 'Results: N passed, 0 failed'.\n" f"output tail: {output[-2000:]}"
    )
