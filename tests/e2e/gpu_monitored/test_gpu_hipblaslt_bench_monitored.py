# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored: hipBLASLt GEMM performance sweep."""

import pytest


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_gpu_hipblaslt_bench_monitored(run_monitored_test):
    """Run hipblaslt_bench under amd-smi monitoring with full validation pipeline."""
    outcome = run_monitored_test()
    assert outcome.status == "PASS", (
        f"hipblaslt_bench failed (exit={outcome.exit_code}):\n{outcome.validation}"
    )
