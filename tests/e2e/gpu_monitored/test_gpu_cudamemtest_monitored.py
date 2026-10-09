# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored: GPU memory robustness (cuda_memtest sub-tests 0-5)."""

import pytest


@pytest.mark.gpu_vram(16)
@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_gpu_cudamemtest_monitored(run_monitored_test):
    """Run cudamemtest under amd-smi monitoring with full validation pipeline."""
    outcome = run_monitored_test()
    assert outcome.status == "PASS", f"cudamemtest failed (exit={outcome.exit_code}):\n{outcome.validation}"
