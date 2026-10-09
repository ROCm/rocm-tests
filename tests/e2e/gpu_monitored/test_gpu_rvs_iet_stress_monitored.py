# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored: RVS IET power delivery stress."""

import pytest


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_gpu_rvs_iet_stress_monitored(run_monitored_test):
    """Run rvs_iet_stress under amd-smi monitoring with full validation pipeline."""
    outcome = run_monitored_test()
    assert outcome.status == "PASS", f"rvs_iet_stress failed (exit={outcome.exit_code}):\n{outcome.validation}"
