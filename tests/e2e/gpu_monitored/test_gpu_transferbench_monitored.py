# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored: TransferBench GPU-to-GPU bandwidth rsweep."""

import pytest


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_gpu_transferbench_monitored(run_monitored_test):
    """Run transferbench under amd-smi monitoring with full validation pipeline."""
    outcome = run_monitored_test()
    assert outcome.status == "PASS", (
        f"transferbench failed (exit={outcome.exit_code}):\n{outcome.validation}"
    )
