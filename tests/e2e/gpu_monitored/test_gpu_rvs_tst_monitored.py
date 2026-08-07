# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored RVS TST (Thermal Stress Test) with full analysis pipeline.

Runs TST configs under continuous amd-smi monitoring, then performs
5-layer validation, anomaly detection, and generates an HTML report.
"""

import pytest


@pytest.mark.runtime.medium
@pytest.mark.parametrize(
    "conf_name",
    [
        "tst_single.conf",
    ],
)
def test_gpu_rvs_tst_monitored(
    run_monitored_rvs,
    rvs_find_conf,
    gpu_conf_dir,
    kernel_health_probe,
    conf_name,
):
    """Run RVS TST module with GPU monitoring and full validation."""
    kernel_health_probe(lookback_sec=300, strict=False)

    conf_file = rvs_find_conf(conf_name, gpu_conf_dir=gpu_conf_dir)
    if conf_file is None:
        pytest.skip(f"No TST config found: {conf_name}")

    result = run_monitored_rvs(
        conf_file=conf_file,
        test_name="gpu_rvs_tst_monitored",
        timeout=1800,
    )

    assert result["passed"], f"RVS TST failed ({conf_name}):\n{result['validation']}"
