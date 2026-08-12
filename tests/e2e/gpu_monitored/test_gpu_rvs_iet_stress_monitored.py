# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored RVS IET (Input Energy Test) with full analysis pipeline.

Runs IET stress configs under continuous amd-smi monitoring, then performs
5-layer validation, anomaly detection, and generates an HTML report.
"""

import pytest


@pytest.mark.runtime.medium
@pytest.mark.parametrize(
    "conf_name",
    [
        "iet_stress.conf",
    ],
)
def test_gpu_rvs_iet_stress_monitored(
    run_monitored_rvs,
    rvs_find_conf,
    gpu_conf_dir,
    kernel_health_probe,
    conf_name,
):
    """Run RVS IET module with GPU monitoring and full validation."""
    kernel_health_probe(lookback_sec=300, strict=False)

    conf_file = rvs_find_conf(conf_name, gpu_conf_dir=gpu_conf_dir)
    if conf_file is None:
        pytest.skip(f"No IET config found: {conf_name}")

    result = run_monitored_rvs(
        conf_file=conf_file,
        test_name="gpu_rvs_iet_stress_monitored",
        timeout=3600,
    )

    assert result["passed"], f"RVS IET failed ({conf_name}):\n{result['validation']}"
