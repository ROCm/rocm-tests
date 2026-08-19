# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmi_fclk_max_limit.py -- amd-smi fclk max clock-limit validation.

Validates ``amd-smi set --clk-limit fclk max`` behaviour across four cases:

    - set an in-range value and confirm every GPU reflects it
    - reject a value below the documented minimum
    - reject a value above the documented maximum
    - enforce the cap while an RCCL all-reduce workload runs

All cap values are derived per-architecture from the min/max defaults in
``FCLK_SPECS``; unlisted architectures are skipped (see the ``fclk_caps``
fixture).
"""

import logging
import re
import time

import pytest

from framework.rocm.libs.amd_smi import parse_fclk_per_gpu
from tests.e2e.amd_smi._fclk import (
    SETTLE_SECS,
    FclkCaps,
    assert_default_max,
    combined_output,
    metric_clock_output,
    set_fclk_max,
)

logger = logging.getLogger(__name__)

# Seconds to let the RCCL workload touch the GPUs before capping.
_RCCL_WARMUP_SECS = 10
# Number of ~SETTLE_SECS samples taken while the cap is enforced under load.
_RCCL_SAMPLE_COUNT = 12


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_amdsmi_fclk_max_set_valid_range(target_executor, fclk_caps: FclkCaps, amd_smi_bin: str):
    """Set fclk max to an in-range value and verify every GPU reflects it."""
    assert_default_max(target_executor, fclk_caps.default_max, amd_smi_bin)

    new_max = fclk_caps.valid_set
    set_fclk_max(target_executor, new_max, amd_smi_bin)
    time.sleep(SETTLE_SECS)

    info = parse_fclk_per_gpu(metric_clock_output(target_executor, amd_smi_bin))
    assert info, "Could not parse FCLK_0 info from 'amd-smi metric -c'"
    mismatched = [g for g in info if g["max_clk"] != new_max]
    assert not mismatched, f"fclk MAX_CLK not updated to {new_max}MHz on: {mismatched}"


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_amdsmi_fclk_max_set_below_min(target_executor, fclk_caps: FclkCaps, amd_smi_bin: str):
    """Attempt fclk max below the minimum and expect an explicit rejection."""
    assert_default_max(target_executor, fclk_caps.default_max, amd_smi_bin)

    output = combined_output(set_fclk_max(target_executor, fclk_caps.below_min, amd_smi_bin))
    expected = re.compile(r"CLK_LIMIT:\s*Cannot set fclk max value less than min", re.IGNORECASE)
    assert expected.search(
        output
    ), f"Expected 'Cannot set fclk max value less than min' not found in output: {output!r}"


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.fast
def test_amdsmi_fclk_max_set_above_max(target_executor, fclk_caps: FclkCaps, amd_smi_bin: str):
    """Attempt fclk max above the maximum and expect a NOT_SUPPORTED reply."""
    assert_default_max(target_executor, fclk_caps.default_max, amd_smi_bin)

    probe = fclk_caps.above_max
    output = combined_output(set_fclk_max(target_executor, probe, amd_smi_bin))
    time.sleep(SETTLE_SECS)
    metric_clock_output(target_executor, amd_smi_bin)

    expected = re.compile(
        rf"CLK_LIMIT:\s*\[AMDSMI_STATUS_NOT_SUPPORTED\]\s*Unable to set max of fclk to {probe}\s*MHz",
        re.IGNORECASE,
    )
    assert expected.search(output), f"Expected AMDSMI_STATUS_NOT_SUPPORTED message not found in output: {output!r}"


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_amdsmi_fclk_max_enforced_under_rccl_workload(
    target_executor,
    fclk_caps: FclkCaps,
    rock_dir: str,
    requested_gpu_count: int,
    amd_smi_bin: str,
):
    """Cap fclk max under an RCCL all-reduce workload and verify enforcement."""
    all_reduce_perf = f"{rock_dir.rstrip('/')}/bin/all_reduce_perf" if rock_dir else "all_reduce_perf"
    assert target_executor.run(
        f"test -f '{all_reduce_perf}'"
    ).ok, f"all_reduce_perf not found at {all_reduce_perf!r}; rccl binaries are required for this test"

    ngpus = requested_gpu_count
    assert ngpus > 0, "No GPUs detected; cannot run rccl workload test"

    # -T 80 bounds the run to ~1.5 minutes; -n 10000 gives enough work per size.
    rccl_cmd = f"{all_reduce_perf} -b 8 -e 2G -f 2 -n 10000 -g {ngpus} -T 80 -d all -o all -y managed"
    cap_mhz = fclk_caps.rccl_cap
    min_acceptable = fclk_caps.default_min

    samples: list[list[dict]] = []
    violations: list[str] = []
    with target_executor.start_background(rccl_cmd):
        time.sleep(_RCCL_WARMUP_SECS)
        logger.info("Set fclk max output: %s", combined_output(set_fclk_max(target_executor, cap_mhz, amd_smi_bin)))

        for i in range(_RCCL_SAMPLE_COUNT):
            time.sleep(SETTLE_SECS)
            info = parse_fclk_per_gpu(metric_clock_output(target_executor, amd_smi_bin))
            if not info:
                violations.append(f"sample#{i} - failed to get fclk info.")
                continue
            samples.append(info)
            for g in info:
                clk = g.get("clk")
                if clk is None:
                    violations.append(f"sample#{i} GPU{g['gpu']} Received invalid CLK={clk} MHz")
                    continue
                if clk < min_acceptable or clk > cap_mhz:
                    violations.append(f"sample#{i} GPU{g['gpu']} CLK={clk}MHz outside [{min_acceptable},{cap_mhz}]MHz")

    assert samples, "No FCLK samples captured during rccl workload"
    assert not violations, "; ".join(violations)
