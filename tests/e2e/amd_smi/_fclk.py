# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
_fclk.py -- Shared helpers for amd-smi fclk max clock-limit tests.

Holds the per-architecture reference values, the derived cap dataclass, and the
thin ``amd-smi`` command wrappers shared by the fixture and the test module.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import time

from framework.common.helpers import ExecutionResult
from framework.rocm.libs.amd_smi import parse_fclk_per_gpu

logger = logging.getLogger(__name__)

# Per-architecture fclk reference values (MHz). Add GPU variants here as the
# suite matures; unlisted architectures are skipped. All other cap values are
# derived from these min/max defaults in ``derive_caps``.
FCLK_SPECS: dict[str, dict[str, int]] = {
    "gfx942": {"fclk_default_max": 2000, "fclk_default_min": 1200},
}

# Step (MHz) used to derive in/out-of-range probes from the min/max defaults.
FCLK_DERIVE_STEP_MHZ = 100

# Seconds to wait for an fclk change to reflect in amd-smi.
SETTLE_SECS = 5


@dataclass(frozen=True)
class FclkCaps:
    """Derived fclk cap values for the active architecture (all MHz)."""

    default_min: int
    default_max: int
    valid_set: int
    below_min: int
    above_max: int
    rccl_cap: int


def derive_caps(spec: dict[str, int]) -> FclkCaps:
    """Compute the per-case fclk caps from an architecture's min/max defaults."""
    step = FCLK_DERIVE_STEP_MHZ
    lo = spec["fclk_default_min"]
    hi = spec["fclk_default_max"]
    return FclkCaps(
        default_min=lo,
        default_max=hi,
        # In-range value: midpoint snapped down to the nearest step.
        valid_set=((lo + hi) // 2 // step) * step,
        # Out-of-range probes: one step beyond the documented bounds.
        below_min=lo - step,
        above_max=hi + step,
        # RCCL cap: just above min so enforcement is easy to detect under load.
        rccl_cap=lo + step,
    )


def combined_output(result: ExecutionResult) -> str:
    """Return the command's stdout and stderr joined as one text blob."""
    return "\n".join(part for part in (result.stdout, result.stderr) if part)


def metric_clock_output(executor) -> str:
    """Return raw ``amd-smi metric -c`` output."""
    return combined_output(executor.run("amd-smi metric -c"))


def set_fclk_max(executor, value_mhz: int) -> ExecutionResult:
    """Run ``amd-smi set --clk-limit fclk max <value_mhz>`` with privilege."""
    return executor.run(f"sudo -n amd-smi set --clk-limit fclk max {value_mhz}")


def restore_default_max(executor, default_max: int) -> None:
    """Best-effort restore of the default fclk max; failures are logged only."""
    try:
        set_fclk_max(executor, default_max)
        time.sleep(SETTLE_SECS)
        metric_clock_output(executor)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to restore fclk max to %dMHz", default_max)


def assert_default_max(executor, default_max: int) -> None:
    """Assert every GPU reports the documented default fclk MAX_CLK."""
    info = parse_fclk_per_gpu(metric_clock_output(executor))
    assert info, "Could not parse FCLK_0 info from 'amd-smi metric -c'"
    bad = [g for g in info if g["max_clk"] != default_max]
    assert not bad, f"Default fclk MAX_CLK mismatch (expected {default_max}MHz): {bad}"
