# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Fixtures for amd-smi fclk max clock-limit tests.
"""

import logging

import pytest

from framework.rocm.libs.amd_smi import list_devices
from tests.e2e.amd_smi._fclk import FCLK_SPECS, FclkCaps, derive_caps, restore_default_max

logger = logging.getLogger(__name__)


@pytest.fixture
def fclk_caps(target_executor) -> FclkCaps:
    """Resolve arch-specific fclk caps; restore the default max on teardown.

    Skips when the detected architecture has no entry in ``FCLK_SPECS``.
    """
    devices = list_devices(target_executor)
    if not devices:
        pytest.skip("Could not enumerate GPUs via amd-smi")
    arch = devices[0].arch
    spec = FCLK_SPECS.get(arch)
    if spec is None:
        pytest.skip(f"fclk max-limit validation supported only on {sorted(FCLK_SPECS)}; detected {arch!r}")

    caps = derive_caps(spec)
    logger.info(
        "arch=%s fclk min=%dMHz max=%dMHz; caps valid_set=%dMHz below_min=%dMHz above_max=%dMHz rccl_cap=%dMHz",
        arch,
        caps.default_min,
        caps.default_max,
        caps.valid_set,
        caps.below_min,
        caps.above_max,
        caps.rccl_cap,
    )
    yield caps
    restore_default_max(target_executor, caps.default_max)
