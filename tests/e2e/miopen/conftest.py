# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Session-level fixtures for the MIOpen driver test area."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def require_miopen_driver(rock_dir: str, target_executor) -> None:
    """Fail early if MIOpenDriver binary is absent or not executable."""
    driver = f"{rock_dir}/bin/MIOpenDriver"
    if not os.access(driver, os.X_OK):
        result = target_executor.run(f"test -x {driver}")
        if not result.ok:
            pytest.fail(
                f"MIOpenDriver not found or not executable at {driver}. "
                "Ensure MIOpen is installed and rock_dir points to the correct ROCm path."
            )
