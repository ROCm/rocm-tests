# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared fixtures for the packaging suite."""

import pytest

from tests.common.ml_provisioning.spec import gfx_family_for_arch


@pytest.fixture(scope="session")
def artifact_group(gpu_arch: str | None) -> str:
    """TheRock artifact group for the target GPU, derived from ``--gpu-arch``.

    Workflows always supply the architecture so the right package/platform is
    selected, so a missing ``--gpu-arch`` is a CI misconfiguration rather than an
    optional resource and fails outright instead of falling back to a default.
    """
    if not gpu_arch:
        pytest.fail(
            "--gpu-arch is required to select the ROCm package/platform for the packaging suite "
            "but was not passed. Pass --gpu-arch <arch> (e.g. --gpu-arch gfx942)."
        )
    return gfx_family_for_arch(gpu_arch)
