# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Test-suite-level fixtures.

This conftest extends the root conftest (which loads all framework plugins).
Fixtures defined here are available to all tests under ``tests/`` but are
NOT visible to framework code — keeping framework and test concerns separate.

Add test-specific shared fixtures here, e.g.:
  - Parametrized hardware configurations
  - Test-data factories
  - Reusable assertion helpers that are test-only concerns

Framework-level fixtures (gpu_fixture, health_fixture, etc.) are loaded
by the root conftest.py via pytest_plugins — do not re-declare them here.
"""

from __future__ import annotations

import pytest

from tests.common.factories import fake_execution_result, fake_gpu_info

# ---------------------------------------------------------------------------
# Fixtures available to all tests in the suite
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_gpu_info():
    """Provide a synthetic GpuInfo for tests that don't need real hardware.

    Returns:
        GpuInfo: Synthetic gfx942 GPU descriptor with 32 GB VRAM.
    """
    return fake_gpu_info()


@pytest.fixture
def mock_ok_result():
    """Provide a synthetic successful ExecutionResult for framework unit tests.

    Returns:
        ExecutionResult: exit_code=0, minimal synthetic stdout.
    """
    return fake_execution_result(exit_code=0, stdout="RESULT_OK\nTHROUGHPUT_TFLOPS=12.5\n")


@pytest.fixture
def mock_fail_result():
    """Provide a synthetic failed ExecutionResult for framework unit tests.

    Returns:
        ExecutionResult: exit_code=1, error message in stderr.
    """
    return fake_execution_result(exit_code=1, stderr="hipErrorInvalidDevice: no GPU found")
