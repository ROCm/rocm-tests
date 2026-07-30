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

import os

import pytest

from tests.common.factories import fake_execution_result, fake_gpu_info
from tests.common.ml_provisioning.fixtures import ensure_pytorch_env, torch_python as _torch_python

_VALID_WORKLOAD_SCALES = ("smoke", "full")

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


@pytest.fixture
def requested_gpu_count(request: pytest.FixtureRequest, target_executor) -> int:
    """Return the ``@pytest.mark.gpu_count(N)`` value for the test (default 2).

    Lets multi-GPU tests pass a count (``-g N`` / ``--ngpus N``) that matches the
    GPUs the framework acquired and exposed via ``ROCR_VISIBLE_DEVICES``.  Shared
    here so every area (rccl, rocm_libs, ...) reuses one definition.

    Args:
        request: The pytest request object.

    Returns:
        Requested GPU count, or 2 when the marker is absent.
    """
    marker = request.node.get_closest_marker("gpu_count")
    if marker and marker.args:
        raw_count = marker.args[0]
        if isinstance(raw_count, str) and raw_count.upper() == "ALL":
            return int(getattr(target_executor, "visible_gpu_count", 1))
        return int(raw_count)
    return 2


# ---------------------------------------------------------------------------
# PyTorch provisioning fixtures — available to every test.
# Backed by tests/common/ml_provisioning/; provisions per execution node via
# --pre-install pytorch=... or lazily here (auto channel: multiarch -> family).
# Never set ROCR_VISIBLE_DEVICES or hardcode an index in tests.
#
# Scope rationale: pytorch_env and torch_python are function-scoped even though
# provisioning is expensive. The session-level sanity cache
# (config._framework_sanity_ok, written by ensure_pytorch_env only after a
# successful install + sanity check) makes every call after the first a near-
# zero-cost dict lookup. Function scope gives transient-failure retry semantics:
# if a lazy install fails on test N, test N+1 re-attempts rather than being
# permanently blocked. With --pre-install (Path B), the pre-install result is
# promoted into the sanity cache on first fixture access, so all subsequent
# tests share it via Phase 1 — no re-install occurs.
# ---------------------------------------------------------------------------


@pytest.fixture
def pytorch_env(request, target_executor, framework_config):
    """Return a pre-installed or lazily provisioned PyTorch environment.

    Function-scoped with a session-level sanity cache (``config._framework_sanity_ok``).
    pytest invokes this fixture per test, but after the first successful provision
    the cache hit in Phase 1 makes subsequent calls O(1) with no subprocess.

    Two paths:

    * **No** ``--pre-install``: lazy install from ``rocm-test.toml`` channels on
      first request; result cached only after sanity passes; retried on failure.
    * **With** ``--pre-install pytorch=...``: pre-install result promoted to the
      sanity cache on first fixture access; all tests share it via Phase 1.
    """
    return ensure_pytorch_env(request, target_executor, framework_config)


@pytest.fixture
def torch_python(pytorch_env) -> str:
    """Return the Python executable for the provisioned PyTorch environment."""
    return _torch_python(pytorch_env)


@pytest.fixture
def require_torch(pytorch_env) -> None:
    """Fail the test when PyTorch (ROCm) installation did not succeed on the execution node."""
    if not pytorch_env.ok:
        pytest.fail(pytorch_env.skip_reason())


@pytest.fixture
def require_torch_tunableop(require_torch, pytorch_env, target_executor) -> None:
    """Skip when the PyTorch build lacks TunableOp support (implies ``require_torch``)."""
    python = _torch_python(pytorch_env)
    if not target_executor.run(f'{python} -c "import torch.cuda.tunable"').ok:
        pytest.skip("PYTORCH_TUNABLEOP not supported in this PyTorch build")


@pytest.fixture(scope="session")
def workload_scale() -> str:
    """Problem-size profile for parametrically-sized workloads.

    Lets heavier CI gates drive larger problems without editing test code,
    following the repo's env-override config idiom.  Set
    ``ROCM_TEST_WORKLOAD_SCALE=full`` to opt into the larger profile; defaults to
    ``smoke`` (fast PR/nightly-friendly sizing).

    Returns:
        ``"smoke"`` or ``"full"``.

    Raises:
        ValueError: If the env var is set to an unrecognised value.
    """
    scale = os.environ.get("ROCM_TEST_WORKLOAD_SCALE", "smoke").strip().lower()
    if scale not in _VALID_WORKLOAD_SCALES:
        raise ValueError(f"ROCM_TEST_WORKLOAD_SCALE must be one of {_VALID_WORKLOAD_SCALES}, got {scale!r}")
    return scale
