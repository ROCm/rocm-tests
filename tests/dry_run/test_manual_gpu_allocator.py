# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_manual_gpu_allocator.py -- DryRun tests for the manual_gpu_allocator fixture.

All tests run without GPU hardware (--no-gpu / hw.cpu_only, ci.pr).
"""

import pytest

# ---------------------------------------------------------------------------
# DryRun: available_gpus returns synthetic pool
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_available_gpus(manual_gpu_allocator):
    """In --no-gpu mode, available_gpus returns 2 synthetic GpuInfo objects."""
    gpus = manual_gpu_allocator.available_gpus
    assert len(gpus) == 2
    assert gpus[0].index == 0
    assert gpus[1].index == 1
    assert all(g.arch == "gfx942" for g in gpus)


# ---------------------------------------------------------------------------
# DryRun: pin() context manager yields a working executor
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_pin_context_manager(manual_gpu_allocator):
    """pin() in --no-gpu mode yields a NodeExecutorGroup backed by DryRunExecutor."""
    with manual_gpu_allocator.pin(gpu_index=0) as executor:
        result = executor.run("echo RESULT_OK")
    # DryRunExecutor always returns ok=True and exit_code=0
    assert result.ok
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# DryRun: explicit acquire / release round-trip
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_acquire_release(manual_gpu_allocator):
    """Explicit acquire()/release() round-trip works in --no-gpu mode."""
    group = manual_gpu_allocator.acquire(gpu_index=1)
    assert group is not None
    result = group.run("echo OK")
    assert result.ok
    manual_gpu_allocator.release(group)
    # After release, _held should be empty (DryRun path; no NodeSlot to track)
    assert manual_gpu_allocator._held == []


# ---------------------------------------------------------------------------
# DryRun: double-release is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_double_release_noop(manual_gpu_allocator):
    """Calling release() twice on the same group does not raise."""
    with manual_gpu_allocator.pin(gpu_index=0) as executor:
        result = executor.run("echo OK")
    assert result.ok
    # Second release of the same group — should be a no-op, not raise
    manual_gpu_allocator.release(executor)


# ---------------------------------------------------------------------------
# DryRun: no leak detected when pin() is used correctly
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dry_run_no_leak_with_pin(manual_gpu_allocator):
    """After using pin() as a context manager, _cleanup() finds nothing to complain about."""
    with manual_gpu_allocator.pin(gpu_index=0):
        pass
    # _cleanup() is called by the fixture teardown — we just verify _held is empty now
    assert manual_gpu_allocator._held == []
