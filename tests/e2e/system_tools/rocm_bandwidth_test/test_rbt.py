# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validates rocm-bandwidth-test: plugin listing, DMA bandwidth, P2P, scaling, schmoo, sweep, and health-check."""

from __future__ import annotations

import pytest

_FAIL_PATTERNS = ["file not found", "plugin not found"]


def _assert_no_failure_patterns(result, label: str) -> None:
    """Assert result.ok and no known failure string in stdout+stderr (case-insensitive)."""
    assert result.ok, f"{label}: exited {result.exit_code}\nstdout: {result.stdout[:2000]}"
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
    for pattern in _FAIL_PATTERNS:
        assert pattern not in combined, f"{label}: output contains {pattern!r}\n{(result.stdout or '')[:2000]}"


def _detect_gpu_count(target_executor, rbt: str) -> int:
    """Count GPU agents from ``rocm-bandwidth-test list``; returns 0 on error."""
    result = target_executor.run(f"{rbt} list")
    return sum(1 for line in (result.stdout or "").splitlines() if "Gpu" in line) if result.ok else 0


def _detect_mi300(target_executor, rbt: str) -> bool:
    """Return True if at least one MI300 GPU appears in the agent list."""
    result = target_executor.run(f"{rbt} list")
    return result.ok and "MI300" in (result.stdout or "")


def _run(target_executor, rbt: str, ld_path: dict, args: str, timeout: int):
    ld = ld_path["LD_LIBRARY_PATH"]
    return target_executor.run(f"env LD_LIBRARY_PATH={ld} {rbt} {args}", timeout=timeout)


@pytest.mark.runtime.fast
def test_rbt_plugin(target_executor, rbt_binary: str, ld_path: dict):
    """List all available plugins via ``rocm-bandwidth-test plugin -i``."""
    result = _run(target_executor, rbt_binary, ld_path, "plugin -i", timeout=120)
    _assert_no_failure_patterns(result, "rbt plugin -i")


@pytest.mark.runtime.medium
def test_rbt_tb(target_executor, rbt_binary: str, ld_path: dict):
    """Transaction-bandwidth baseline across all CPU/GPU agent pairs."""
    result = _run(target_executor, rbt_binary, ld_path, "run tb", timeout=900)
    _assert_no_failure_patterns(result, "rbt run tb")


@pytest.mark.runtime.medium
def test_rbt_tb_p2p(target_executor, rbt_binary: str, ld_path: dict):
    """GPU-to-GPU peer-to-peer DMA bandwidth."""
    result = _run(target_executor, rbt_binary, ld_path, "run tb p2p", timeout=900)
    _assert_no_failure_patterns(result, "rbt run tb p2p")


@pytest.mark.runtime.medium
def test_rbt_tb_scaling(target_executor, rbt_binary: str, ld_path: dict):
    """Bandwidth scaling across transfer sizes for all agent pairs."""
    result = _run(target_executor, rbt_binary, ld_path, "run tb scaling", timeout=900)
    _assert_no_failure_patterns(result, "rbt run tb scaling")


@pytest.mark.runtime.medium
def test_rbt_tb_schmoo(target_executor, rbt_binary: str, ld_path: dict):
    """Dense transfer-size/concurrency schmoo; requires at least 2 GPU agents."""
    if _detect_gpu_count(target_executor, rbt_binary) < 2:
        pytest.skip("test_rbt_tb_schmoo requires at least 2 GPU agents — skipping on single-GPU platform.")
    result = _run(target_executor, rbt_binary, ld_path, "run tb schmoo", timeout=1800)
    _assert_no_failure_patterns(result, "rbt run tb schmoo")


@pytest.mark.runtime.medium
def test_rbt_tb_sweep(target_executor, rbt_binary: str, ld_path: dict):
    """Full DMA sweep bounded by SWEEP_TIME_LIMIT=10s."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} SWEEP_TIME_LIMIT=10 {rbt_binary} run tb sweep", timeout=1800
    )
    _assert_no_failure_patterns(result, "rbt run tb sweep")


@pytest.mark.runtime.medium
def test_rbt_tb_rsweep(target_executor, rbt_binary: str, ld_path: dict):
    """Reverse DMA sweep bounded by SWEEP_TIME_LIMIT=10s."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} SWEEP_TIME_LIMIT=10 {rbt_binary} run tb rsweep", timeout=1800
    )
    _assert_no_failure_patterns(result, "rbt run tb rsweep")


@pytest.mark.runtime.medium
def test_rbt_tb_one2all(target_executor, rbt_binary: str, ld_path: dict):
    """One-to-all multi-GPU bandwidth with NUM_GPU_DEVICES=2; requires at least 2 GPU agents."""
    if _detect_gpu_count(target_executor, rbt_binary) < 2:
        pytest.skip("test_rbt_tb_one2all requires at least 2 GPU agents — skipping on single-GPU platform.")
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} NUM_GPU_DEVICES=2 {rbt_binary} run tb one2all", timeout=1800
    )
    _assert_no_failure_patterns(result, "rbt run tb one2all")


@pytest.mark.runtime.fast
def test_rbt_healthcheck(target_executor, rbt_binary: str, ld_path: dict):
    """GPU health-check sub-test; skipped on non-MI300 platforms."""
    if not _detect_mi300(target_executor, rbt_binary):
        pytest.skip("test_rbt_healthcheck targets MI300-class GPUs — no MI300 detected.")
    result = _run(target_executor, rbt_binary, ld_path, "run tb healthcheck", timeout=300)
    _assert_no_failure_patterns(result, "rbt run tb healthcheck")
