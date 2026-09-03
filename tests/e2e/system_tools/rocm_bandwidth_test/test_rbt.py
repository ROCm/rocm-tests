# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""rocm-bandwidth-test validation: plugin listing, DMA bandwidth, and GPU health-check."""

from __future__ import annotations

import pytest

_FAIL_PATTERNS = ["file not found", "plugin not found"]


def _assert_no_failure_patterns(result, label: str) -> None:
    """Fail if result is non-zero or a known failure string appears in stdout or stderr."""
    assert result.ok, f"{label}: exited {result.exit_code}\nstdout: {(result.stdout or '')[:2000]}"
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).lower()
    for pattern in _FAIL_PATTERNS:
        assert pattern not in combined, f"{label}: output contains {pattern!r}\n{(result.stdout or '')[:2000]}"


def _detect_gpu_count(target_executor, rbt: str) -> int:
    """Return the number of GPU agents reported by rocm-bandwidth-test list."""
    result = target_executor.run(f"{rbt} list")
    return sum(1 for line in (result.stdout or "").splitlines() if "Gpu" in line) if result.ok else 0


def _detect_mi300(target_executor, rbt: str) -> bool:
    """Return True when at least one MI300 GPU appears in rocm-bandwidth-test list output."""
    result = target_executor.run(f"{rbt} list")
    return result.ok and "MI300" in (result.stdout or "")


def _run(target_executor, rbt: str, ld_path: dict, args: str, timeout: int):
    ld = ld_path["LD_LIBRARY_PATH"]
    return target_executor.run(f"env LD_LIBRARY_PATH={ld} {rbt} {args}", timeout=timeout)


@pytest.mark.runtime.fast
def test_rbt_plugin(target_executor, rbt_binary: str, ld_path: dict):
    """Verify rocm-bandwidth-test loads and lists all available plugins."""
    result = _run(target_executor, rbt_binary, ld_path, "plugin -i", timeout=120)
    _assert_no_failure_patterns(result, "rbt plugin -i")


@pytest.mark.runtime.medium
def test_rbt_tb(target_executor, rbt_binary: str, ld_path: dict):
    """Measure transaction-bandwidth baseline across all CPU and GPU agent pairs."""
    result = _run(target_executor, rbt_binary, ld_path, "run tb", timeout=900)
    _assert_no_failure_patterns(result, "rbt run tb")


@pytest.mark.runtime.medium
def test_rbt_tb_p2p(target_executor, rbt_binary: str, ld_path: dict):
    """Measure GPU-to-GPU peer-to-peer DMA bandwidth."""
    result = _run(target_executor, rbt_binary, ld_path, "run tb p2p", timeout=900)
    _assert_no_failure_patterns(result, "rbt run tb p2p")


@pytest.mark.runtime.medium
def test_rbt_tb_scaling(target_executor, rbt_binary: str, ld_path: dict):
    """Measure how DMA bandwidth scales with transfer size across all agent pairs."""
    result = _run(target_executor, rbt_binary, ld_path, "run tb scaling", timeout=900)
    _assert_no_failure_patterns(result, "rbt run tb scaling")


@pytest.mark.runtime.medium
def test_rbt_tb_schmoo(target_executor, rbt_binary: str, ld_path: dict):
    """Iterate over a dense grid of transfer sizes and concurrency counts across GPU agents."""
    if _detect_gpu_count(target_executor, rbt_binary) < 2:
        pytest.skip("requires at least 2 GPU agents — skipping on single-GPU platform.")
    result = _run(target_executor, rbt_binary, ld_path, "run tb schmoo", timeout=1800)
    _assert_no_failure_patterns(result, "rbt run tb schmoo")


@pytest.mark.runtime.medium
def test_rbt_tb_sweep(target_executor, rbt_binary: str, ld_path: dict):
    """Run a full DMA sweep with SWEEP_TIME_LIMIT=10 to bound wall time."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} SWEEP_TIME_LIMIT=10 {rbt_binary} run tb sweep", timeout=1800
    )
    _assert_no_failure_patterns(result, "rbt run tb sweep")


@pytest.mark.runtime.medium
def test_rbt_tb_rsweep(target_executor, rbt_binary: str, ld_path: dict):
    """Run a reverse DMA sweep with SWEEP_TIME_LIMIT=10 to bound wall time."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} SWEEP_TIME_LIMIT=10 {rbt_binary} run tb rsweep", timeout=1800
    )
    _assert_no_failure_patterns(result, "rbt run tb rsweep")


@pytest.mark.runtime.medium
def test_rbt_tb_one2all(target_executor, rbt_binary: str, ld_path: dict):
    """Fan transfers from one GPU to all others with NUM_GPU_DEVICES=2."""
    if _detect_gpu_count(target_executor, rbt_binary) < 2:
        pytest.skip("requires at least 2 GPU agents — skipping on single-GPU platform.")
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} NUM_GPU_DEVICES=2 {rbt_binary} run tb one2all", timeout=1800
    )
    _assert_no_failure_patterns(result, "rbt run tb one2all")


@pytest.mark.runtime.fast
def test_rbt_healthcheck(target_executor, rbt_binary: str, ld_path: dict):
    """Run the built-in GPU health-check; skipped when no MI300 GPU is detected."""
    if not _detect_mi300(target_executor, rbt_binary):
        pytest.skip("no MI300 GPU detected — health-check sub-test is MI300-only.")
    result = _run(target_executor, rbt_binary, ld_path, "run tb healthcheck", timeout=300)
    _assert_no_failure_patterns(result, "rbt run tb healthcheck")
