# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_ubb_power.py -- UBB power metric validation via amd-smi.

Validates UBB_POWER reporting at idle (per GPU), under CoralGemm load (load > idle),
and node-level THRESHOLD via amd-smi node -p.
"""

from __future__ import annotations

import logging
import re
import time

import pytest

logger = logging.getLogger("rocm.test")

# GPU architectures known to populate UBB_POWER and THRESHOLD fields.
_UBB_SUPPORTED_ARCHS: frozenset[str] = frozenset({"gfx950", "gfx942"})

# CoralGemm workload args matching the original test invocation.
_CORAL_GEMM_ARGS = "R_64F R_64F R_64F R_64F OP_N OP_T 8640 8640 8640 8640 8640 8640 12 300"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_unsupported_arch(gpu_arch: str | None) -> None:
    """Skip gracefully when the GPU architecture does not support UBB fields."""
    if gpu_arch is not None and gpu_arch not in _UBB_SUPPORTED_ARCHS:
        pytest.skip(f"UBB power fields not supported on {gpu_arch} (supported: {sorted(_UBB_SUPPORTED_ARCHS)})")


def _parse_ubb_power(output: str) -> float | None:
    """Return the first numeric UBB_POWER watt value from output, or None if absent/N/A."""
    match = re.search(r"UBB_POWER:\s*([\d.]+)\s*W", output)
    return float(match.group(1)) if match else None


def _parse_threshold(output: str) -> float | None:
    """Return the THRESHOLD watt value from amd-smi node -p output, or None if absent/N/A."""
    match = re.search(r"THRESHOLD:\s*([\d.]+)\s*W", output)
    return float(match.group(1)) if match else None


def _gpu_ids(executor, amd_smi: str) -> list[str]:
    """Return GPU IDs reported by amd-smi list; skips if none found."""
    result = executor.run(f"{amd_smi} list")
    assert result.ok, f"amd-smi list failed:\n{result.stderr[:500]}"
    ids = re.findall(r"GPU:\s*(\d+)", result.stdout or "")
    if not ids:
        pytest.skip("No GPUs reported by amd-smi list")
    return ids


def _gpu_id_for_oam0(executor, amd_smi: str) -> str:
    """Return the GPU ID whose OAM_ID is 0; skips if none found.

    UBB/node power is tied to OAM_ID 0 — querying the wrong GPU ID returns N/A.
    """
    result = executor.run(f"{amd_smi} list -e")
    assert result.ok, f"amd-smi list -e failed:\n{result.stderr[:500]}"

    current_gpu: str | None = None
    for line in (result.stdout or "").splitlines():
        gpu_match = re.search(r"GPU:\s*(\d+)", line)
        if gpu_match:
            current_gpu = gpu_match.group(1)
        if current_gpu and re.search(r"OAM_ID:\s*0\b", line):
            logger.info("_gpu_id_for_oam0: GPU %s has OAM_ID 0", current_gpu)
            return current_gpu

    pytest.skip("No GPU with OAM_ID 0 found via 'amd-smi list -e' — UBB power not available on this node")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.runtime.fast
def test_ubb_power_default(target_executor, ubb_env, gpu_arch: str | None) -> None:
    """Verify UBB_POWER is reported for the GPU with OAM_ID 0 at idle.

    UBB power is tied to OAM_ID 0; querying other GPUs returns N/A.
    Fails if UBB_POWER field is absent or N/A for that GPU.
    """
    _skip_unsupported_arch(gpu_arch)

    logger.info("test_ubb_power_default: finding GPU with OAM_ID 0")
    gpu_id = _gpu_id_for_oam0(target_executor, ubb_env.amd_smi)

    cmd = f"{ubb_env.amd_smi} metric --power -g {gpu_id}"
    logger.info("test_ubb_power_default: running '%s'", cmd)
    result = target_executor.run(cmd)
    assert result.ok, f"amd-smi metric --power -g {gpu_id} failed:\n{result.stderr[:500]}"

    watts = _parse_ubb_power(result.stdout or "")
    assert watts is not None, f"GPU {gpu_id} (OAM_ID 0): UBB_POWER absent or N/A in output:\n{result.stdout[:500]}"
    logger.info("test_ubb_power_default: PASS — GPU %s (OAM_ID 0) UBB_POWER = %.1f W", gpu_id, watts)


@pytest.mark.runtime.medium
def test_ubb_power_workload(
    target_executor,
    ubb_env,
    coral_gemm_binary: str,
    gpu_arch: str | None,
) -> None:
    """Verify UBB_POWER under CoralGemm load exceeds idle baseline for the OAM_ID 0 GPU.

    Resolves the correct GPU via OAM_ID 0, captures idle UBB_POWER, launches CoralGemm,
    then polls five times asserting load > idle on every reading.
    """
    _skip_unsupported_arch(gpu_arch)

    logger.info("test_ubb_power_workload: finding GPU with OAM_ID 0")
    gpu_id = _gpu_id_for_oam0(target_executor, ubb_env.amd_smi)
    cmd = f"{ubb_env.amd_smi} metric --power -g {gpu_id}"

    logger.info("test_ubb_power_workload: capturing idle UBB_POWER for GPU %s (OAM_ID 0)", gpu_id)
    idle_result = target_executor.run(cmd)
    assert idle_result.ok, f"amd-smi metric --power failed at idle:\n{idle_result.stderr[:500]}"

    idle_watts = _parse_ubb_power(idle_result.stdout or "")
    if idle_watts is None:
        pytest.skip("UBB_POWER not populated at idle — hardware may not support this field")
    logger.info("test_ubb_power_workload: idle UBB_POWER = %.1f W", idle_watts)

    workload_cmd = f"{coral_gemm_binary} {_CORAL_GEMM_ARGS}"
    logger.info("test_ubb_power_workload: launching CoralGemm workload")

    with target_executor.start_background(
        workload_cmd,
        log_path="output/artifacts/executor-logs/test_ubb_power_workload__coral_gemm.log",
        console_label="coral-gemm",
    ) as workload:
        time.sleep(5)  # ramp-up: let the GPU reach operating power before polling
        assert workload.is_alive, "CoralGemm workload exited before measurement began"
        logger.info("test_ubb_power_workload: workload running — polling UBB_POWER (5 attempts)")

        all_passed = True
        for attempt in range(5):
            if not workload.is_alive:
                logger.info("test_ubb_power_workload: workload finished at attempt %d", attempt)
                break

            poll = target_executor.run(cmd)
            if not poll.ok:
                logger.warning("test_ubb_power_workload: poll %d failed (exit=%d)", attempt, poll.exit_code)
                all_passed = False
                continue

            load_watts = _parse_ubb_power(poll.stdout or "")
            if load_watts is None:
                logger.warning("test_ubb_power_workload: UBB_POWER missing in poll %d output", attempt)
                all_passed = False
                continue

            logger.info(
                "test_ubb_power_workload: attempt %d — idle=%.1f W  load=%.1f W",
                attempt,
                idle_watts,
                load_watts,
            )
            if load_watts <= idle_watts:
                logger.error(
                    "test_ubb_power_workload: load %.1f W not greater than idle %.1f W at attempt %d",
                    load_watts,
                    idle_watts,
                    attempt,
                )
                all_passed = False

            time.sleep(2)

    assert all_passed, f"GPU {gpu_id}: UBB_POWER under load never exceeded idle baseline of {idle_watts:.1f} W"
    logger.info("test_ubb_power_workload: PASS — load UBB_POWER exceeded idle on GPU %s", gpu_id)


@pytest.mark.runtime.fast
def test_ubb_threshold(target_executor, ubb_env, gpu_arch: str | None) -> None:
    """Verify node-level UBB power THRESHOLD is reported by amd-smi node -p.

    Asserts THRESHOLD field is present and holds a positive numeric watt value.
    """
    _skip_unsupported_arch(gpu_arch)

    cmd = f"{ubb_env.amd_smi} node -p"
    logger.info("test_ubb_threshold: running '%s'", cmd)
    result = target_executor.run(cmd)
    assert result.ok, f"amd-smi node -p failed:\n{result.stderr[:500]}"

    if "THRESHOLD" not in (result.stdout or ""):
        pytest.skip("THRESHOLD field absent in 'amd-smi node -p' — hardware may not support it")

    threshold_w = _parse_threshold(result.stdout or "")
    assert threshold_w is not None, "THRESHOLD field present but value is N/A or non-numeric"
    assert threshold_w > 0, f"THRESHOLD reported as {threshold_w} W — expected a positive value"

    logger.info("test_ubb_threshold: PASS — THRESHOLD = %.1f W", threshold_w)
