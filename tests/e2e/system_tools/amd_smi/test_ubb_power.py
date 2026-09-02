# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_ubb_power.py -- amd-smi power metric validation.

Validates amd-smi UBB_POWER field reporting at idle (per GPU), under CoralGemm load
(load > idle), and node-level THRESHOLD via amd-smi node -p.

UBB_POWER is only populated for the GPU whose OAM_ID is 0. On multi-OAM systems the
logical GPU ID (e.g. -g 0) does not always map to OAM slot 0, so all tests that read
UBB_POWER first resolve the correct GPU ID via 'amd-smi list -e'.
"""

from __future__ import annotations

import logging
import pathlib
import re
import time

import pytest

logger = logging.getLogger("rocm.test")

# GPU architectures known to report the UBB_POWER and THRESHOLD fields via amd-smi.
# gfx942 (MI300X) is explicitly included alongside gfx950.
_UBB_SUPPORTED_ARCHS: frozenset[str] = frozenset({"gfx950", "gfx942"})

# CoralGemm workload args matching the original test invocation.
_CORAL_GEMM_ARGS = "R_64F R_64F R_64F R_64F OP_N OP_T 8640 8640 8640 8640 8640 8640 12 300"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_unsupported_arch(gpu_arch: str | None) -> None:
    """Skip when the GPU architecture is not known to report amd-smi power fields."""
    if gpu_arch is not None and gpu_arch not in _UBB_SUPPORTED_ARCHS:
        pytest.skip(f"amd-smi power fields not supported on {gpu_arch} (supported: {sorted(_UBB_SUPPORTED_ARCHS)})")


def _parse_ubb_power(output: str) -> float | None:
    """Return the first numeric value for the UBB_POWER field from amd-smi output, or None if absent/N/A."""
    match = re.search(r"UBB_POWER:\s*([\d.]+)\s*W", output)
    return float(match.group(1)) if match else None


def _parse_threshold(output: str) -> float | None:
    """Return the THRESHOLD watt value from amd-smi node -p output, or None if absent/N/A."""
    match = re.search(r"THRESHOLD:\s*([\d.]+)\s*W", output)
    return float(match.group(1)) if match else None


def _gpu_id_for_oam0(executor, amd_smi: str) -> str:
    """Return the logical GPU ID whose OAM_ID is 0, as reported by 'amd-smi list -e'.

    UBB_POWER and node-level power fields are only populated for the GPU with OAM_ID 0.
    On multi-OAM systems the logical GPU ID does not always match OAM slot 0, so this
    lookup is required before querying UBB_POWER.
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

    pytest.fail("No GPU with OAM_ID 0 found via 'amd-smi list -e' — OAM_ID 0 is required for UBB power reporting")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.runtime.fast
def test_ubb_power_default(target_executor, ubb_env, gpu_arch: str | None) -> None:
    """Verify amd-smi reports a numeric UBB_POWER value at idle for the GPU with OAM_ID 0.

    Resolves the correct GPU via OAM_ID 0 before querying, since UBB_POWER is only
    populated for that GPU; querying any other GPU ID returns N/A.
    Fails if the UBB_POWER field is absent or N/A in the output.
    """
    _skip_unsupported_arch(gpu_arch)

    logger.info("test_ubb_power_default: finding GPU with OAM_ID 0")
    gpu_id = _gpu_id_for_oam0(target_executor, ubb_env.amd_smi)

    cmd = f"{ubb_env.amd_smi} metric --power -g {gpu_id}"
    logger.info("test_ubb_power_default: running '%s'", cmd)
    result = target_executor.run(cmd)
    assert result.ok, f"amd-smi metric --power -g {gpu_id} failed:\n{result.stderr[:500]}"

    watts = _parse_ubb_power(result.stdout or "")
    assert (
        watts is not None
    ), f"GPU {gpu_id} (OAM_ID 0): UBB_POWER absent or N/A in output:\n{(result.stdout or '')[:500]}"
    logger.info("test_ubb_power_default: PASS — GPU %s (OAM_ID 0) UBB_POWER = %.1f W", gpu_id, watts)


@pytest.mark.runtime.medium
def test_ubb_power_workload(
    target_executor,
    ubb_env,
    coral_gemm_binary: str,
    rock_dir: str,
    gpu_arch: str | None,
) -> None:
    """Verify amd-smi UBB_POWER under CoralGemm load exceeds the idle baseline.

    Resolves the GPU with OAM_ID 0, captures idle UBB_POWER, launches CoralGemm,
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
    assert (
        idle_watts is not None
    ), f"GPU {gpu_id} (OAM_ID 0): UBB_POWER absent or N/A at idle:\n{(idle_result.stdout or '')[:500]}"
    logger.info("test_ubb_power_workload: idle UBB_POWER = %.1f W", idle_watts)

    rocm_path = rock_dir or "/opt/rocm"
    gemm_abs = str(pathlib.Path(coral_gemm_binary).resolve())
    gemm_dir = str(pathlib.Path(gemm_abs).parent)
    workload_cmd = (
        f"cd {gemm_dir} && "
        f"HIP_PLATFORM=amd ROCM_PATH={rocm_path} "
        f"LD_LIBRARY_PATH={rocm_path}/lib:${{LD_LIBRARY_PATH:-}} "
        f"PATH={rocm_path}/bin:$PATH "
        f"{gemm_abs} {_CORAL_GEMM_ARGS}"
    )
    logger.info("test_ubb_power_workload: launching CoralGemm workload from %s", gemm_dir)

    with target_executor.start_background(
        workload_cmd,
        log_path="output/artifacts/executor-logs/test_ubb_power_workload__coral_gemm.log",
        console_label="coral-gemm",
    ) as workload:
        time.sleep(5)  # ramp-up: let the GPU reach operating power before polling
        assert workload.is_alive, "CoralGemm workload exited before measurement began"
        logger.info("test_ubb_power_workload: workload running — polling UBB_POWER (5 attempts)")

        any_exceeded = False
        for attempt in range(5):
            if not workload.is_alive:
                logger.info("test_ubb_power_workload: workload finished at attempt %d", attempt)
                break

            poll = target_executor.run(cmd)
            if not poll.ok:
                logger.warning("test_ubb_power_workload: poll %d failed (exit=%d)", attempt, poll.exit_code)
                time.sleep(2)
                continue

            load_watts = _parse_ubb_power(poll.stdout or "")
            if load_watts is None:
                logger.warning("test_ubb_power_workload: UBB_POWER missing in poll %d output", attempt)
                time.sleep(2)
                continue

            logger.info(
                "test_ubb_power_workload: attempt %d — idle=%.1f W  load=%.1f W",
                attempt,
                idle_watts,
                load_watts,
            )
            if load_watts > idle_watts:
                any_exceeded = True
                logger.info(
                    "test_ubb_power_workload: attempt %d PASS — load %.1f W > idle %.1f W",
                    attempt,
                    load_watts,
                    idle_watts,
                )
            else:
                logger.warning(
                    "test_ubb_power_workload: attempt %d — load %.1f W not yet above idle %.1f W",
                    attempt,
                    load_watts,
                    idle_watts,
                )

            time.sleep(2)  # allow power to stabilise between samples

    assert any_exceeded, f"GPU {gpu_id}: UBB_POWER never exceeded idle baseline of {idle_watts:.1f} W across 5 polls"
    logger.info("test_ubb_power_workload: PASS — load UBB_POWER exceeded idle on GPU %s", gpu_id)


@pytest.mark.runtime.fast
def test_ubb_threshold(target_executor, ubb_env, gpu_arch: str | None) -> None:
    """Verify amd-smi node -p reports a positive numeric THRESHOLD value.

    Asserts the THRESHOLD field is present and holds a positive watt value.
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
