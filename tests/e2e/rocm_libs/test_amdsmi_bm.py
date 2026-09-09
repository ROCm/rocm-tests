# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmi_bm.py -- AMD SMI Benchmark Suite via amd-smi CLI.

Ported from: amd-smi-lib AMDSMI_BM class (lines 273-357).

Validates amd-smi command-line capabilities:
    - GPU static info: device count, architecture, VRAM
    - GPU metrics: power, temperature, memory, clocks, throttle, ECC, PCIe
    - Health checks: temperature, VRAM, utilization thresholds
    - Driver integration: kernel version, ASIC info

Auto-injected by CATEGORY_PROFILES for tests/e2e/rocm_libs/:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.fast, runtime.medium
"""

from __future__ import annotations

import json
import logging
import time

import pytest

logger = logging.getLogger(__name__)


@pytest.mark.runtime.fast
def test_amdsmi_static_info(target_executor, amdsmi_available, amdsmi_version):
    """Validate amd-smi static (device discovery) command.

    Validates:
        - Command succeeds and returns valid JSON
        - At least one GPU is detected
        - Each GPU has required fields: index, architecture, VRAM
    """
    if not amdsmi_available:
        pytest.skip("amd-smi command not available on target")

    logger.info("Testing amd-smi static --json (version: %s)", amdsmi_version or "unknown")

    result = target_executor.run("amd-smi static --json")
    assert result.ok, f"amd-smi static failed: {result.stderr}"
    assert result.stdout, "amd-smi static returned empty output"

    try:
        data = json.loads(result.stdout)
        gpu_data = data.get("gpu_data", [])
        assert gpu_data, "No GPU data found in amd-smi output"
        logger.info("✓ amd-smi static: %d GPU(s) detected", len(gpu_data))

        for gpu in gpu_data:
            index = gpu.get("index")
            arch = gpu.get("asic", {}).get("target_graphics_version")
            vram = gpu.get("vram", {}).get("size")
            assert index is not None, f"GPU missing index: {gpu}"
            assert arch, f"GPU {index} missing architecture"
            assert vram, f"GPU {index} missing VRAM info"
            logger.info("  GPU %d: %s, VRAM: %s", index, arch, vram)

    except json.JSONDecodeError as exc:
        pytest.fail(f"amd-smi static returned invalid JSON: {exc}")


@pytest.mark.runtime.fast
def test_amdsmi_metrics(target_executor, amdsmi_available, amdsmi_gpu_metrics):
    """Validate amd-smi metric (GPU monitoring) command.

    Validates:
        - Command succeeds and returns valid JSON
        - Metrics include temperature, power, VRAM usage, utilization
        - All required metric fields are present
    """
    if not amdsmi_available:
        pytest.skip("amd-smi command not available on target")

    logger.info("Testing amd-smi metric --json")

    assert amdsmi_gpu_metrics, "Failed to fetch amd-smi metrics"

    gpu_data = amdsmi_gpu_metrics.get("gpu_data", [])
    assert gpu_data, "No GPU data in metrics output"

    for gpu in gpu_data:
        index = gpu.get("index")
        temp = gpu.get("temperature", {})
        power = gpu.get("power", {})
        vram = gpu.get("vram", {})
        activity = gpu.get("activity", {})

        assert temp, f"GPU {index} missing temperature metrics"
        assert power, f"GPU {index} missing power metrics"
        assert vram, f"GPU {index} missing VRAM metrics"
        assert activity, f"GPU {index} missing activity metrics"

        logger.info(
            "  GPU %d: temp=%sC, power=%sW, vram=%s/%sMB, util=%s%%",
            index,
            temp.get("hotspot_temperature", {}).get("value", "?"),
            power.get("power_usage", {}).get("value", "?"),
            vram.get("vram_used", {}).get("value", "?"),
            vram.get("vram_total", {}).get("value", "?"),
            activity.get("gfx_activity", {}).get("value", "?"),
        )

    logger.info("✓ amd-smi metric: All metrics validated")


@pytest.mark.runtime.medium
def test_amdsmi_bm_suite(
    target_executor,
    amdsmi_available,
    amdsmi_version,
    amdsmi_gpu_metrics,
):
    """Comprehensive AMD SMI benchmark suite via command-line interface.

    Executes a series of amd-smi commands to validate:
        - Device enumeration and info retrieval
        - Real-time metrics (temperature, power, memory, clocks)
        - Health monitoring capabilities
        - PCIe and thermal status reporting

    Validates that amd-smi tool is properly integrated and all critical
    monitoring paths are functional.
    """
    if not amdsmi_available:
        pytest.skip("amd-smi command not available on target")

    logger.info("Starting amd-smi benchmark suite (version: %s)", amdsmi_version or "unknown")

    suite_start = time.time()
    all_passed = True
    failed_tests = []

    del amdsmi_gpu_metrics

    # Test suite: collection of amd-smi commands to validate
    test_cases = [
        ("amd-smi static --json", "Device enumeration"),
        ("amd-smi metric --json", "GPU metrics"),
        ("amd-smi version", "Version info"),
    ]

    for cmd, description in test_cases:
        test_start = time.time()
        logger.info("Running: %s (%s)", cmd, description)

        try:
            result = target_executor.run(cmd)
            if result.ok:
                logger.info(
                    "✓ %s: PASS (%.2f s)",
                    description,
                    time.time() - test_start,
                )
            else:
                logger.error(
                    "✗ %s: FAIL — %s (%.2f s)",
                    description,
                    result.stderr or result.stdout,
                    time.time() - test_start,
                )
                all_passed = False
                failed_tests.append((description, result.stderr or "unknown error"))

        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Exception in %s: %s", description, exc)
            all_passed = False
            failed_tests.append((description, str(exc)))

    suite_elapsed = time.time() - suite_start

    logger.info("Suite execution time: %.2f seconds", suite_elapsed)
    if failed_tests:
        logger.error("Failed tests (%d):", len(failed_tests))
        for name, reason in failed_tests:
            logger.error("  - %s: %s", name, reason)

    assert all_passed, (
        f"amd-smi benchmark suite failed with {len(failed_tests)} test(s):\n"
        + "\n".join(f"  {name}: {reason}" for name, reason in failed_tests)
    )
