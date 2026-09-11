# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Fixtures for AMD SMI gtest suite.

Provides: ``amdsmitst_binary`` (function-scoped fixture that locates the pre-installed
amdsmitst gtest binary on the node selected for the current test).
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


def _locate_amdsmitst_binary(executor, rock_dir: str) -> str:
    """Locate the amdsmitst gtest binary from rock_dir.

    The amdsmitst binary is pre-installed by the AMD SMI library (amd-smi-lib)
    into the ROCm share directory. We check two canonical locations in order:
        1. <rock_dir>/share/amd_smi/tests/amdsmitst
        2. <rock_dir>/share/amd_smi/amdsmitst

    Args:
        executor: NodeExecutorGroup or compatible executor to run commands on the test node.
        rock_dir: Resolved path to the TheRock/ROCm installation.

    Returns:
        Absolute path to the amdsmitst binary.

    Raises:
        pytest.fail.Exception: When amdsmitst is not found at either location.
    """
    candidates = [
        f"{rock_dir}/share/amd_smi/tests/amdsmitst",
        f"{rock_dir}/share/amd_smi/amdsmitst",
    ]

    for candidate in candidates:
        probe = executor.run(f"test -x {candidate} && echo FOUND")
        if (probe.stdout or "").strip() == "FOUND":
            logger.info("amdsmitst_binary: located at %s", candidate)
            return candidate

    pytest.fail(
        f"amdsmitst gtest binary not found in rock_dir={rock_dir} — "
        f"checked: {', '.join(candidates)}. "
        f"Ensure amd-smi-lib is installed with TheRock."
    )


@pytest.fixture
def amdsmitst_binary(target_executor, rock_dir: str) -> str:
    """Resolve the amdsmitst gtest binary from the ROCm install.

    The amdsmitst binary is a gtest suite that validates the amd-smi library's
    core functionality across all supported AMD GPU types. It exits 0 when all
    test cases pass and prints gtest PASSED/FAILED markers to stdout.

    Args:
        target_executor: NodeExecutorGroup for running commands on the test node.
        rock_dir: Resolved ROCm installation path from builder_plugin.

    Returns:
        Absolute path to the amdsmitst binary on the test node.
    """
    return _locate_amdsmitst_binary(target_executor, rock_dir)


# ============================================================================
# AMD SMI Library Benchmark Suite Fixtures
# ============================================================================

# Full test list for amd-smi-lib benchmark suite.
# Each test name maps to a callable in the imported amd-smi-lib test module.
AMDSMI_FULL_TESTS = [
    "AMDSMI_version",
    "AMDSMI_list",
    "AMDSMI_list_json",
    "AMDSMI_list_csv",
    "AMDSMI_static",
    "AMDSMI_static_asic",
    "AMDSMI_static_json",
    "AMDSMI_static_csv",
    "AMDSMI_static_file",
    "AMDSMI_static_json_file",
    "AMDSMI_static_csv_file",
    "AMDSMI_metric",
    "AMDSMI_metric_overdrive",
    "AMDSMI_metric_json",
    "AMDSMI_metric_csv",
    "AMDSMI_metric_watch1",
    "AMDSMI_metric_watch1_watchtime3",
    "AMDSMI_metric_watch1_iterations3",
    "AMDSMI_metric_watch1_json",
    "AMDSMI_process",
    "AMDSMI_process_with_workload",
    "AMDSMI_process_json",
    "AMDSMI_process_csv",
    "AMDSMI_process_watch1",
    "AMDSMI_process_watch1_watchtime3",
    "AMDSMI_process_watch1_iterations3",
    "AMDSMI_process_watch1_json",
    "AMDSMI_process_watch1_csv",
    "AMDSMI_firmware",
    "AMDSMI_firmware_json",
    "AMDSMI_firmware_csv",
    "AMDSMI_badpages",
    "AMDSMI_badpages_json",
    "AMDSMI_badpages_csv",
    "AMDSMI_set_fan_speed",
    "AMDSMI_reset_fan_speed",
    "AMDSMI_monitor_without_workload",
    "AMDSMI_monitor_with_workload",
    "AMDSMI_monitor_without_workload_with_arguments",
    "AMDSMI_monitor_with_workload_with_arguments",
    "AMDSMI_gpureset",
    "AMDSMI_gpuresetall",
    "AMDSMI_set_powercap",
    "AMDSMI_power_cap_above_max",
    "AMDSMI_monitor_qt",
    "AMDSMI_monitor_qt_workload",
    "AMDSMI_monitor_qt_json",
    "AMDSMI_monitor_qt_json_workload",
    "AMDSMI_monitor_qt_csv",
    "AMDSMI_monitor_qt_csv_workload",
    "AMDSMI_monitor_qt_file",
    "AMDSMI_monitor_qt_file_workload",
    "AMDSMI_monitor_qt_file_json",
    "AMDSMI_monitor_qt_file_json_workload",
    "AMDSMI_monitor_qt_file_csv",
    "AMDSMI_monitor_qt_file_csv_workload",
    "AMDSMI_monitor_qt_w",
    "AMDSMI_monitor_qt_w_workload",
    "AMDSMI_monitor_qt_w_json",
    "AMDSMI_monitor_qt_w_json_workload",
    "AMDSMI_monitor_qt_w_csv",
    "AMDSMI_monitor_qt_w_csv_workload",
    "AMDSMI_monitor_qt_w_file",
    "AMDSMI_monitor_qt_w_file_workload",
    "AMDSMI_monitor_qt_w_file_json",
    "AMDSMI_monitor_qt_w_file_json_workload",
    "AMDSMI_monitor_qt_w_file_csv",
    "AMDSMI_monitor_qt_w_file_csv_workload",
    "AMDSMI_monitor_qt_gpu",
    "AMDSMI_monitor_qt_gpu_workload",
    "AMDSMI_monitor_qt_gpu_json",
    "AMDSMI_monitor_qt_gpu_json_workload",
    "AMDSMI_monitor_qt_gpu_csv",
    "AMDSMI_monitor_qt_gpu_csv_workload",
    "AMDSMI_monitor_qt_gpu_file",
    "AMDSMI_monitor_qt_gpu_file_workload",
    "AMDSMI_monitor_qt_gpu_file_json",
    "AMDSMI_monitor_qt_gpu_file_json_workload",
    "AMDSMI_monitor_qt_gpu_file_csv",
    "AMDSMI_monitor_qt_gpu_file_csv_workload",
    "AMDSMI_monitor_qt_gpu_w",
    "AMDSMI_monitor_qt_gpu_w_workload",
    "AMDSMI_monitor_qt_gpu_w_json",
    "AMDSMI_monitor_qt_gpu_w_json_workload",
    "AMDSMI_monitor_qt_gpu_w_csv",
    "AMDSMI_monitor_qt_gpu_w_csv_workload",
    "AMDSMI_monitor_qt_gpu_w_file",
    "AMDSMI_monitor_qt_gpu_w_file_workload",
    "AMDSMI_monitor_qt_gpu_w_file_json",
    "AMDSMI_monitor_qt_gpu_w_file_json_workload",
    "AMDSMI_monitor_qt_gpu_w_file_csv",
    "AMDSMI_monitor_qt_gpu_w_file_csv_workload",
    "AMDSMI_xgmi_plpd",
    "amdsmi_GPU_driver_version",
    "amd_smi_set_gfx_clock",
    "amd_smi_isolation",
    "amd_smi_gfxVersion",
    "AMDSMI_detect_PassThrough",
    "AMDSMI_static_clock",
    "AMDSMI_static_clock_SOC",
    "AMDSMI_metric_c",
    "AMDSMI_metric_P",
    "amd_smi_Flat_Process",
    "amdsmi_PLDM",
    "amd_smi_voltage_vddboard",
    "AMDSMI_cpu_affinity",
    "AMDSMI_socket_affinity",
    "AMDSMI_xgmi_custom_gpus",
    "AMDSMI_node_power_management",
    "AMDSMI_ptl_status",
    "AMDSMI_ptl_format",
    "AMDSMI_metric_time",
    "AMDSMI_set_mclk",
    "AMDSMI_power_profile_set_get_workload",
    "AMDSMI_power_profile_perf_comparison",
]


def _install_amdsmi(executor) -> None:
    """Install amd-smi-lib package via pip on the test node.

    Uses target_executor to handle both local and remote-node modes.

    Args:
        executor: NodeExecutorGroup for running commands on the test node.

    Raises:
        RuntimeError: When pip install fails.
    """
    logger.info("Installing amd-smi-lib...")
    result = executor.run("pip install amd-smi-lib", timeout=300)
    if not result.ok:
        logger.error("amd-smi-lib installation failed: %s", result.stderr)
        raise RuntimeError(f"Failed to install amd-smi-lib: {result.stderr}")
    logger.info("amd-smi-lib installed successfully.")


@pytest.fixture
def amdsmi_installed(target_executor) -> None:
    """Session-scoped fixture: install amd-smi-lib once at the start of the test suite.

    Args:
        target_executor: NodeExecutorGroup for running commands on the test node.
    """
    _install_amdsmi(target_executor)


@pytest.fixture(scope="session")
def amdsmi_testlist() -> list[str]:
    """Session-scoped fixture: return the full amd-smi-lib benchmark test list."""
    return AMDSMI_FULL_TESTS.copy()


@pytest.fixture(scope="session")
def amdsmi_app_version(amdsmi_installed) -> str | None:
    """Session-scoped fixture: fetch the installed amd-smi-lib version string.

    Returns:
        Version string (e.g. "1.0.0"), or None if not available.
    """
    del amdsmi_installed
    try:
        # Import SystemInfo from amd-smi-lib to fetch the app version
        from amdsmi_lib import SystemInfo  # pylint: disable=import-outside-toplevel

        sys_info = SystemInfo(password=None)
        version = sys_info.get_app_version(packages=["amd-smi-lib"]).get("amd-smi-lib")
        logger.info("amd-smi-lib version: %s", version)
        return version
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to retrieve amd-smi-lib version: %s", exc)
        return None
