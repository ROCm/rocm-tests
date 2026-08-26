# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared PCI device-id → GPU short-name map for RVS config lookup.

Single source of truth consumed by ``gpu_monitored`` workloads, ``rvs.py``,
and ``tests/e2e/rvs/conftest.py``.  Mirrors the vendored
``rvs_config_mapping.csv`` device keys.
"""

from __future__ import annotations

import logging
import re

from framework.executors.local_executor import run_cmd_get_stdout_stderr

logger = logging.getLogger(__name__)

# PCI device ID + revision -> short name (RVS conf subdirectory)
GPU_DEVICE_MAP: dict[str, str] = {
    "66a1_00": "MI50",
    "66a1_06": "MI50",
    "738c_01": "MI100",
    "738c_cc": "MI100",
    "740f_02": "MI210",
    "7410_02": "MI210",
    "740c_01": "MI250",
    "7408_00": "MI250",
    "74a0_00": "MI300A",
    "74b4_00": "MI300A",
    "74a1_00": "MI300X",
    "74b5_00": "MI300X",
    "74a9_00": "MI300X-HF",
    "74bd_00": "MI300X-HF",
    "74a2_00": "MI308X",
    "74b6_00": "MI308X",
    "74a8_00": "MI308X-HF",
    "74bc_00": "MI308X-HF",
    "74a5_00": "MI325X",
    "74b9_00": "MI325X",
    "75a0_00": "MI350X",
    "75b0_00": "MI350X",
    "75a3_00": "MI355X",
    "75b3_00": "MI355X",
    "73a3_00": "nv21",
    "73ae_00": "nv21",
    "7448_00": "nv31",
    "7448_ec": "nv31",
    "744c_c0": "nv31",
    "744c_c8": "nv31",
    "744c_cc": "nv31",
    "744c_ce": "nv31",
    "744c_cf": "nv31",
    "744c_e0": "nv31",
    "744c_ec": "nv31",
    "744c_e8": "nv31",
    "744c_ee": "nv31",
    "745e_cc": "nv31",
    "7449_00": "nv31",
    "744a_00": "nv31",
    "7460_00": "nv32",
    "7461_00": "nv32",
    "7470_00": "nv32",
    "747e_c8": "nv32",
    "747e_c9": "nv32",
    "747e_ff": "nv32",
    "747e_d8": "nv32",
    "747e_d9": "nv32",
    "747e_db": "nv32",
    "748f_30": "gfx1200",
    "748f_31": "gfx1200",
    "748f_32": "RX9060",
    "748f_f0": "gfx1200",
    "748f_f1": "gfx1200",
    "748f_f2": "gfx1200",
    "748f_f3": "RX9060",
    "7590_c0": "gfx1200",
    "7590_c7": "RX9060",
    "746f_30": "RX9070GRE",
    "746f_31": "gfx1201",
    "746f_32": "RX9070",
    "746f_f0": "RX9070GRE",
    "746f_f1": "RX9070GRE",
    "746f_f2": "RX9070GRE",
    "746f_f3": "RX9070",
    "746f_f4": "RX9070",
    "746f_f5": "gfx1201",
    "746f_f6": "RX9070GRE",
    "7550_c0": "gfx1201",
    "7550_c3": "RX9070GRE",
    "7551_c0": "gfx1201",
}


def short_name_for_device(device_id: str) -> str:
    """Map ``<device>_<revision>`` to RVS config directory short name."""
    return GPU_DEVICE_MAP.get((device_id or "").lower(), "")


def detect_gpu_conf_dir_from_lspci(*, cmake_executor=None) -> str:
    """Detect GPU PCI device ID via lspci and map to RVS config directory name."""
    if cmake_executor is not None:
        result = cmake_executor.run("lspci -n -d 1002: | grep -E '0300|1200' | head -1")
        line = (result.stdout or "").strip()
    else:
        rc, stdout, _stderr = run_cmd_get_stdout_stderr(
            "bash",
            "-c",
            "lspci -n -d 1002: | grep -E '0300|1200' | head -1",
            timeout=10,
            quiet=True,
        )
        line = stdout.strip() if rc == 0 else ""

    if not line:
        logger.warning("No AMD GPU detected via lspci")
        return ""

    match = re.search(r"1002:([0-9a-f]{4})", line, re.IGNORECASE)
    if not match:
        logger.warning("Could not parse device ID from lspci line: %s", line)
        return ""
    device_id = match.group(1).lower()

    rev_match = re.search(r"\(rev\s+([0-9a-f]+)\)", line, re.IGNORECASE)
    rev = rev_match.group(1).lower() if rev_match else "00"

    key = f"{device_id}_{rev}"
    gpu_name = GPU_DEVICE_MAP.get(key, "")

    if gpu_name:
        logger.info("Detected GPU: device_id=%s, rev=%s, key=%s -> %s", device_id, rev, key, gpu_name)
    else:
        logger.warning(
            "GPU detected (device_id=%s, rev=%s, key=%s) but no mapping found",
            device_id,
            rev,
            key,
        )
    return gpu_name
