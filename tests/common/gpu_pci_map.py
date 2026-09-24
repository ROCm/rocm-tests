# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shared PCI device-id → GPU short-name map for RVS config lookup.

Single source of truth for resolving a detected device to the per-GPU directory
name its RVS configs live under.
"""

from __future__ import annotations

import json
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
    # Keyed by power variant rather than device id; see POWER_VARIANT_LOOKUP_KEYS.
    "75a8_00_450w": "MI350P-450W",
    "75a8_00_600w": "MI350P-600W",
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

# Device ids shared by several board power variants that ship different RVS
# configs, mapped to ``{max_power_watts: GPU_DEVICE_MAP key}``. Both MI350P
# boards report the same device and revision, so the board power limit is the
# only thing that tells their config trees apart.
POWER_VARIANT_LOOKUP_KEYS: dict[str, dict[int, str]] = {
    "75a8_00": {
        450: "75a8_00_450w",
        600: "75a8_00_600w",
    },
}

# Config directories that ship a module's conf under a name other than the one
# callers ask for, keyed by directory -> {requested name: name on disk}.
CONFIG_FILENAME_OVERRIDES: dict[str, dict[str, str]] = {
    "MI350P-450W": {"babel.conf": "babel_single.conf"},
    "MI350P-600W": {"babel.conf": "babel_single.conf"},
}


class ConfDirUnresolvedError(RuntimeError):
    """The GPU could not be resolved to an RVS config directory.

    Raised rather than returning an empty name, which would fall through to the
    generic config and run settings this GPU was never qualified with.
    """


def short_name_for_device(device_id: str) -> str:
    """Map ``<device>_<revision>`` to RVS config directory short name."""
    return GPU_DEVICE_MAP.get((device_id or "").lower(), "")


def conf_filename_for(gpu_conf_dir: str, config_name: str) -> str:
    """Return the name *config_name* is stored under inside ``gpu_conf_dir``.

    Directories without an override return the requested name unchanged.
    """
    return CONFIG_FILENAME_OVERRIDES.get(gpu_conf_dir, {}).get(config_name, config_name)


# Vendor 0x1002 is AMD. The classes cover VGA (0300), display controller (0380)
# and processing accelerator (1200) so datacenter parts are not missed: MI210
# reports 0380, which a VGA-only filter drops entirely.
_SYSFS_PCI_ID_CMD = (
    "for dev in /sys/bus/pci/devices/*; do "
    "vendor=$(cat $dev/vendor 2>/dev/null); class=$(cat $dev/class 2>/dev/null); "
    "case $vendor:$class in "
    "0x1002:0x0300*|0x1002:0x0380*|0x1002:0x1200*) "
    "echo $(cat $dev/device 2>/dev/null)_$(cat $dev/revision 2>/dev/null);; "
    "esac; done | tail -n 1"
)

# Ordered ``(name, command, pattern, default_revision)`` detection candidates.
# Every entry is a fallback rather than an exclusive choice: amd-smi and rocm-smi
# need a healthy driver plus access to /dev/kfd and /dev/dri/renderD*, so they
# report N/A for a caller outside the render group, while lspci and PCI sysfs
# still answer.
# lspci is matched on the textual class names because numeric class filters miss
# parts that enumerate as "Display controller".
_DETECTION_METHODS: tuple[tuple[str, str, re.Pattern, str], ...] = (
    (
        "amd-smi",
        "amd-smi static --asic 2>/dev/null",
        re.compile(r"DEVICE_ID:\s*0x([0-9a-fA-F]{4}).*?REV_ID:\s*0x([0-9a-fA-F]{1,2})", re.DOTALL),
        "",
    ),
    (
        "amd-smi with sudo",
        "sudo -n amd-smi static --asic 2>/dev/null",
        re.compile(r"DEVICE_ID:\s*0x([0-9a-fA-F]{4}).*?REV_ID:\s*0x([0-9a-fA-F]{1,2})", re.DOTALL),
        "",
    ),
    (
        "rocm-smi",
        "rocm-smi --showid 2>/dev/null",
        re.compile(r"Device ID:\s*0x([0-9a-fA-F]{4}).*?Device Rev:\s*0x([0-9a-fA-F]{1,2})", re.DOTALL),
        "",
    ),
    (
        "lspci",
        "lspci -nn -v 2>/dev/null | grep -iE 'vga|display|accelerators' | grep -i amd | tail -n 1",
        re.compile(
            r"\[(?:[0-9a-fA-F]{4}):(?P<DID>[0-9a-fA-F]{4})\](?:.*?rev (?P<RID>[0-9a-fA-F]+))?",
            re.IGNORECASE,
        ),
        "00",
    ),
    ("pci sysfs", _SYSFS_PCI_ID_CMD, re.compile(r"0x([0-9a-fA-F]{4})_0x([0-9a-fA-F]{1,2})"), ""),
)


def _run_detection(cmd: str, cmake_executor=None) -> str:
    """Run a detection command, returning stdout ("" when it could not run)."""
    if cmake_executor is not None:
        return (cmake_executor.run(cmd).stdout or "").strip()
    # The exit code is deliberately ignored: the pattern match is the gate, so a
    # command that exits non-zero but still printed usable ids is accepted, and a
    # missing binary simply yields no output and falls through to the next method.
    _rc, stdout, _stderr = run_cmd_get_stdout_stderr("bash", "-c", cmd, timeout=30, quiet=True)
    return (stdout or "").strip()


def _device_revision_from_output(output: str, pattern: re.Pattern, default_revision: str = "") -> str:
    """Extract ``<device_id>_<revision>`` from detection output, else ``""``.

    ``default_revision`` covers lspci, which omits ``rev`` entirely for revision 0.
    """
    for match in pattern.finditer(output or ""):
        groups = match.groupdict()
        if groups:
            device_id, revision_id = groups.get("DID"), groups.get("RID")
        else:
            device_id, revision_id = match.group(1), match.group(2)
        revision_id = revision_id or default_revision
        if device_id and revision_id:
            return f"{device_id.lower()}_{revision_id.lower()}"
    return ""


def detect_device_revision(*, cmake_executor=None) -> str:
    """Return the GPU ``<device_id>_<revision>`` key, trying each source in order."""
    for name, cmd, pattern, default_revision in _DETECTION_METHODS:
        output = _run_detection(cmd, cmake_executor)
        key = _device_revision_from_output(output, pattern, default_revision)
        if key:
            logger.debug("PCI device_id_revision from %s: %s", name, key)
            return key
        logger.debug("PCI device id lookup via %s returned no usable ids; trying the next method", name)
    logger.warning("No valid PCI DeviceID/RevisionID found via amd-smi, rocm-smi, lspci or PCI sysfs")
    return ""


def _max_power_watts(cmake_executor=None) -> int | None:
    """Return the board's max power limit in watts, or ``None`` if unreadable."""
    output = _run_detection("amd-smi static --limit --json 2>/dev/null", cmake_executor)
    # amd-smi folds warnings into stdout, so the JSON object has to be cut out of
    # the surrounding text rather than parsed whole.
    start, end = output.find("{"), output.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        limit = json.loads(output[start : end + 1])["gpu_data"][0]["limit"]
    except (ValueError, KeyError, IndexError, TypeError):
        logger.warning("amd-smi reported no usable power limit")
        return None

    # MI300 and MI350 report the board limit under ppt0; other parts use max_power.
    ppt0 = limit.get("ppt0") if isinstance(limit, dict) else None
    for node in (ppt0.get("max_power_limit") if isinstance(ppt0, dict) else None, limit.get("max_power")):
        if isinstance(node, dict):
            try:
                return int(node["value"])
            except (KeyError, TypeError, ValueError):
                continue
    return None


def _resolve_power_variant(key: str, *, cmake_executor=None) -> str:
    """Refine a device key that several board power variants share.

    Returns *key* unchanged for devices with a single variant, and raises when a
    shared device cannot be disambiguated.
    """
    variants = POWER_VARIANT_LOOKUP_KEYS.get(key)
    if not variants:
        return key

    watts = _max_power_watts(cmake_executor)
    variant = variants.get(watts)
    if not variant:
        raise ConfDirUnresolvedError(
            f"Device {key} ships in {sorted(variants)}W variants with different RVS configs, "
            f"but amd-smi reported {f'{watts}W' if watts else 'no usable power limit'}"
        )

    logger.info("Device %s resolved to %s from its %sW power limit", key, variant, watts)
    return variant


def detect_device_key(*, cmake_executor=None) -> str:
    """Detect the GPU and return its ``<device>_<revision>`` key.

    Shared devices come back as their power-variant key (``75a8_00_450w``).

    Raises:
        ConfDirUnresolvedError: no GPU was detected, or a shared device's power
            variant could not be determined.
    """
    key = detect_device_revision(cmake_executor=cmake_executor)
    if not key:
        raise ConfDirUnresolvedError("No GPU detected via amd-smi, rocm-smi, lspci or PCI sysfs")
    return _resolve_power_variant(key, cmake_executor=cmake_executor)


def detect_gpu_conf_dir(*, cmake_executor=None) -> str:
    """Detect the GPU and map it to its RVS config directory name.

    Raises:
        ConfDirUnresolvedError: no GPU was detected, the detected device has no
            GPU_DEVICE_MAP entry, or a shared device's power variant could not
            be determined.
    """
    key = detect_device_key(cmake_executor=cmake_executor)
    gpu_name = GPU_DEVICE_MAP.get(key, "")
    if not gpu_name:
        raise ConfDirUnresolvedError(f"GPU {key} has no GPU_DEVICE_MAP entry, so its RVS config directory is unknown")

    logger.info("Detected GPU: key=%s -> %s", key, gpu_name)
    return gpu_name
