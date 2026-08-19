# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
amd_smi.py -- AMD GPU metrics via amd-smi JSON output.

Minimum supported version: ROCm 7.1.0 / amd-smi 26+.

AmdSmiMetrics: parses temperature, VRAM, utilization, ECC, and clock metrics.
Executor-agnostic: works with LocalExecutor, SshExecutor, and ContainerExecutor
via the same CLI invocation — no separate code path for local vs remote.

ROCm 7.1.0+ JSON schema (amd-smi 26+):
  amd-smi static  → {"gpu_data": [{...}, ...]}
  amd-smi metric  → {"gpu_data": [{...}, ...]}

  VRAM total:    gpu_data[N].vram.size           → {"value": N, "unit": "MB"}
  VRAM usage:    gpu_data[N].vram.{vram_total, vram_used, vram_free}
  Architecture:  gpu_data[N].asic.target_graphics_version  → "gfx942"
  Temperature:   gpu_data[N].temperature.hotspot_temperature → {"value": N, "unit": "C"}
  Utilization:   gpu_data[N].activity.gfx_activity          → {"value": N, "unit": "%"}
  ECC:           gpu_data[N].ecc.total_correctable_count     → int
  Clock state:   gpu_data[N].clock.performance_level         → str
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class GpuDeviceInfo:
    """Parsed device information from ``amd-smi static --json``.

    Attributes:
        index:       GPU ordinal (0-based).
        arch:        GFX architecture string (e.g. ``"gfx942"``).
        vram_total:  Total VRAM in MB.
        bdf:         PCI Bus:Device.Function address string.
        driver_ver:  amdgpu kernel driver version string.
        asic_serial: ASIC serial number for hardware identification.
    """

    index: int
    arch: str
    vram_total: int
    bdf: str = ""
    driver_ver: str = "unknown"
    asic_serial: str = "unknown"


@dataclass
class GpuThermalInfo:
    """Parsed thermal data from ``amd-smi metric --json``.

    Attributes:
        index:        GPU ordinal.
        temp_edge:    Edge (die) temperature in Celsius.
        temp_hotspot: Hot-spot temperature in Celsius.
        fan_rpm:      Fan speed in RPM (-1 if not available).
    """

    index: int
    temp_edge: int | None = None
    temp_hotspot: int | None = None
    fan_rpm: int = -1


@dataclass
class GpuVramInfo:
    """Parsed VRAM usage from ``amd-smi metric --json``.

    Attributes:
        index:    GPU ordinal.
        total_mb: Total VRAM in MB.
        used_mb:  Used VRAM in MB.
        free_mb:  Free VRAM in MB.
    """

    index: int
    total_mb: int
    used_mb: int
    free_mb: int


# ---------------------------------------------------------------------------
# Schema helpers (ROCm 7.1.0+ / amd-smi 26+)
# ---------------------------------------------------------------------------


def _unwrap_entries(data: Any) -> list[dict]:
    """Extract the device list from the ``gpu_data`` wrapper (amd-smi 26+ / ROCm 7.1.0+).

    Both ``amd-smi static`` and ``amd-smi metric`` return
    ``{"gpu_data": [{...}, ...]}`` in ROCm 7.1.0+.

    Args:
        data: Parsed top-level JSON value (dict or list).

    Returns:
        List of per-device entry dicts.  Empty list on unexpected structure.
    """
    if isinstance(data, dict):
        return list(data.get("gpu_data", []))
    # Guard against unexpected bare-list responses.
    return list(data) if isinstance(data, list) else []


def _to_mb(node: Any) -> int:
    """Convert a VRAM value from amd-smi JSON to integer MB.

    ROCm 7.1.0+ always returns ``{"value": N, "unit": "MB"}`` dicts for VRAM
    fields.  A plain int is accepted as a fallback (value already in MB).

    Args:
        node: Raw value from the amd-smi JSON tree.

    Returns:
        Integer MB value, or 0 if the input is unrecognised.
    """
    if isinstance(node, dict):
        return int(node.get("value", 0))
    if isinstance(node, (int, float)):
        return int(node)
    return 0


def _to_scalar(node: Any) -> int | None:
    """Extract a scalar integer from a ``{"value": N}`` dict or a plain int.

    Used for temperature and other metric fields that ROCm 7.1.0+ returns as
    ``{"value": N, "unit": "..."}`` objects.

    Args:
        node: Raw value from the amd-smi JSON tree.

    Returns:
        Integer value, or None if *node* is None or unrecognised.
    """
    if node is None:
        return None
    if isinstance(node, dict):
        return int(node.get("value", 0))
    if isinstance(node, (int, float)):
        return int(node)
    return None


# ---------------------------------------------------------------------------
# Binary resolution
# ---------------------------------------------------------------------------


def resolve_amd_smi_bin(executor: AbstractExecutor, rock_dir: str | None = None) -> str:
    """Return a runnable ``amd-smi`` command for *executor*.

    Prefers the system PATH entry; falls back to ``<rock_dir>/bin/amd-smi`` when
    the executor cannot find ``amd-smi`` on PATH (TheRock installs it there and
    does not always export it).  Resolution runs through the executor so it works
    identically for local, container, and SSH backends.

    Args:
        executor: Any executor with a ``.run(command)`` method.
        rock_dir: Optional TheRock/ROCm install root that provides
            ``bin/amd-smi``.  Ignored when empty or None.

    Returns:
        The command string to invoke ``amd-smi`` — either ``"amd-smi"`` (on
        PATH) or the absolute ``<rock_dir>/bin/amd-smi`` fallback.  Defaults to
        ``"amd-smi"`` when neither can be confirmed.
    """
    if executor.run("command -v amd-smi").ok:
        return "amd-smi"
    if rock_dir:
        candidate = f"{rock_dir.rstrip('/')}/bin/amd-smi"
        if executor.run(f"test -f '{candidate}'").ok:
            logger.debug("amd-smi not on PATH — using rock_dir binary at %s", candidate)
            return candidate
    return "amd-smi"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def amd_smi_version(executor: AbstractExecutor, amd_smi_bin: str = "amd-smi") -> tuple[int, int, int] | None:
    """Return the ``amd-smi`` version as an ``(major, minor, patch)`` tuple.

    Args:
        executor:    Any executor with a ``.run(command)`` method.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        Version tuple, or None if not detectable.
    """
    result = executor.run(f"{amd_smi_bin} --version 2>/dev/null")
    if not result.ok or not result.stdout.strip():
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def require_amd_smi_version(
    executor: AbstractExecutor, major: int, minor: int = 0, amd_smi_bin: str = "amd-smi"
) -> None:
    """Fail the current test if ``amd-smi`` version is below ``major.minor``.

    Args:
        executor:    Any executor with a ``.run()`` method.
        major:       Minimum required major version.
        minor:       Minimum required minor version (default 0).
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Raises:
        pytest.fail.Exception: When ``amd-smi`` is absent or below the required version —
            missing ``amd-smi`` is a prerequisite failure, not a resource shortage.
    """
    import pytest  # pylint: disable=import-outside-toplevel

    ver = amd_smi_version(executor, amd_smi_bin)
    if ver is None:
        pytest.fail(f"amd-smi not detectable — cannot assert >= {major}.{minor}")
    if ver[:2] < (major, minor):
        pytest.fail(f"amd-smi {major}.{minor}+ required; installed {ver[0]}.{ver[1]}.{ver[2]}")


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


def list_devices(executor: AbstractExecutor, amd_smi_bin: str = "amd-smi") -> list[GpuDeviceInfo]:
    """Return device descriptors for all AMD GPUs visible to the executor.

    Parses ``amd-smi static --json`` using the ROCm 7.1.0+ schema.
    Works identically for local and remote executors.

    Args:
        executor:    Any executor with a ``.run(command)`` method.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        List of GpuDeviceInfo, one per detected GPU.  Empty on failure.

    Example::

        devices = list_devices(cpu_executor)
        assert len(devices) >= 1
    """
    result = executor.run(f"{amd_smi_bin} static --json")
    if not result.ok:
        logger.warning("amd-smi static failed (exit %d): %s", result.exit_code, result.stderr)
        return []
    try:
        entries = _unwrap_entries(json.loads(result.stdout))
        devices = []
        for i, dev in enumerate(entries):
            asic = dev.get("asic", {})
            devices.append(
                GpuDeviceInfo(
                    index=i,
                    arch=asic.get("target_graphics_version", "unknown"),
                    vram_total=_to_mb(dev.get("vram", {}).get("size", 0)),
                    bdf=dev.get("bdf", ""),
                    driver_ver=dev.get("driver", {}).get("driver_version", "unknown"),
                    asic_serial=asic.get("asic_serial", "unknown"),
                )
            )
        return devices
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse amd-smi static output: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Metric queries
# ---------------------------------------------------------------------------


def query_gpu_temp(executor: AbstractExecutor, gpu_index: int = 0, amd_smi_bin: str = "amd-smi") -> int | None:
    """Return the hot-spot temperature in Celsius for *gpu_index*.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal to query.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        Temperature in Celsius, or None if unavailable.
    """
    return _query_thermal(executor, gpu_index, amd_smi_bin).temp_hotspot


def query_vram_usage(
    executor: AbstractExecutor, gpu_index: int = 0, amd_smi_bin: str = "amd-smi"
) -> GpuVramInfo | None:
    """Return VRAM usage for *gpu_index* in MB.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal to query.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        GpuVramInfo with total/used/free in MB, or None if unavailable.
    """
    entry = _run_metric_json(executor, gpu_index, amd_smi_bin)
    if entry is None:
        return None
    return _parse_vram(entry, gpu_index)


def query_ecc_errors(executor: AbstractExecutor, gpu_index: int = 0, amd_smi_bin: str = "amd-smi") -> int | None:
    """Return the total correctable ECC error count for *gpu_index*.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal to query.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        Total correctable ECC error count, or None if unavailable.
    """
    entry = _run_metric_json(executor, gpu_index, amd_smi_bin)
    if entry is None:
        return None
    return _parse_ecc(entry)


def query_gpu_utilization(executor: AbstractExecutor, gpu_index: int = 0, amd_smi_bin: str = "amd-smi") -> int | None:
    """Return the GFX compute utilization percentage for *gpu_index*.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal to query.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        Integer percentage (0-100), or None if unavailable.
    """
    entry = _run_metric_json(executor, gpu_index, amd_smi_bin)
    if entry is None:
        return None
    return _parse_util(entry)


def query_clock_state(executor: AbstractExecutor, gpu_index: int = 0, amd_smi_bin: str = "amd-smi") -> str | None:
    """Return the current GPU performance level (clock state) for *gpu_index*.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal to query.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        Performance level string (e.g. ``"auto"``, ``"high"``), or None.
    """
    entry = _run_metric_json(executor, gpu_index, amd_smi_bin)
    if entry is None:
        return None
    return _parse_clock(entry)


# ---------------------------------------------------------------------------
# Single-call metric helpers (used by GpuMonitor / GpuBackgroundMonitor)
# ---------------------------------------------------------------------------


def _run_metric_json(executor: AbstractExecutor, gpu_index: int, amd_smi_bin: str = "amd-smi") -> dict | None:
    """Run ``amd-smi metric --gpu N --json`` exactly once and return the parsed entry dict.

    All per-metric callers in the monitor module use this to share a single
    subprocess invocation per GPU per poll cycle.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal to query.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        Parsed first entry dict from the ``gpu_data`` array, or None on any failure.
    """
    result = executor.run(f"{amd_smi_bin} metric --gpu {gpu_index} --json")
    if not result.ok:
        return None
    try:
        entries = _unwrap_entries(json.loads(result.stdout))
        return entries[0] if entries else None
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_temp(entry: dict) -> int | None:
    """Extract the hot-spot temperature (Celsius) from a pre-parsed metric entry dict."""
    return _to_scalar(entry.get("temperature", {}).get("hotspot_temperature"))


def _parse_vram(entry: dict, gpu_index: int) -> GpuVramInfo | None:
    """Extract VRAM usage from a pre-parsed metric entry dict."""
    vram = entry.get("vram", {})
    total = _to_mb(vram.get("vram_total", 0))
    used = _to_mb(vram.get("vram_used", 0))
    free = _to_mb(vram.get("vram_free", 0))
    if total == 0 and used == 0:
        return None
    return GpuVramInfo(index=gpu_index, total_mb=total, used_mb=used, free_mb=free)


def _parse_util(entry: dict) -> int | None:
    """Extract GFX compute utilization (0-100 %) from a pre-parsed metric entry dict."""
    raw = entry.get("activity", {}).get("gfx_activity")
    val = _to_scalar(raw)
    if val is None:
        return None
    return max(0, min(100, val))


def _parse_ecc(entry: dict) -> int | None:
    """Extract total correctable ECC error count from a pre-parsed metric entry dict."""
    val = entry.get("ecc", {}).get("total_correctable_count")
    return int(val) if val is not None else None


def _parse_clock(entry: dict) -> str | None:
    """Extract the GPU performance level (clock state) from a pre-parsed metric entry dict."""
    val = entry.get("clock", {}).get("performance_level")
    return str(val) if val is not None else None


# ---------------------------------------------------------------------------
# Clock-limit text parsing (``amd-smi metric -c``)
# ---------------------------------------------------------------------------


def _fclk_field(section: str, label: str) -> int | None:
    """Return the integer MHz value of *label* within an ``FCLK_0`` text section."""
    found = re.search(rf"{label}\s*:\s*(\d+)\s*MHz", section)
    return int(found.group(1)) if found else None


def parse_fclk_per_gpu(metric_output: str) -> list[dict]:
    """Parse per-GPU ``FCLK_0`` CLK/MIN_CLK/MAX_CLK from ``amd-smi metric -c`` text.

    The indented ``amd-smi`` text output groups each GPU under a header line
    (``GPU: <id>`` or the older ``GPU <id>:``).  Within a GPU block, the
    ``FCLK_0:`` subsection ends at the next sibling ``<NAME>_<digit>:`` line
    (e.g. ``SOCCLK_0:``).

    Args:
        metric_output: Raw stdout from ``amd-smi metric -c``.

    Returns:
        One dict per GPU with keys ``gpu``, ``clk``, ``min_clk``, ``max_clk``
        (integer MHz, or None when a field is absent).  Empty on no match.
    """
    results: list[dict] = []
    # Match both "GPU: 0" and the older "GPU 0:" header styles.
    gpu_header_re = re.compile(r"(?m)^\s*GPU\s*(?::\s*(\d+)|(\d+)\s*:)\s*$")
    headers = list(gpu_header_re.finditer(metric_output))
    if not headers:
        return results

    for idx, match in enumerate(headers):
        gpu_id = int(match.group(1) or match.group(2))
        block_start = match.end()
        block_end = headers[idx + 1].start() if idx + 1 < len(headers) else len(metric_output)
        block = metric_output[block_start:block_end]

        # FCLK_0 subsection: consecutive lines indented deeper than the label.
        fclk_match = re.search(
            r"(?ms)^(?P<indent>[ \t]*)FCLK_0\s*:\s*\n(?P<body>(?:(?P=indent)[ \t]+.*\n)+)",
            block,
        )
        if not fclk_match:
            continue
        section = fclk_match.group("body")
        results.append(
            {
                "gpu": gpu_id,
                "clk": _fclk_field(section, "CLK"),
                "min_clk": _fclk_field(section, "MIN_CLK"),
                "max_clk": _fclk_field(section, "MAX_CLK"),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_thermal(executor: AbstractExecutor, gpu_index: int, amd_smi_bin: str = "amd-smi") -> GpuThermalInfo:
    """Parse thermal data from ``amd-smi metric --gpu N --json``.

    Args:
        executor:    Any executor with a ``.run()`` method.
        gpu_index:   AMD GPU ordinal.
        amd_smi_bin: ``amd-smi`` command to invoke; pass the result of
            :func:`resolve_amd_smi_bin` when the binary is not on PATH.

    Returns:
        GpuThermalInfo — fields are None when parsing fails.
    """
    entry = _run_metric_json(executor, gpu_index, amd_smi_bin)
    if entry is None:
        return GpuThermalInfo(index=gpu_index)
    try:
        temp = entry.get("temperature", {})
        fan_raw = temp.get("fan_speed_rpm") or entry.get("fan", {}).get("speed_rpm")
        fan = _to_scalar(fan_raw)
        return GpuThermalInfo(
            index=gpu_index,
            temp_edge=_to_scalar(temp.get("edge_temperature")),
            temp_hotspot=_to_scalar(temp.get("hotspot_temperature")),
            fan_rpm=fan if fan is not None else -1,
        )
    except (KeyError, TypeError) as exc:
        logger.warning("Failed to parse thermal info for GPU %d: %s", gpu_index, exc)
        return GpuThermalInfo(index=gpu_index)
