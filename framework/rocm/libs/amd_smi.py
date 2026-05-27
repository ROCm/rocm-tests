# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
amd_smi.py -- Version-agnostic AMD SMI query helpers.

ROCm releases change the ``amd-smi`` JSON schema between versions.  This module
abstracts those differences with a ``_get()`` cascade helper that tries each
known key-path in priority order.  Adding support for a new schema variant is a
single-line path addition — test code never changes.

Known schema variants handled:

    Temperature:
        ROCm 6.x  ``temperature.hotspot_temperature``
        ROCm 5.x  ``thermal.gfx.value``
        Alt        ``temperature.edge_temperature``

    VRAM total:
        ROCm 6.x  ``vram.total.value``  (nested dict with "value"/"unit")
        ROCm 5.x  ``vram_total_mb``     (flat integer, MB)
        Alt        ``vram_info.vram_total_mb``

    BDF address:
        ROCm 6.x  ``bdf``              (top-level key)
        ROCm 5.x  ``gpu.bdf``          (nested under "gpu")

Usage::

    from framework.rocm.libs.amd_smi import list_devices, query_gpu_temp, require_amd_smi_version
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
# Shared unit-conversion helper
# ---------------------------------------------------------------------------


def _to_mb(node: Any) -> int:
    """Convert a VRAM value from amd-smi JSON to integer MB.

    Handles three schema variants across ROCm versions:
    - ``{"value": N, "unit": "MB"}`` dict  (ROCm 6.x nested)
    - plain int already in MB              (ROCm 5.x flat)
    - plain int in bytes (> 1 GiB)        (convert ÷ 1 MiB)

    Args:
        node: Raw value from the amd-smi JSON tree.

    Returns:
        Integer MB value, or 0 if the input is unrecognised.
    """
    if isinstance(node, dict):
        return int(node.get("value", 0))
    if isinstance(node, int):
        return node // (1024 * 1024) if node > 1024 * 1024 else node
    return 0


# ---------------------------------------------------------------------------
# Core cascade helper
# ---------------------------------------------------------------------------


def _get(data: dict, *paths: tuple, default: Any = None) -> Any:
    """Try each key-path in order; return the first non-None result or *default*.

    Insulates callers from JSON schema differences across ROCm versions.  Each
    *path* is a tuple of keys to traverse.  The first path that resolves to a
    non-None value wins.

    Args:
        data:    Top-level dict from parsed JSON output.
        *paths:  Key-path tuples to attempt in priority order.
        default: Value returned when all paths fail (default: None).

    Returns:
        First non-None value found by traversing any path, or *default*.

    Example::

        temp = _get(entry,
            ("temperature", "hotspot_temperature"),   # ROCm 6.x
            ("thermal", "gfx", "value"),              # ROCm 5.x
            ("temperature", "edge_temperature"),       # fallback
        )
    """
    for path in paths:
        node = data
        try:
            for key in path:
                node = node[key]
            if node is not None:
                return node
        except (KeyError, TypeError, IndexError):
            continue
    return default


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def amd_smi_version(executor: AbstractExecutor) -> tuple[int, int, int] | None:
    """Return the ``amd-smi`` version as an ``(major, minor, patch)`` tuple.

    Args:
        executor: Any executor with a ``.run(command)`` method.

    Returns:
        Version tuple, or None if not detectable.
    """
    result = executor.run("amd-smi --version 2>/dev/null")
    if not result.ok or not result.stdout.strip():
        return None
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", result.stdout)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def require_amd_smi_version(executor: AbstractExecutor, major: int, minor: int = 0) -> None:
    """Fail the current test if ``amd-smi`` version is below ``major.minor``.

    Args:
        executor: Any executor with a ``.run()`` method.
        major:    Minimum required major version.
        minor:    Minimum required minor version (default 0).

    Raises:
        pytest.fail.Exception: When ``amd-smi`` is absent or below the required version —
            missing ``amd-smi`` is a prerequisite failure, not a resource shortage.
    """
    import pytest  # pylint: disable=import-outside-toplevel

    ver = amd_smi_version(executor)
    if ver is None:
        pytest.fail(f"amd-smi not detectable — cannot assert >= {major}.{minor}")
    if ver[:2] < (major, minor):
        pytest.fail(f"amd-smi {major}.{minor}+ required; installed {ver[0]}.{ver[1]}.{ver[2]}")


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------


def list_devices(executor: AbstractExecutor) -> list[GpuDeviceInfo]:
    """Return device descriptors for all AMD GPUs visible to the executor.

    Handles ``amd-smi static --json`` schema differences across ROCm versions
    using the ``_get()`` cascade helper.

    Args:
        executor: Any executor with a ``.run(command)`` method.

    Returns:
        List of GpuDeviceInfo, one per detected GPU.  Empty on failure.

    Example::

        devices = list_devices(cpu_executor)
        assert len(devices) >= 1
    """
    result = executor.run("amd-smi static --json")
    if not result.ok:
        logger.warning("amd-smi static failed (exit %d): %s", result.exit_code, result.stderr)
        return []
    try:
        raw: list[dict] = json.loads(result.stdout)
        devices = []
        for i, dev in enumerate(raw):
            # VRAM total: ROCm 6.x nested dict vs 5.x flat MB int
            vram_raw = _get(
                dev,
                ("vram", "total"),  # 6.x nested
                ("vram_total_mb",),  # 5.x flat
                ("vram_info", "vram_total_mb"),  # alternative
            )
            vram_total = _to_mb(vram_raw)

            # BDF: top-level in 6.x, nested under "gpu" in 5.x
            bdf = _get(
                dev,
                ("bdf",),
                ("gpu", "bdf"),
                default="",
            )

            devices.append(
                GpuDeviceInfo(
                    index=i,
                    arch=_get(
                        dev,
                        ("asic", "target_graphics_version"),
                        ("asic", "arch"),
                        default="unknown",
                    ),
                    vram_total=int(vram_total),
                    bdf=str(bdf),
                    driver_ver=_get(dev, ("driver", "driver_version"), default="unknown"),
                    asic_serial=_get(dev, ("asic", "asic_serial"), default="unknown"),
                )
            )
        return devices
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Failed to parse amd-smi static output: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Metric queries
# ---------------------------------------------------------------------------


def query_gpu_temp(executor: AbstractExecutor, gpu_index: int = 0) -> int | None:
    """Return the hot-spot temperature in Celsius for *gpu_index*.

    Handles ``temperature.hotspot_temperature`` (ROCm 6.x),
    ``thermal.gfx.value`` (ROCm 5.x), and ``temperature.edge_temperature``
    (fallback) schema variants.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal to query.

    Returns:
        Temperature in Celsius, or None if unavailable.
    """
    return _query_thermal(executor, gpu_index).temp_hotspot


def query_vram_usage(executor: AbstractExecutor, gpu_index: int = 0) -> GpuVramInfo | None:
    """Return VRAM usage for *gpu_index* in MB.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal to query.

    Returns:
        GpuVramInfo with total/used/free in MB, or None if unavailable.
    """
    result = executor.run(f"amd-smi metric --gpu {gpu_index} --json")
    if not result.ok:
        return None
    try:
        data = json.loads(result.stdout)
        entry = data[0] if isinstance(data, list) else data
        vram = entry.get("vram", {})
        return GpuVramInfo(
            index=gpu_index,
            total_mb=_to_mb(_get(vram, ("vram_total",), ("total",), default=0)),
            used_mb=_to_mb(_get(vram, ("vram_used",), ("used",), default=0)),
            free_mb=_to_mb(_get(vram, ("vram_free",), ("free",), default=0)),
        )
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
        logger.warning("Failed to parse VRAM info for GPU %d: %s", gpu_index, exc)
        return None


def query_ecc_errors(executor: AbstractExecutor, gpu_index: int = 0) -> int | None:
    """Return the total correctable ECC error count for *gpu_index*.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal to query.

    Returns:
        Total correctable ECC error count, or None if unavailable.
    """
    result = executor.run(f"amd-smi metric --gpu {gpu_index} --json")
    if not result.ok:
        return None
    try:
        data = json.loads(result.stdout)
        entry = data[0] if isinstance(data, list) else data
        ecc = entry.get("ecc", {})
        return _get(  # type: ignore[no-any-return]
            ecc,
            ("total_correctable_count",),
            ("correctable_count",),
        )
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


def query_gpu_utilization(executor: AbstractExecutor, gpu_index: int = 0) -> int | None:
    """Return the GFX compute utilization percentage for *gpu_index*.

    Handles schema differences across ROCm versions using the ``_get()`` cascade.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal to query.

    Returns:
        Integer percentage (0-100), or None if unavailable.
    """
    result = executor.run(f"amd-smi metric --gpu {gpu_index} --json")
    if not result.ok:
        return None
    try:
        data = json.loads(result.stdout)
        entry = data[0] if isinstance(data, list) else data
        raw = _get(
            entry,
            ("activity", "gfx_activity"),  # ROCm 6.x
            ("usage", "gfx_usage"),  # ROCm 5.x alt
            ("gfx_activity",),  # flat fallback
            default=None,
        )
        if raw is None:
            return None
        if isinstance(raw, dict):
            raw = raw.get("value", raw)
        val = int(str(raw).rstrip("%").strip())
        return max(0, min(100, val))
    except (json.JSONDecodeError, KeyError, TypeError, IndexError, ValueError) as exc:
        logger.debug("Failed to parse utilization for GPU %d: %s", gpu_index, exc)
        return None


def query_clock_state(executor: AbstractExecutor, gpu_index: int = 0) -> str | None:
    """Return the current GPU performance level (clock state) for *gpu_index*.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal to query.

    Returns:
        Performance level string (e.g. ``"auto"``, ``"high"``), or None.
    """
    result = executor.run(f"amd-smi metric --gpu {gpu_index} --json")
    if not result.ok:
        return None
    try:
        data = json.loads(result.stdout)
        entry = data[0] if isinstance(data, list) else data
        return _get(  # type: ignore[no-any-return]
            entry,
            ("clock", "performance_level"),
            ("clocks", "performance_level"),
        )
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Single-call metric helpers (used by GpuMonitor / GpuBackgroundMonitor)
# ---------------------------------------------------------------------------


def _run_metric_json(executor: AbstractExecutor, gpu_index: int) -> dict | None:
    """Run ``amd-smi metric --gpu N --json`` exactly once and return the parsed entry dict.

    All per-metric callers in the monitor module use this to share a single
    subprocess invocation per GPU per poll cycle.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal to query.

    Returns:
        Parsed first entry dict from the JSON array, or None on any failure.
    """
    result = executor.run(f"amd-smi metric --gpu {gpu_index} --json")
    if not result.ok:
        return None
    try:
        data = json.loads(result.stdout)
        return data[0] if isinstance(data, list) else data  # type: ignore[no-any-return]
    except (json.JSONDecodeError, TypeError):
        return None


def _parse_temp(entry: dict) -> int | None:
    """Extract the hot-spot temperature (Celsius) from a pre-parsed metric entry dict."""
    return _get(  # type: ignore[no-any-return]
        entry,
        ("temperature", "hotspot_temperature"),  # ROCm 6.x
        ("thermal", "gfx", "value"),  # ROCm 5.x
        ("temperature", "edge_temperature"),  # fallback
    )


def _parse_vram(entry: dict, gpu_index: int) -> GpuVramInfo | None:
    """Extract VRAM usage from a pre-parsed metric entry dict."""
    vram = entry.get("vram", {})
    total = _to_mb(_get(vram, ("vram_total",), ("total",), default=0))
    used = _to_mb(_get(vram, ("vram_used",), ("used",), default=0))
    free = _to_mb(_get(vram, ("vram_free",), ("free",), default=0))
    if total == 0 and used == 0:
        return None
    return GpuVramInfo(index=gpu_index, total_mb=total, used_mb=used, free_mb=free)


def _parse_util(entry: dict) -> int | None:
    """Extract GFX compute utilization (0-100 %) from a pre-parsed metric entry dict."""
    raw = _get(
        entry,
        ("activity", "gfx_activity"),  # ROCm 6.x
        ("usage", "gfx_usage"),  # ROCm 5.x alt
        ("gfx_activity",),  # flat fallback
        default=None,
    )
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("value", raw)
    try:
        val = int(str(raw).rstrip("%").strip())
        return max(0, min(100, val))
    except (ValueError, TypeError):
        return None


def _parse_ecc(entry: dict) -> int | None:
    """Extract total correctable ECC error count from a pre-parsed metric entry dict."""
    ecc = entry.get("ecc", {})
    return _get(  # type: ignore[no-any-return]
        ecc,
        ("total_correctable_count",),
        ("correctable_count",),
    )


def _parse_clock(entry: dict) -> str | None:
    """Extract the GPU performance level (clock state) from a pre-parsed metric entry dict."""
    return _get(  # type: ignore[no-any-return]
        entry,
        ("clock", "performance_level"),
        ("clocks", "performance_level"),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _query_thermal(executor: AbstractExecutor, gpu_index: int) -> GpuThermalInfo:
    """Parse thermal data from ``amd-smi metric --gpu N --json``.

    Handles multi-schema temperature keys across ROCm versions.

    Args:
        executor:  Any executor with a ``.run()`` method.
        gpu_index: AMD GPU ordinal.

    Returns:
        GpuThermalInfo — fields are None when parsing fails.
    """
    result = executor.run(f"amd-smi metric --gpu {gpu_index} --json")
    if not result.ok:
        return GpuThermalInfo(index=gpu_index)
    try:
        data = json.loads(result.stdout)
        entry = data[0] if isinstance(data, list) else data

        # Temperature hot-spot: ROCm 6.x nested vs 5.x thermal.gfx.value
        hotspot = _get(
            entry,
            ("temperature", "hotspot_temperature"),  # ROCm 6.x
            ("thermal", "gfx", "value"),  # ROCm 5.x
            ("temperature", "edge_temperature"),  # fallback
        )
        edge = _get(
            entry,
            ("temperature", "edge_temperature"),
            ("temperature", "hotspot_temperature"),
        )
        fan = _get(
            entry,
            ("temperature", "fan_speed_rpm"),
            ("fan", "speed_rpm"),
            default=-1,
        )
        return GpuThermalInfo(
            index=gpu_index,
            temp_edge=edge,
            temp_hotspot=hotspot,
            fan_rpm=fan if fan is not None else -1,
        )
    except (json.JSONDecodeError, KeyError, TypeError, IndexError) as exc:
        logger.warning("Failed to parse thermal info for GPU %d: %s", gpu_index, exc)
        return GpuThermalInfo(index=gpu_index)
