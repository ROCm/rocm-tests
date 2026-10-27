# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""High-intensity memory sizing shared by the GPU memory tests."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor


DEFAULT_FREE_VRAM_PCT = 90
MIN_HOST_RESERVE_MB = 32 * 1024
HOST_RESERVE_PCT = 10
# Ceiling on the reserve as a share of total RAM. Without it the 32 GiB floor
# above exceeds total RAM on a small host and reserves everything.
HOST_RESERVE_MAX_PCT = 50
HOST_MEMORY_RECOVERY_TIMEOUT_SEC = 300
HOST_MEMORY_RECOVERY_INTERVAL_SEC = 5


def host_reserve_mb(total_mb: int) -> int:
    """Host RAM to hold back from a host-backed allocation target.

    ``max(32 GiB, 10% of total)``, then capped at ``HOST_RESERVE_MAX_PCT`` of
    total. The 32 GiB floor is a rounding error on the hosts this suite targets
    (102 GiB on a 1 TiB machine) but exceeds total RAM outright on a small
    dev/CI runner, which drove the target to 0 and produced a spurious FAIL no
    matter how much memory was free. The cap stops the floor inverting into
    "reserve everything".

    Public so the sizing contract is testable against this function rather
    than against a copy of the arithmetic in a test.
    """
    return min(
        max(MIN_HOST_RESERVE_MB, total_mb * HOST_RESERVE_PCT // 100),
        total_mb * HOST_RESERVE_MAX_PCT // 100,
    )


def _value_mb(field) -> int | None:
    """Convert one amd-smi memory field to whole MB, or ``None`` if unusable.

    ``None`` means "no reading" -- the field is absent, non-numeric (``N/A``)
    or carries a unit we don't know. A reading of **0 is returned as 0**, not
    as ``None``: a GPU with no free VRAM is the most important input to
    ``query_min_free_vram_mb``, and dropping it there let sizing proceed as
    though the exhausted GPU were not in the fleet. Only a negative figure is
    rejected as nonsense.
    """
    if isinstance(field, dict):
        value = field.get("value")
        unit = str(field.get("unit", "MB")).upper()
    else:
        value = field
        unit = "MB"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    scales = {"B": 1 / (1024 * 1024), "KB": 1 / 1024, "MB": 1, "GB": 1024}
    scale = scales.get(unit)
    if scale is None:
        return None
    result = int(number * scale)
    return result if result >= 0 else None


def query_min_free_vram_mb(
    rocm_root: Path,
    monitor_executor: AbstractExecutor | None = None,
) -> int | None:
    """Return the smallest free-VRAM reading across visible GPUs.

    The smallest reading is the point of this function: every GPU gets the
    same allocation, so the most constrained GPU decides what all of them can
    take. A GPU reporting 0 free therefore drives the result to 0, which
    ``high_intensity_blocks_mb`` turns into ``None`` via its 1 GiB floor --
    the callers then FAIL asking for an explicit size rather than allocating
    a figure that GPU cannot honour.
    """
    from tests.common.gpu_monitored.executor_bridge import run_command_captured

    amd_smi = rocm_root / "bin" / "amd-smi"
    command = str(amd_smi) if amd_smi.is_file() else "amd-smi"
    result = run_command_captured(
        monitor_executor,
        [command, "metric", "-m", "--json"],
        timeout=30,
    )
    if result.exit_code != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    rows = data.get("gpu_data", data.get("gpu", [])) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return None
    values = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        memory = row.get("mem_usage", row.get("memory_usage", {}))
        if not isinstance(memory, dict):
            continue
        value = _value_mb(memory.get("free_vram", memory.get("free_visible_vram")))
        if value is not None:
            values.append(value)
    return min(values) if values else None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _cgroup_memory_mb(root: Path = Path("/sys/fs/cgroup")) -> tuple[int, int] | None:
    """Return ``(limit_mb, available_mb)`` for this cgroup, or None if unlimited.

    ``/proc/meminfo`` reports the **host**'s memory even inside a
    memory-limited container, so sizing from it alone can target far more
    than the cgroup allows and get the workload OOM-killed while the host
    still looks idle. Consult the cgroup limit so the container's own
    ceiling wins. Handles v2 (``memory.max`` / ``memory.current``) and v1
    (``memory.limit_in_bytes`` / ``memory.usage_in_bytes``); both report a
    sentinel far above real RAM when unlimited.
    """
    v2_max, v2_cur = root / "memory.max", root / "memory.current"
    if v2_max.is_file():
        raw = ""
        try:
            raw = v2_max.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        if raw == "max":
            return None  # explicitly unlimited
        limit = int(raw) if raw.isdigit() else None
        current = _read_int(v2_cur)
        if limit:
            limit_mb = limit // (1024 * 1024)
            if current is None:
                return limit_mb, 0
            return limit_mb, max(0, limit - current) // (1024 * 1024)
        return None

    v1 = root / "memory"
    limit = _read_int(v1 / "memory.limit_in_bytes")
    if limit is None:
        return None
    # v1 signals "no limit" with a value near the word size; anything at or
    # above a petabyte is not a real container budget.
    if limit >= 1 << 50:
        return None
    current = _read_int(v1 / "memory.usage_in_bytes")
    if current is None:
        return limit // (1024 * 1024), 0
    return limit // (1024 * 1024), max(0, limit - current) // (1024 * 1024)


def _host_memory_mb(path: Path = Path("/proc/meminfo")) -> tuple[int, int] | None:
    """Return ``(total_mb, available_mb)``, bounded by the cgroup limit."""
    try:
        fields = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            key, _, remainder = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                fields[key] = int(remainder.strip().split()[0]) // 1024
        if fields.get("MemTotal", 0) > 0 and fields.get("MemAvailable", 0) > 0:
            total, available = fields["MemTotal"], fields["MemAvailable"]
            cgroup = _cgroup_memory_mb()
            if cgroup is not None:
                total = min(total, cgroup[0])
                available = min(available, cgroup[1])
            return total, available
    except (OSError, ValueError, IndexError):
        pass
    return None


def high_intensity_blocks_mb(
    rocm_root: Path,
    num_gpus: int,
    *,
    host_backed: bool,
    host_recovery_timeout_sec: float = 0,
    host_recovery_interval_sec: float = HOST_MEMORY_RECOVERY_INTERVAL_SEC,
    require_full_target: bool = False,
    monitor_executor: AbstractExecutor | None = None,
) -> int | None:
    """Size a high-pressure workload while retaining runtime headroom.

    GPU allocation targets 90% of the least-free visible GPU. HMM allocations
    are also bounded by host ``MemAvailable`` after reserving 10% of total host
    RAM (at least 32 GiB), divided across GPU worker threads. Inside a
    memory-limited container the cgroup ceiling replaces the host figures, so
    the target cannot exceed what the container is allowed to allocate.
    """
    free_vram_mb = query_min_free_vram_mb(rocm_root, monitor_executor)
    if free_vram_mb is None:
        return None
    target = free_vram_mb * DEFAULT_FREE_VRAM_PCT // 100
    if host_backed:
        if num_gpus < 1:
            return None
        deadline = time.monotonic() + max(0, host_recovery_timeout_sec)
        best_host_target = 0
        announced_wait = False
        while True:
            host_memory = _host_memory_mb()
            if host_memory is None:
                return None
            total_mb, available_mb = host_memory
            reserve_mb = host_reserve_mb(total_mb)
            host_target = max(0, available_mb - reserve_mb) // num_gpus
            best_host_target = max(best_host_target, host_target)
            if host_target >= target:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not announced_wait:
                print(
                    f"  [memory-sizing] Host memory currently permits "
                    f"{host_target} MB/GPU, below the {target} MB/GPU "
                    f"automatic target; waiting up to "
                    f"{int(host_recovery_timeout_sec)}s for reclamation"
                )
                announced_wait = True
            time.sleep(min(host_recovery_interval_sec, remaining))

        # Size from the *latest* reading, not the high-water mark seen during
        # the wait. Available host memory can trend down while we poll, and a
        # target taken from an earlier peak may exceed what is actually free by
        # the time the workload allocates. ``best_host_target`` is kept only to
        # report how much the host offered at its best.
        if require_full_target and host_target < target:
            print(
                f"  [memory-sizing] Host memory permits {host_target} MB/GPU "
                f"(best {best_host_target} MB/GPU during the wait); refusing "
                f"to reduce the {target} MB/GPU automatic stress target"
            )
            return None
        target = min(target, host_target)
    return target if target >= 1024 else None
