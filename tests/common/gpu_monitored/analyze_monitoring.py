# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Enhanced monitoring analysis for GPU test runs.

Reads power_temp.csv to enrich summary.json with:
- Per-GPU statistics and cross-GPU imbalance detection
- Serial/parallel workload pattern classification
- Thermal/clock throttling event detection
- Power cap utilization, saturation, and stability analysis
- Memory temperature and thermal equilibrium analysis
- Steady-state vs transient segmentation
- Monitoring-based validation for silent failure detection

All analysis is additive — existing summary.json fields are preserved.
All anomaly flags are warnings/informational; exit codes are never changed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
import os
import sys
from typing import Any

try:
    from tests.common.gpu_monitored import csv_schema as _csv_schema  # type: ignore
except ImportError:  # pragma: no cover - script-style invocation

    class _CsvSchemaFallback:
        TIMESTAMP = "timestamp"
        GPU = "gpu"
        POWER_USAGE = "power_usage"
        MAX_POWER = "max_power"
        HOTSPOT_TEMP = "hotspot_temperature"
        MEM_TEMP = "memory_temperature"
        GFX_CLK = "gfx_clk"
        GFX_UTIL = "gfx"
        MEM_UTIL = "mem"
        MEM_CLK = "mem_clock"
        VRAM_USED = "vram_used"
        VRAM_TOTAL = "vram_total"
        VRAM_PCT = "vram_percent"

    _csv_schema = _CsvSchemaFallback()

FALLBACK_THROTTLE_TEMP_C = 85.0
FALLBACK_MEM_TEMP_WARN_C = 95.0

IMBALANCE_PCT = 15.0
THROTTLE_CLK_DROP_PCT = 20.0
POWER_CAP_RATIO = 0.99
TEMP_SLOPE_WARN_C_PER_MIN = 0.1
STEADY_UTIL_THRESH = 50.0
RAMP_UTIL_THRESH = 80.0


def _get_workload_profile(test_name: str) -> dict[str, Any] | None:
    """Return the monitoring profile for ``test_name`` from the registry.

    The profile dict lives on each test's ``TestSpec.workload_profile``
    attribute (see ``tests/common/gpu_monitored/registry.py``). Keeping
    one source of truth avoids drift between validation thresholds and
    test definitions.

    Unknown test names return ``None`` (validation is skipped).
    """
    try:
        from tests.common.gpu_monitored.registry import get_test
    except ImportError:
        return None
    t = get_test(test_name)
    return t.workload_profile if t is not None else None


def _query_gpu_limits() -> dict[str, float]:  # noqa: C901
    """Query thermal/power limits from amd-smi static. Falls back to
    conservative defaults if amd-smi is unavailable."""
    from framework.executors.local_executor import run_cmd_get_stdout_stderr
    from framework.rocm.libs.amd_smi import _get

    limits: dict[str, float] = {
        "throttle_temp_c": FALLBACK_THROTTLE_TEMP_C,
        "mem_temp_warn_c": FALLBACK_MEM_TEMP_WARN_C,
    }
    amd_smi = "amd-smi"
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path and os.path.isfile(os.path.join(rocm_path, "bin", "amd-smi")):
        amd_smi = os.path.join(rocm_path, "bin", "amd-smi")
    try:
        rc, out, _stderr = run_cmd_get_stdout_stderr(
            amd_smi,
            "static",
            "-g",
            "0",
            "--json",
            timeout=10,
            quiet=True,
        )
        if rc != 0:
            return limits
        data = json.loads(out)
        gpu_list = data.get("gpu_data", data) if isinstance(data, dict) else data
        gpu0 = gpu_list[0] if isinstance(gpu_list, list) else gpu_list
        lim = gpu0.get("limit", {})
        field_map = {
            "slowdown_hotspot_temperature": "throttle_temp_c",
            "slowdown_vram_temperature": "mem_temp_warn_c",
            "shutdown_hotspot_temperature": "shutdown_temp_c",
            "shutdown_vram_temperature": "shutdown_mem_temp_c",
        }
        for json_key, limit_key in field_map.items():
            val = _get(lim, (json_key, "value"), (json_key,))
            if val is not None and val != "N/A":
                limits[limit_key] = float(val)

        ppt0 = lim.get("ppt0", {})
        max_pwr = _get(ppt0, ("max_power_limit", "value"), ("max_power_limit",))
        if max_pwr is not None and max_pwr != "N/A":
            limits["max_power_w"] = float(max_pwr)

        clk = gpu0.get("clock", {})
        for clk_domain, limit_key in [("sys", "max_gfx_clk_mhz"), ("mem", "max_mem_clk_mhz")]:
            levels = clk.get(clk_domain, {})
            if isinstance(levels, dict):
                freq_levels = levels.get("frequency_levels", {})
                if freq_levels:
                    vals: list[float] = []
                    for v in freq_levels.values():
                        if isinstance(v, dict):
                            vals.append(float(v.get("value", 0)))
                        elif isinstance(v, str):
                            vals.append(float(v.split()[0]))
                        elif isinstance(v, (int, float)):
                            vals.append(float(v))
                    if vals:
                        max_val = max(vals)
                        if max_val > 0:
                            limits[limit_key] = max_val
    except Exception:
        pass

    if "max_gfx_clk_mhz" not in limits:
        try:
            rc, out, _stderr = run_cmd_get_stdout_stderr(
                amd_smi,
                "metric",
                "-c",
                "-g",
                "0",
                "--json",
                timeout=10,
                quiet=True,
            )
            if rc != 0:
                return limits
            data = json.loads(out)
            gpu_list = data.get("gpu_data", data) if isinstance(data, dict) else data
            gpu0 = gpu_list[0] if isinstance(gpu_list, list) else gpu_list
            max_clk_val = _get(
                gpu0,
                ("clock", "gfx_0", "max_clk", "value"),
                ("clock", "gfx_0", "max_clk"),
            )
            if max_clk_val is not None and max_clk_val != "N/A":
                limits["max_gfx_clk_mhz"] = float(max_clk_val)
        except Exception:
            pass
    return limits


_GPU_LIMITS_CACHE: dict[str, float] | None = None


def _get_gpu_limits() -> dict[str, float]:
    global _GPU_LIMITS_CACHE
    if _GPU_LIMITS_CACHE is None:
        _GPU_LIMITS_CACHE = _query_gpu_limits()
    return _GPU_LIMITS_CACHE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sf(v, default=0.0):
    """Safe float conversion.

    Rejects ``NaN``/``Inf`` — amd-smi occasionally emits ``NaN`` in the
    CSV when a sensor is transient-out, and ``float("nan")`` parses
    successfully; letting it through poisons every min/max/mean below.
    """
    try:
        s = str(v).strip()
        if not s or s.upper() in ("N/A", "NA", "NAN"):
            return default
        fv = float(s)
    except (ValueError, TypeError):
        return default
    if math.isnan(fv) or math.isinf(fv):
        return default
    return fv


def _stats(vals: list[float]) -> dict[str, Any] | None:
    vals = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
    if not vals:
        return None
    n = len(vals)
    avg = sum(vals) / n
    var = sum((v - avg) ** 2 for v in vals) / n if n > 1 else 0.0
    return {
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
        "avg": round(avg, 1),
        "std": round(math.sqrt(var), 1),
        "samples": n,
    }


def _linreg(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """OLS linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys, strict=True))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return slope, intercept, r2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r.get(_csv_schema.TIMESTAMP, "0"))
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "t": t,
                    "gpu": str(r.get(_csv_schema.GPU, "0")),
                    "power": _sf(r.get(_csv_schema.POWER_USAGE)),
                    "max_power": _sf(r.get(_csv_schema.MAX_POWER), 1000),
                    "hotspot_temp": _sf(r.get(_csv_schema.HOTSPOT_TEMP)),
                    "mem_temp": _sf(r.get(_csv_schema.MEM_TEMP)),
                    "gfx_clk": _sf(r.get(_csv_schema.GFX_CLK)),
                    "gfx_util": _sf(r.get(_csv_schema.GFX_UTIL)),
                    "mem_util": _sf(r.get(_csv_schema.MEM_UTIL)),
                    "mem_clk": _sf(r.get(_csv_schema.MEM_CLK)),
                    "vram_used": _sf(r.get(_csv_schema.VRAM_USED)),
                    "vram_total": _sf(r.get(_csv_schema.VRAM_TOTAL)),
                    "vram_pct": _sf(r.get(_csv_schema.VRAM_PCT)),
                }
            )
    return rows


def group_by_gpu(rows: list[dict]) -> dict[str, list[dict]]:
    by_gpu: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu"]].append(r)
    return dict(by_gpu)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def detect_steady_state(gpu_rows: list[dict]) -> tuple[int, int]:
    """Find steady-state window via the first/last samples with gfx_util
    above the ramp threshold. Falls back to the full range so downstream
    code never receives an empty slice.

    Handles the case where only a single sample crosses the threshold by
    keeping that sample's index as both start and end (a one-sample
    steady state), instead of falling back to the full range — the old
    fallback would drag all the ramp/idle samples into the "steady"
    bucket and skew averages.
    """
    n = len(gpu_rows)
    if n < 5:
        return 0, max(0, n - 1)

    start: int | None = None
    for i in range(n):
        if gpu_rows[i]["gfx_util"] >= RAMP_UTIL_THRESH:
            start = i
            break

    if start is None:
        return 0, max(0, n - 1)

    end = start
    for i in range(n - 1, start - 1, -1):
        if gpu_rows[i]["gfx_util"] >= RAMP_UTIL_THRESH:
            end = i
            break

    return start, end


def detect_workload_pattern(by_gpu: dict[str, list[dict]]) -> str:
    if len(by_gpu) <= 1:
        return "single_gpu"

    ts_set = sorted({r["t"] for rows in by_gpu.values() for r in rows})
    if len(ts_set) < 3:
        return "unknown"

    gpu_at_ts: dict[int, int] = defaultdict(int)
    for _gpu_id, rows in by_gpu.items():
        for r in rows:
            if r["gfx_util"] >= STEADY_UTIL_THRESH:
                gpu_at_ts[r["t"]] += 1

    active_counts = sorted(gpu_at_ts.get(t, 0) for t in ts_set)
    median = active_counts[len(active_counts) // 2]
    num_gpus = len(by_gpu)

    if median == 0:
        return "idle"
    if median <= 1:
        return "serial"
    if median >= num_gpus * 0.7:
        return "parallel"
    return "mixed"


def compute_per_gpu(by_gpu: dict[str, list[dict]], steady: dict[str, tuple[int, int]], pattern: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for gpu_id in sorted(by_gpu, key=lambda g: int(g) if g.isdigit() else 0):
        rows = by_gpu[gpu_id]
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        ss = rows[s : e + 1] if e >= s else rows

        if pattern == "serial":
            active = [r for r in rows if r["gfx_util"] >= STEADY_UTIL_THRESH]
            use = active if active else ss
        else:
            use = ss

        result[gpu_id] = {
            "power": _stats([r["power"] for r in use]),
            "hotspot_temp": _stats([r["hotspot_temp"] for r in use]),
            "mem_temp": _stats([r["mem_temp"] for r in use if r["mem_temp"] > 0]),
            "gfx_clk": _stats([r["gfx_clk"] for r in use]),
            "gfx_util": _stats([r["gfx_util"] for r in use]),
            "vram_pct": _stats([r["vram_pct"] for r in use]),
            "total_samples": len(rows),
            "steady_state_samples": len(ss),
            "active_samples": len([r for r in rows if r["gfx_util"] >= STEADY_UTIL_THRESH]),
        }
    return result


def detect_imbalance(per_gpu: dict[str, dict], pattern: str) -> list[dict]:
    if pattern == "serial" or len(per_gpu) <= 1:
        return []
    flags: list[dict] = []
    for metric in ("power", "gfx_clk", "gfx_util"):
        avgs: dict[str, float] = {}
        for gid, st in per_gpu.items():
            s = st.get(metric)
            if s and s["avg"] > 0:
                avgs[gid] = s["avg"]
        if len(avgs) < 2:
            continue
        fleet_mean = sum(avgs.values()) / len(avgs)
        if fleet_mean < 1:
            continue
        for gid, avg in avgs.items():
            dev = abs(avg - fleet_mean) / fleet_mean * 100
            if dev > IMBALANCE_PCT:
                flags.append(
                    {
                        "gpu": gid,
                        "metric": metric,
                        "gpu_avg": round(avg, 1),
                        "fleet_mean": round(fleet_mean, 1),
                        "deviation_pct": round(dev, 1),
                    }
                )
    return flags


def detect_throttling(by_gpu: dict[str, list[dict]], steady: dict[str, tuple[int, int]]) -> list[dict]:
    events: list[dict] = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        ss = rows[s : e + 1]
        if len(ss) < 5:
            continue
        clks = [r["gfx_clk"] for r in ss if r["gfx_clk"] > 0]
        if not clks:
            continue
        avg_clk = sum(clks) / len(clks)
        floor = avg_clk * (1 - THROTTLE_CLK_DROP_PCT / 100)
        hit = sum(1 for r in ss if r["hotspot_temp"] >= _get_gpu_limits()["throttle_temp_c"] and r["gfx_clk"] < floor)
        if hit > 0:
            events.append(
                {
                    "gpu": gpu_id,
                    "throttle_samples": hit,
                    "total_steady_samples": len(ss),
                    "throttle_pct": round(hit / len(ss) * 100, 1),
                    "max_temp_c": round(max(r["hotspot_temp"] for r in ss), 1),
                    "ss_avg_clk_mhz": round(avg_clk, 0),
                }
            )
    return events


def analyze_power(by_gpu: dict[str, list[dict]], steady: dict[str, tuple[int, int]]) -> dict:
    all_ss: list[dict] = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        all_ss.extend(rows[s : e + 1])
    if not all_ss:
        return {}

    powers = [r["power"] for r in all_ss]
    caps = [r["max_power"] for r in all_ss if r["max_power"] > 0]
    avg_cap = sum(caps) / len(caps) if caps else 1000.0

    avg_pwr = sum(powers) / len(powers) if powers else 0
    cap_util = round(avg_pwr / avg_cap * 100, 1) if avg_cap > 0 else 0.0
    at_cap = sum(1 for r in all_ss if r["max_power"] > 0 and r["power"] >= r["max_power"] * POWER_CAP_RATIO)
    cap_sat = round(at_cap / len(all_ss) * 100, 1) if all_ss else 0.0
    pwr_std = round(math.sqrt(sum((p - avg_pwr) ** 2 for p in powers) / len(powers)), 1) if len(powers) > 1 else 0.0

    ramp_s = 0
    first_gpu = sorted(by_gpu.keys())[0] if by_gpu else None
    if first_gpu:
        frows = by_gpu[first_gpu]
        fs, fe = steady.get(first_gpu, (0, max(0, len(frows) - 1)))
        ss_pwr = [r["power"] for r in frows[fs : fe + 1]]
        ss_avg = sum(ss_pwr) / len(ss_pwr) if ss_pwr else 0
        thr90 = ss_avg * 0.9
        if frows and ss_avg > 0:
            t0 = frows[0]["t"]
            for r in frows:
                if r["power"] >= thr90:
                    ramp_s = max(0, r["t"] - t0)
                    break

    return {
        "cap_utilization_pct": cap_util,
        "cap_saturation_pct": cap_sat,
        "power_std_w": pwr_std,
        "ramp_up_seconds": ramp_s,
        "avg_power_cap_w": round(avg_cap, 0),
    }


def analyze_mem_temp(by_gpu: dict[str, list[dict]], steady: dict[str, tuple[int, int]]) -> dict:
    all_ss: list[dict] = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        all_ss.extend(rows[s : e + 1])
    if not all_ss:
        return {}

    mem_t = [r["mem_temp"] for r in all_ss if r["mem_temp"] > 0]
    deltas = [r["hotspot_temp"] - r["mem_temp"] for r in all_ss if r["hotspot_temp"] > 0 and r["mem_temp"] > 0]

    result: dict[str, Any] = {
        "mem_temp": _stats(mem_t),
        "hotspot_mem_delta": _stats(deltas),
        "mem_temp_exceeds_threshold": bool(mem_t and max(mem_t) >= _get_gpu_limits()["mem_temp_warn_c"]),
    }

    rising_gpus: list[dict] = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        ss = rows[s : e + 1]
        if len(ss) < 30:
            continue
        tail = ss[-60:] if len(ss) >= 60 else ss[-30:]
        if len(tail) < 10:
            continue
        xs = [float(r["t"] - tail[0]["t"]) for r in tail]
        ys = [r["hotspot_temp"] for r in tail]
        if max(ys) - min(ys) < 3:
            continue
        slope, _, r2 = _linreg(xs, ys)
        slope_pm = slope * 60
        if slope_pm > TEMP_SLOPE_WARN_C_PER_MIN and r2 > 0.5:
            rising_gpus.append({"gpu": gpu_id, "slope_c_per_min": round(slope_pm, 2), "r2": round(r2, 2)})

    result["temp_still_rising"] = len(rising_gpus) > 0
    result["rising_gpus"] = rising_gpus
    return result


def compute_steady_state_metrics(by_gpu: dict[str, list[dict]], steady: dict[str, tuple[int, int]]) -> dict:
    """Aggregate steady-state samples across all GPUs and compute
    ``start_offset_s``/``end_offset_s`` relative to the *global* t0.
    """
    all_ss: list[dict] = []
    all_rows = [r for rows in by_gpu.values() for r in rows]
    if not all_rows:
        return {}
    global_t0 = min(r["t"] for r in all_rows)

    earliest: int | None = None
    latest: int | None = None
    for gpu_id, rows in by_gpu.items():
        if not rows:
            continue
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        e = min(e, len(rows) - 1)
        s = max(0, min(s, e))
        ss = rows[s : e + 1]
        all_ss.extend(ss)
        so = rows[s]["t"] - global_t0
        eo = rows[e]["t"] - global_t0
        if earliest is None or so < earliest:
            earliest = so
        if latest is None or eo > latest:
            latest = eo
    if not all_ss:
        return {}
    return {
        "start_offset_s": earliest if earliest is not None else 0,
        "end_offset_s": latest if latest is not None else 0,
        "sample_count": len(all_ss),
        "metrics": {
            "power": _stats([r["power"] for r in all_ss]),
            "hotspot_temp": _stats([r["hotspot_temp"] for r in all_ss]),
            "mem_temp": _stats([r["mem_temp"] for r in all_ss if r["mem_temp"] > 0]),
            "gfx_clk": _stats([r["gfx_clk"] for r in all_ss]),
            "gfx_util": _stats([r["gfx_util"] for r in all_ss]),
            "vram_pct": _stats([r["vram_pct"] for r in all_ss]),
        },
    }


def validate_monitoring(test_name: str, per_gpu: dict, pattern: str, run_dir: str = "") -> dict:  # noqa: C901
    """Check monitoring data against expected workload profiles.
    Returns warnings only — never changes pass/fail.
    """
    if run_dir:
        try:
            sp = os.path.join(run_dir, "summary.json")
            if os.path.exists(sp):
                with open(sp) as _f:
                    sd = json.load(_f)
                ec = sd.get("exit_code", 0)
                dur = sd.get("duration_seconds", 0)
                if sd.get("unsupported", False):
                    return {"status": "SKIPPED", "reason": "test returned UNSUPPORTED", "warnings": []}
                if dur < 5 and ec != 0:
                    return {"status": "SKIPPED", "reason": f"test too short ({dur}s) with exit {ec}", "warnings": []}
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    profile = _get_workload_profile(test_name)
    if profile is None:
        return {"status": "SKIPPED", "reason": "no expected profile for this test", "warnings": []}

    warnings: list[str] = []
    is_serial = profile.get("serial", False) or pattern == "serial"

    if is_serial:
        active = [(gid, st) for gid, st in per_gpu.items() if st.get("active_samples", 0) > 0]
        if not active:
            warnings.append("No GPU had any active samples during the test")
        else:
            min_util = profile.get("min_util", 0)
            min_vram_pct = profile.get("min_vram_pct", 0)
            any_meets_util = False
            for _gid, st in active:
                util = st.get("gfx_util")
                if util and util["avg"] >= min_util:
                    any_meets_util = True
                    break
            if not any_meets_util:
                worst = max(
                    (st.get("gfx_util", {}).get("avg", 0) for _, st in active),
                    default=0,
                )
                warnings.append(
                    f"No GPU's active window reached expected " f"{min_util}% gfx util (best was {worst}% avg)"
                )
            if min_vram_pct > 0:
                any_meets_vram = False
                for _gid, st in active:
                    vram = st.get("vram_pct")
                    if vram and vram["max"] >= min_vram_pct:
                        any_meets_vram = True
                        break
                if not any_meets_vram:
                    warnings.append(f"No GPU reached expected {min_vram_pct}% max VRAM")
    else:
        for gpu_id in sorted(per_gpu, key=lambda g: int(g) if g.isdigit() else 0):
            st = per_gpu[gpu_id]
            label = f"GPU {gpu_id}"
            util = st.get("gfx_util")
            if util and util["avg"] < profile["min_util"]:
                warnings.append(f"{label}: avg GFX util {util['avg']}% below expected {profile['min_util']}%")

            vram = st.get("vram_pct")
            if vram and profile["min_vram_pct"] > 0 and vram["max"] < profile["min_vram_pct"]:
                warnings.append(f"{label}: max VRAM {vram['max']}% below expected {profile['min_vram_pct']}%")

    return {"status": "OK" if not warnings else "WARNING", "warnings": warnings}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def health_checks_text(a: dict) -> str:
    lines = ["  Health Checks:"]

    thr = a.get("throttle_events", [])
    if thr:
        worst = max(e["throttle_pct"] for e in thr)
        lines.append(f"    Throttling : WARNING ({len(thr)} GPU(s), up to {worst}% of steady state)")
    else:
        lines.append("    Throttling : OK")

    imb = a.get("imbalanced_gpus", [])
    pat = a.get("workload_pattern", "unknown")
    if imb:
        mx = max(i["deviation_pct"] for i in imb)
        lines.append(f"    GPU Imbalance : WARNING (max deviation {mx}%)")
    elif pat == "serial":
        lines.append("    GPU Imbalance : N/A (serial workload)")
    else:
        lines.append("    GPU Imbalance : OK")

    n_gpus = len(a.get("per_gpu", {}))
    lines.append(f"    Workload : {pat} ({n_gpus} GPU(s))")

    mt = a.get("mem_temp_analysis", {})
    ms = mt.get("mem_temp")
    if ms:
        ds = mt.get("hotspot_mem_delta")
        d_avg = ds["avg"] if ds else 0
        exceeds = mt.get("mem_temp_exceeds_threshold", False)
        tag = "WARNING" if exceeds else "OK"
        lines.append(f"    Mem Temp : {tag} (max {ms['max']}C, delta to hotspot: {d_avg}C)")
    else:
        lines.append("    Mem Temp : N/A")

    if mt.get("temp_still_rising"):
        rg = mt.get("rising_gpus", [])
        sl = ", ".join(f"GPU{g['gpu']}:+{g['slope_c_per_min']}C/min" for g in rg[:3])
        lines.append(f"    Temp Trend : WARNING (still rising: {sl})")
    else:
        lines.append("    Temp Trend : STABLE")

    pw = a.get("power_analysis", {})
    cu = pw.get("cap_utilization_pct", 0)
    cs = pw.get("cap_saturation_pct", 0)
    cw = pw.get("avg_power_cap_w", 0)
    if cu > 0:
        lines.append(f"    Power Cap : {cu}% avg, {cs}% at cap ({cw:.0f}W cap)")
    else:
        lines.append("    Power Cap : N/A")

    mv = a.get("monitoring_validation", {})
    mvs = mv.get("status", "SKIPPED")
    mvw = mv.get("warnings", [])
    if mvw:
        lines.append(f"    Monitor Valid. : {mvs} ({len(mvw)} warning(s))")
        for w in mvw[:5]:
            lines.append(f"      - {w}")
    else:
        lines.append(f"    Monitor Valid. : {mvs}")

    return "\n".join(lines) + "\n"


def enrich_summary(path: str, analysis: dict):
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    existing["per_gpu"] = analysis.get("per_gpu", {})
    existing["gpu_limits"] = analysis.get("gpu_limits", {})
    existing["workload_pattern"] = analysis.get("workload_pattern", "unknown")
    existing["anomalies"] = {
        "throttle_events": analysis.get("throttle_events", []),
        "imbalanced_gpus": analysis.get("imbalanced_gpus", []),
        "power_cap_saturated_pct": analysis.get("power_analysis", {}).get("cap_saturation_pct", 0),
        "temp_still_rising": analysis.get("mem_temp_analysis", {}).get("temp_still_rising", False),
    }
    existing["power_analysis"] = analysis.get("power_analysis", {})
    mt = analysis.get("mem_temp_analysis", {})
    existing["mem_temp_analysis"] = {k: v for k, v in mt.items() if k != "rising_gpus"}
    existing["monitoring_validation"] = analysis.get("monitoring_validation", {})
    existing["steady_state"] = analysis.get("steady_state", {})

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(run_dir: str, test_name: str) -> dict:
    rows = load_csv(os.path.join(run_dir, "power_temp.csv"))
    if not rows:
        return {}

    by_gpu = group_by_gpu(rows)
    if not by_gpu:
        return {}

    steady: dict[str, tuple[int, int]] = {}
    for gid, gr in by_gpu.items():
        steady[gid] = detect_steady_state(gr)

    pattern = detect_workload_pattern(by_gpu)
    per_gpu = compute_per_gpu(by_gpu, steady, pattern)
    imbalanced = detect_imbalance(per_gpu, pattern)
    throttle_events = detect_throttling(by_gpu, steady)
    power_info = analyze_power(by_gpu, steady)
    mem_info = analyze_mem_temp(by_gpu, steady)
    ss_metrics = compute_steady_state_metrics(by_gpu, steady)
    mon_val = validate_monitoring(test_name, per_gpu, pattern, run_dir)

    return {
        "gpu_limits": _get_gpu_limits(),
        "per_gpu": per_gpu,
        "workload_pattern": pattern,
        "throttle_events": throttle_events,
        "imbalanced_gpus": imbalanced,
        "power_analysis": power_info,
        "mem_temp_analysis": mem_info,
        "steady_state": ss_metrics,
        "monitoring_validation": mon_val,
    }


def main():
    ap = argparse.ArgumentParser(description="Enhanced GPU monitoring analysis")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-name", required=True)
    args = ap.parse_args()

    try:
        analysis = run_analysis(args.run_dir, args.test_name)
        if not analysis:
            return

        enrich_summary(os.path.join(args.run_dir, "summary.json"), analysis)

        text = health_checks_text(analysis)
        with open(os.path.join(args.run_dir, "health_checks.txt"), "w") as f:
            f.write(text)
    except Exception as e:
        print(f"  [analysis] non-fatal: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
