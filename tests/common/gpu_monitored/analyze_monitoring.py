# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#!/usr/bin/env python3
"""
Enhanced monitoring analysis for GPU test runs.

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

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    # Prefer the shared schema when importable (i.e. when this module is
    # used as part of ``tests.common.gpu_monitored``). Keep a local
    # fallback so ``python -m tests.common.gpu_monitored.analyze_monitoring``
    # stand-alone invocation still works if the relative import fails.
    from tests.common.gpu_monitored import csv_schema as _csv_schema  # type: ignore
    from tests.common.gpu_monitored.monitoring import _MAX_CORRUPT_ROWS  # type: ignore
except ImportError:  # pragma: no cover - script-style invocation
    # Same bound the monitoring reader uses; duplicated only for the
    # stand-alone invocation path, where the relative import is unavailable.
    _MAX_CORRUPT_ROWS = 100

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
# Minimum steady-state samples before a GPU's average is comparable against
# the rest of the fleet. ``detect_steady_state`` deliberately allows a
# one-sample steady state (see its docstring), which is fine for reporting
# that GPU on its own but makes a cross-GPU average meaningless. Mirrors the
# ``len(ss) < 5`` guard detect_throttling already applies.
IMBALANCE_MIN_SAMPLES = 5
# Minimum absolute spread (max - min across GPUs) before a metric's relative
# deviation is worth reporting. Relative deviation explodes when the whole
# fleet sits at a near-identical low value: a 4-GPU MI300A transferbench run
# measured 1.4/1.6/1.8/2.0% GFX util -- a 0.6-point spread -- yet cleared the
# 15% relative bar because the mean was only 1.7. Only ``gfx_util`` (a
# percentage) needs a floor; power (W) and clock (MHz) spreads are already
# meaningful at their own scale, so they keep their existing behaviour.
IMBALANCE_MIN_SPREAD = {"gfx_util": 5.0}
THROTTLE_CLK_DROP_PCT = 20.0
POWER_CAP_RATIO = 0.99
TEMP_SLOPE_WARN_C_PER_MIN = 0.1
STEADY_UTIL_THRESH = 50.0
RAMP_UTIL_THRESH = 80.0

def _get_workload_profile(test_name: str) -> Optional[Dict[str, Any]]:
    """Return the monitoring profile for ``test_name`` from ``TestSpec``.

    The profile dict lives on each test's ``TestSpec.workload_profile``
    attribute (see ``tests/base.py``). We used to maintain a parallel
    ``WORKLOAD_PROFILES`` dict here, which drifted in practice (the
    ``inference_server_stress`` values differed from the TestSpec for
    months before anyone noticed). Keeping one source of truth removes
    that class of bug.

    A test whose ``TestSpec`` sets no ``workload_profile`` returns
    ``None`` (monitoring validation is skipped). Unknown test names also
    return ``None``.
    """
    try:
        from tests.common.gpu_monitored.workloads import get_test
    except ImportError:
        # Standalone ``python -m tests.common.gpu_monitored.analyze_monitoring``
        # invocations go through the package anyway; the ImportError
        # branch only fires if the package layout is broken.
        return None
    t = get_test(test_name)
    return t.spec.workload_profile if t is not None else None


def _query_gpu_limits() -> Dict[str, float]:
    """Query thermal/power limits from amd-smi static. Falls back to
    conservative defaults if amd-smi is unavailable."""
    from framework.executors.local_executor import run_cmd_get_stdout_stderr
    from framework.rocm.libs.amd_smi import _to_scalar, _unwrap_entries

    limits = {
        "throttle_temp_c": FALLBACK_THROTTLE_TEMP_C,
        "mem_temp_warn_c": FALLBACK_MEM_TEMP_WARN_C,
    }
    amd_smi = "amd-smi"
    rocm_path = os.environ.get("ROCM_PATH", "")
    if rocm_path and os.path.isfile(os.path.join(rocm_path, "bin", "amd-smi")):
        amd_smi = os.path.join(rocm_path, "bin", "amd-smi")
    try:
        rc, stdout, _stderr = run_cmd_get_stdout_stderr(
            amd_smi, "static", "-g", "0", "--json", timeout=10, quiet=True
        )
        if rc != 0:
            raise RuntimeError(f"amd-smi static exited {rc}")
        data = json.loads(stdout)
        entries = _unwrap_entries(data)
        gpu0 = entries[0] if entries else (data if isinstance(data, dict) else {})
        lim = gpu0.get("limit", {})
        field_map = {
            "slowdown_hotspot_temperature": "throttle_temp_c",
            "slowdown_vram_temperature": "mem_temp_warn_c",
            "shutdown_hotspot_temperature": "shutdown_temp_c",
            "shutdown_vram_temperature": "shutdown_mem_temp_c",
        }
        for json_key, limit_key in field_map.items():
            val = _to_scalar(lim.get(json_key))
            if val is not None:
                limits[limit_key] = float(val)

        ppt0 = lim.get("ppt0", {})
        max_pwr = _to_scalar(ppt0.get("max_power_limit"))
        if max_pwr is not None:
            limits["max_power_w"] = float(max_pwr)

        clk = gpu0.get("clock", {})
        for clk_domain, limit_key in [("sys", "max_gfx_clk_mhz"), ("mem", "max_mem_clk_mhz")]:
            levels = clk.get(clk_domain, {})
            if isinstance(levels, dict):
                freq_levels = levels.get("frequency_levels", {})
                if freq_levels:
                    vals = []
                    for v in freq_levels.values():
                        scalar = _to_scalar(v)
                        if scalar is not None and scalar > 0:
                            vals.append(float(scalar))
                        elif isinstance(v, str):
                            try:
                                vals.append(float(v.split()[0]))
                            except (ValueError, IndexError):
                                pass
                    if vals:
                        max_val = max(vals)
                        if max_val > 0:
                            limits[limit_key] = max_val
    except Exception:
        pass

    if "max_gfx_clk_mhz" not in limits:
        try:
            rc, stdout, _stderr = run_cmd_get_stdout_stderr(
                amd_smi, "metric", "-c", "-g", "0", "--json", timeout=10, quiet=True
            )
            if rc != 0:
                raise RuntimeError(f"amd-smi metric exited {rc}")
            data = json.loads(stdout)
            entries = _unwrap_entries(data)
            gpu0 = entries[0] if entries else (data if isinstance(data, dict) else {})
            clk_info = gpu0.get("clock", {})
            gfx0 = clk_info.get("gfx_0", {})
            val = _to_scalar(gfx0.get("max_clk"))
            if val is not None:
                limits["max_gfx_clk_mhz"] = float(val)
        except Exception:
            pass
    return limits

_GPU_LIMITS_CACHE: Optional[Dict[str, float]] = None


def _get_gpu_limits() -> Dict[str, float]:
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


def _stats(vals: List[float]) -> Optional[Dict[str, Any]]:
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


def _linreg(xs: List[float], ys: List[float]) -> Tuple[float, float, float]:
    """OLS linear regression. Returns (slope, intercept, r_squared)."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0, 0.0
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    mean_y = sy / n
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return slope, intercept, r2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path: str) -> List[Dict[str, Any]]:
    """Rows only. Use ``load_csv_with_stats`` when the caller reports coverage."""
    return load_csv_with_stats(path)[0]


def load_csv_with_stats(
    path: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Rows plus how much of the file had to be skipped.

    Dropping unusable rows keeps one garbled monitor write from costing a
    healthy run its whole analysis, but silently shortening the input makes
    every number downstream describe less of the run than it appears to.
    The counts travel with the rows so ``run_analysis`` can say so in
    ``health_checks.txt`` and ``summary.json`` -- the artifacts a triager
    reads as authoritative -- mirroring ``MonitoringEvidence.scan_aborted``
    on the collector side.
    """
    stats: Dict[str, Any] = {"undecodable_rows": 0, "scan_aborted": False}
    if not os.path.exists(path):
        return [], stats
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        undecodable = 0
        while True:
            # A row the reader itself cannot decode (a field beyond
            # field_size_limit, i.e. a partial or interleaved monitor write)
            # raises out of the iterator. Unguarded, that exception left
            # ``_safe_analyze`` to fail the whole enhanced-analysis step, so a
            # single bad row cost the run its throttle, imbalance, power and
            # thermal reporting *and* failed an otherwise healthy test --
            # the same "one bad row aborts everything" behaviour already fixed
            # in ``monitoring.collect_monitoring_evidence``. Skip the row and
            # keep the rest, bounded so an undecodable row cannot spin.
            if undecodable > _MAX_CORRUPT_ROWS:
                print(f"  [analyze] WARNING: stopped reading {path} after more "
                      f"than {_MAX_CORRUPT_ROWS} undecodable rows; analysis "
                      f"covers only the rows before that point")
                stats["scan_aborted"] = True
                break
            try:
                r = next(reader)
            except StopIteration:
                break
            except csv.Error:
                undecodable += 1
                continue
            # Skip rows with an unparseable/missing timestamp — those are
            # garbage lines from amd-smi (partial flush, header echo) and
            # they'd otherwise all collapse onto t=0 which breaks the
            # steady-state detector.
            try:
                t = int(r.get(_csv_schema.TIMESTAMP, "0"))
            except (ValueError, TypeError):
                continue
            vram_used = _sf(r.get(_csv_schema.VRAM_USED))
            vram_total = _sf(r.get(_csv_schema.VRAM_TOTAL))
            vram_pct_raw = r.get(_csv_schema.VRAM_PCT)
            vram_pct_text = str(vram_pct_raw or "").strip()
            vram_pct = _sf(vram_pct_raw) if vram_pct_text not in ("", "N/A") else (
                (vram_used / vram_total * 100) if vram_total > 0 else 0
            )
            rows.append({
                "t": t,
                "gpu": str(r.get(_csv_schema.GPU) or "0"),
                "power": _sf(r.get(_csv_schema.POWER_USAGE)),
                "max_power": _sf(r.get(_csv_schema.MAX_POWER), 1000),
                "hotspot_temp": _sf(r.get(_csv_schema.HOTSPOT_TEMP)),
                "mem_temp": _sf(r.get(_csv_schema.MEM_TEMP)),
                "gfx_clk": _sf(r.get(_csv_schema.GFX_CLK)),
                "gfx_util": _sf(r.get(_csv_schema.GFX_UTIL)),
                "mem_util": _sf(r.get(_csv_schema.MEM_UTIL)),
                "mem_clk": _sf(r.get(_csv_schema.MEM_CLK)),
                "vram_used": vram_used,
                "vram_total": vram_total,
                "vram_pct": vram_pct,
            })
    stats["undecodable_rows"] = undecodable
    return rows, stats


def group_by_gpu(rows: List[Dict]) -> Dict[str, List[Dict]]:
    by_gpu: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_gpu[r["gpu"]].append(r)
    return dict(by_gpu)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def detect_steady_state(gpu_rows: List[Dict]) -> Tuple[int, int]:
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

    start: Optional[int] = None
    for i in range(n):
        if gpu_rows[i]["gfx_util"] >= RAMP_UTIL_THRESH:
            start = i
            break

    if start is None:
        # No sample ever hit the ramp threshold — use the full range so
        # downstream `rows[s:e+1]` still yields data.
        return 0, max(0, n - 1)

    end = start
    for i in range(n - 1, start - 1, -1):
        if gpu_rows[i]["gfx_util"] >= RAMP_UTIL_THRESH:
            end = i
            break

    return start, end


def detect_workload_pattern(by_gpu: Dict[str, List[Dict]]) -> str:
    if len(by_gpu) <= 1:
        return "single_gpu"

    ts_set = sorted({r["t"] for rows in by_gpu.values() for r in rows})
    if len(ts_set) < 3:
        return "unknown"

    # Count active GPUs per timestamp across the full timeline. Timestamps
    # with zero active GPUs were previously excluded, which biased the
    # median high: a workload that sat idle 95% of the time with brief
    # parallel bursts was labelled "parallel" because the median was
    # computed only over the 5% of active samples. Including the idle
    # samples makes the median reflect the actual duty cycle.
    gpu_at_ts: Dict[int, int] = defaultdict(int)
    for gpu_id, rows in by_gpu.items():
        for r in rows:
            if r["gfx_util"] >= STEADY_UTIL_THRESH:
                gpu_at_ts[r["t"]] += 1

    active_counts = sorted(gpu_at_ts.get(t, 0) for t in ts_set)
    median = active_counts[len(active_counts) // 2]
    num_gpus = len(by_gpu)

    if median == 0:
        # Describe what was measured, not a state we cannot observe from GFX
        # utilisation alone. "idle" was actively wrong for data-movement and
        # burst workloads: an 8x MI325X transferbench run sat at 2090 MHz and
        # ~205W (vs ~140W at rest) moving 2 GB/s aggregate, yet was labelled
        # idle purely because no GPU sustained STEADY_UTIL_THRESH. A reader
        # triaging that report would conclude the workload never ran.
        return "low_gfx_util"
    if median <= 1:
        return "serial"
    if median >= num_gpus * 0.7:
        return "parallel"
    return "mixed"


def compute_per_gpu(by_gpu: Dict[str, List[Dict]],
                    steady: Dict[str, Tuple[int, int]],
                    pattern: str,
                    serial_profile: bool = False) -> Dict[str, Dict]:
    result = {}
    for gpu_id in sorted(by_gpu, key=lambda g: int(g) if g.isdigit() else 0):
        rows = by_gpu[gpu_id]
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        ss = rows[s:e + 1] if e >= s else rows

        if serial_profile or pattern == "serial":
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


def comparable_gpu_count(per_gpu: Dict[str, Dict],
                         exclusions: List[Dict]) -> int:
    """How many GPUs actually took part in the cross-GPU comparison.

    Only per-GPU exclusions reduce the count. A spread-floor entry withholds
    one *metric* across the whole fleet while every GPU still participates, so
    counting it here under-reported eligibility and could record a comparison
    that did happen as ``inconclusive``.

    Shared by ``run_analysis`` (which stores the verdict) and
    ``health_checks_text`` (which renders it) so the two cannot disagree --
    they previously carried separate copies of this arithmetic and did.
    """
    dropped = sum(
        1 for e in (exclusions or []) if e.get("metric") == "excluded"
    )
    return max(0, len(per_gpu or {}) - dropped)


def _eligible_avgs(per_gpu: Dict[str, Dict], metric: str) -> Dict[str, float]:
    """Per-GPU averages for ``metric``, keeping only comparable GPUs.

    A GPU whose steady window collapsed to a handful of samples has an
    "average" over that blip, not over the run. Comparing it with GPUs
    averaged over hundreds of samples invents an imbalance: on an 8x MI325X
    transferbench run GPU 2's window was a single 91% sample against seven
    305-sample GPUs at ~0.8%, which pulled the fleet mean to 12.1 and
    flagged all eight GPUs (up to 655%).

    An absent sample count is treated as long enough, so a stats dict
    missing the field can never silently suppress a real imbalance. Shared
    with ``find_imbalance_exclusions`` so the eligibility rule and the
    report of what it excluded cannot disagree.
    """
    avgs: Dict[str, float] = {}
    for gid, st in per_gpu.items():
        samples = st.get("steady_state_samples", IMBALANCE_MIN_SAMPLES)
        if samples < IMBALANCE_MIN_SAMPLES:
            continue
        s = st.get(metric)
        if s:
            avg = s.get("avg")
            if isinstance(avg, (int, float)) and math.isfinite(avg) and avg >= 0:
                avgs[gid] = avg
    return avgs


def detect_imbalance(per_gpu: Dict[str, Dict], pattern: str,
                     serial_profile: bool = False) -> List[Dict]:
    # Cross-GPU imbalance is meaningless for a workload that drives one GPU at
    # a time: the busy GPU necessarily deviates hugely from the idle ones.
    # Honour the *declared* profile as well as the detected pattern -- a serial
    # test whose utilisation stays below STEADY_UTIL_THRESH is classified
    # "low_gfx_util", not "serial", so the pattern check alone missed it and flagged
    # the working GPU (hipblaslt_bench on MI325X: one GPU at 1112 MHz against a
    # 286 MHz fleet mean -> a bogus 288% deviation on a healthy run).
    if serial_profile or pattern == "serial" or len(per_gpu) <= 1:
        return []
    flags = []
    for metric in ("power", "gfx_clk", "gfx_util"):
        avgs = _eligible_avgs(per_gpu, metric)
        if len(avgs) < 2:
            continue
        fleet_mean = sum(avgs.values()) / len(avgs)
        if fleet_mean < 1:
            continue
        spread_floor = IMBALANCE_MIN_SPREAD.get(metric, 0.0)
        if spread_floor and (max(avgs.values()) - min(avgs.values())) < spread_floor:
            continue
        for gid, avg in avgs.items():
            dev = abs(avg - fleet_mean) / fleet_mean * 100
            if dev > IMBALANCE_PCT:
                flags.append({
                    "gpu": gid, "metric": metric,
                    "gpu_avg": round(avg, 1),
                    "fleet_mean": round(fleet_mean, 1),
                    "deviation_pct": round(dev, 1),
                })
    return flags


def find_imbalance_exclusions(
    per_gpu: Dict[str, Dict],
    pattern: str,
    serial_profile: bool = False,
) -> List[Dict]:
    """Report everything held back from the cross-GPU comparison.

    Two things can hold a reading back, and both have to be visible or a
    real (if small) imbalance disappears without trace:

    * a GPU whose steady window is too short to average, and
    * a metric whose absolute spread across the fleet is below its floor,
      which suppresses the *whole metric* rather than one GPU.
    """
    if serial_profile or pattern == "serial":
        return []
    exclusions = []
    for gid, stats in sorted(per_gpu.items()):
        samples = stats.get("steady_state_samples", IMBALANCE_MIN_SAMPLES)
        if samples < IMBALANCE_MIN_SAMPLES:
            exclusions.append({
            "gpu": gid, "metric": "excluded",
            "steady_state_samples": samples,
            "note": (f"excluded from cross-GPU comparison: steady window has "
                     f"{samples} sample(s), below the {IMBALANCE_MIN_SAMPLES} "
                     f"needed for a meaningful average"),
        })

    # Spread-floor suppression. Reported only when it actually withheld a
    # deviation that cleared the relative bar -- otherwise every healthy run
    # would carry a note about a metric that had nothing to say.
    for metric, spread_floor in sorted(IMBALANCE_MIN_SPREAD.items()):
        if not spread_floor:
            continue
        avgs = _eligible_avgs(per_gpu, metric)
        if len(avgs) < 2:
            continue
        fleet_mean = sum(avgs.values()) / len(avgs)
        if fleet_mean < 1:
            continue
        spread = max(avgs.values()) - min(avgs.values())
        if spread >= spread_floor:
            continue
        withheld = sorted(
            gid for gid, avg in avgs.items()
            if abs(avg - fleet_mean) / fleet_mean * 100 > IMBALANCE_PCT
        )
        if not withheld:
            continue
        exclusions.append({
            "gpu": ", ".join(str(g) for g in withheld),
            "metric": metric,
            "spread": round(spread, 2),
            "spread_floor": spread_floor,
            "note": (f"{metric} deviation on GPU(s) "
                     f"{', '.join(str(g) for g in withheld)} not reported: "
                     f"fleet spread {spread:.2f} is below the "
                     f"{spread_floor} floor, so the relative deviation is "
                     f"not meaningful at this scale"),
        })
    return exclusions


def detect_throttling(by_gpu: Dict[str, List[Dict]],
                      steady: Dict[str, Tuple[int, int]]) -> List[Dict]:
    events = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        ss = rows[s:e + 1]
        if len(ss) < 5:
            continue
        clks = [r["gfx_clk"] for r in ss if r["gfx_clk"] > 0]
        if not clks:
            continue
        avg_clk = sum(clks) / len(clks)
        floor = avg_clk * (1 - THROTTLE_CLK_DROP_PCT / 100)
        hit = sum(1 for r in ss if r["hotspot_temp"] >= _get_gpu_limits()["throttle_temp_c"] and r["gfx_clk"] < floor)
        if hit > 0:
            events.append({
                "gpu": gpu_id,
                "throttle_samples": hit,
                "total_steady_samples": len(ss),
                "throttle_pct": round(hit / len(ss) * 100, 1),
                "max_temp_c": round(max(r["hotspot_temp"] for r in ss), 1),
                "ss_avg_clk_mhz": round(avg_clk, 0),
            })
    return events


def analyze_power(by_gpu: Dict[str, List[Dict]],
                  steady: Dict[str, Tuple[int, int]]) -> Dict:
    all_ss = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        all_ss.extend(rows[s:e + 1])
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
        ss_pwr = [r["power"] for r in frows[fs:fe + 1]]
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


def analyze_mem_temp(by_gpu: Dict[str, List[Dict]],
                     steady: Dict[str, Tuple[int, int]]) -> Dict:
    all_ss = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        all_ss.extend(rows[s:e + 1])
    if not all_ss:
        return {}

    mem_t = [r["mem_temp"] for r in all_ss if r["mem_temp"] > 0]
    deltas = [r["hotspot_temp"] - r["mem_temp"] for r in all_ss
              if r["hotspot_temp"] > 0 and r["mem_temp"] > 0]

    result: Dict[str, Any] = {
        "mem_temp": _stats(mem_t),
        "hotspot_mem_delta": _stats(deltas),
        "mem_temp_exceeds_threshold": bool(mem_t and max(mem_t) >= _get_gpu_limits()["mem_temp_warn_c"]),
    }

    rising_gpus = []
    for gpu_id, rows in by_gpu.items():
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        ss = rows[s:e + 1]
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


def compute_steady_state_metrics(by_gpu: Dict[str, List[Dict]],
                                 steady: Dict[str, Tuple[int, int]]) -> Dict:
    """Aggregate steady-state samples across all GPUs and compute
    ``start_offset_s``/``end_offset_s`` relative to the *global* t0.

    Using a global t0 (min timestamp across all GPUs) means the start/end
    offsets are directly comparable across GPUs. The previous version
    subtracted each GPU's own ``rows[0]["t"]``, so staggered GPU start
    times produced apples-to-oranges offsets; in practice amd-smi
    samples all GPUs together, but the fix future-proofs the report.
    """
    all_ss = []
    # Compute a single global t0 across every GPU's first sample so the
    # per-GPU offsets live on the same timeline.
    all_rows = [r for rows in by_gpu.values() for r in rows]
    if not all_rows:
        return {}
    global_t0 = min(r["t"] for r in all_rows)

    earliest: Optional[int] = None
    latest: Optional[int] = None
    for gpu_id, rows in by_gpu.items():
        if not rows:
            continue
        s, e = steady.get(gpu_id, (0, max(0, len(rows) - 1)))
        e = min(e, len(rows) - 1)
        s = max(0, min(s, e))
        ss = rows[s:e + 1]
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


def validate_monitoring(test_name: str, per_gpu: Dict, pattern: str,
                        run_dir: str = "") -> Dict:
    """Check monitoring data against expected workload profiles.
    Returns warnings only — never changes pass/fail.

    Skips monitoring validation when:
    - ``summary.json`` marks the run as ``unsupported`` (the test itself
      decided the environment didn't qualify)
    - The run was very short and failed (we don't have enough signal to
      meaningfully validate monitoring profiles)
    """
    if run_dir:
        try:
            sp = os.path.join(run_dir, "summary.json")
            if os.path.exists(sp):
                with open(sp) as _f:
                    sd = json.load(_f)
                ec = sd.get("exit_code", 0)
                dur = sd.get("duration_seconds", 0)
                # The runner now writes an explicit ``unsupported`` flag;
                # prefer it over the previous dead-code check for
                # ``exit_code == 200`` (no test ever set that value).
                if sd.get("unsupported", False):
                    return {"status": "SKIPPED", "reason": "test returned UNSUPPORTED", "warnings": []}
                if dur < 5 and ec != 0:
                    return {"status": "SKIPPED", "reason": f"test too short ({dur}s) with exit {ec}", "warnings": []}
        except (json.JSONDecodeError, IOError, KeyError):
            pass

    profile = _get_workload_profile(test_name)
    if profile is None:
        return {"status": "SKIPPED", "reason": "no expected profile for this test", "warnings": []}

    warnings = []
    is_serial = profile.get("serial", False) or pattern == "serial"

    if is_serial:
        # For serial workloads, stats in per_gpu are already computed over
        # each GPU's active window (see compute_per_gpu). We want to catch
        # both "no GPU ever activated" AND "every GPU that activated did
        # so weakly" — the original code only did the first.
        min_util = profile.get("min_util", 0)
        min_vram_pct = profile.get("min_vram_pct", 0)
        active = [(gid, st) for gid, st in per_gpu.items()
                  if st.get("active_samples", 0) > 0]
        if not active:
            # ``active_samples`` counts only samples at or above
            # STEADY_UTIL_THRESH, so this probe is purely a GFX-utilisation
            # signal. ``min_util == 0`` is a test declaring that GFX util is
            # not its health signal, in which case the absence of high-util
            # samples is expected rather than a defect -- hipblaslt_bench runs
            # each GEMM shape as its own short-lived process and spends
            # microseconds in the kernels against tens of milliseconds of
            # host-side setup, so a healthy run never reaches the threshold at
            # a 1 Hz sample rate. The non-serial branch already no-ops on
            # ``min_util == 0``; match that here instead of contradicting the
            # profile.
            if min_util > 0:
                warnings.append("No GPU had any active samples during the test")
        else:
            any_meets_util = False
            for gid, st in active:
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
                    f"No GPU's active window reached expected "
                    f"{min_util}% gfx util (best was {worst}% avg)"
                )
            if min_vram_pct > 0:
                any_meets_vram = False
                for gid, st in active:
                    vram = st.get("vram_pct")
                    if vram and vram["max"] >= min_vram_pct:
                        any_meets_vram = True
                        break
                if not any_meets_vram:
                    warnings.append(
                        f"No GPU reached expected {min_vram_pct}% max VRAM"
                    )
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
def health_checks_text(a: Dict) -> str:
    lines = ["  Health Checks:"]

    # State how much of the telemetry these checks are based on before
    # reporting the checks themselves, so a truncated input cannot be read
    # as a clean bill of health.
    dropped = a.get("undecodable_rows") or 0
    if a.get("scan_aborted"):
        unavailable = a.get("analysis_unavailable", False)
        lines.append(
            f"    Data Coverage  : WARNING (stopped reading power_temp.csv "
            f"after {dropped} undecodable row(s); "
            + ("no usable rows survived, so analysis is unavailable)"
               if unavailable else
               "the checks below cover only the rows before that point)")
        )
    elif dropped:
        lines.append(
            f"    Data Coverage  : OK ({dropped} undecodable row(s) skipped)"
        )

    if a.get("analysis_unavailable"):
        return "\n".join(lines) + "\n"

    thr = a.get("throttle_events", [])
    if thr:
        worst = max(e["throttle_pct"] for e in thr)
        lines.append(f"    Throttling     : WARNING ({len(thr)} GPU(s), up to {worst}% of steady state)")
    else:
        lines.append("    Throttling     : OK")

    # Keep a compatibility fallback for summaries written before exclusions
    # moved to their own field.
    all_imb = a.get("imbalanced_gpus", [])
    imb = [i for i in all_imb if not i.get("low_confidence")]
    skipped = a.get(
        "imbalance_exclusions",
        [i for i in all_imb if i.get("low_confidence")],
    )
    pat = a.get("workload_pattern", "unknown")
    short_window = [i for i in skipped if i.get("metric") == "excluded"]
    floored = [i for i in skipped if i.get("spread_floor") is not None]
    comparison_status = a.get("imbalance_comparison_status")
    if comparison_status is None:
        eligible = comparable_gpu_count(a.get("per_gpu", {}), skipped)
        comparison_status = "compared" if eligible >= 2 else "inconclusive"
    if imb:
        mx = max(i["deviation_pct"] for i in imb)
        lines.append(f"    GPU Imbalance  : WARNING (max deviation {mx}%)")
    elif pat == "serial" or a.get("serial_profile"):
        lines.append("    GPU Imbalance  : N/A (serial workload)")
    elif comparison_status == "inconclusive":
        lines.append("    GPU Imbalance  : N/A (fewer than 2 comparable GPUs)")
    else:
        lines.append("    GPU Imbalance  : OK")
    if skipped:
        # Two kinds share this field: a GPU dropped for a short steady
        # window, and a metric whose spread was too small to read anything
        # into. Render them separately so neither is described as the other.
        if short_window:
            ids = ", ".join(str(i["gpu"]) for i in short_window)
            lines.append(f"    Imbalance note : GPU(s) {ids} excluded — steady "
                         f"window under {IMBALANCE_MIN_SAMPLES} samples")
        for item in floored:
            lines.append(f"    Imbalance note : {item['metric']} deviation on "
                         f"GPU(s) {item['gpu']} withheld — fleet spread "
                         f"{item['spread']} below the "
                         f"{item['spread_floor']} floor")

    n_gpus = len(a.get("per_gpu", {}))
    lines.append(f"    Workload       : {pat} ({n_gpus} GPU(s))")

    mt = a.get("mem_temp_analysis", {})
    ms = mt.get("mem_temp")
    if ms:
        ds = mt.get("hotspot_mem_delta")
        d_avg = ds["avg"] if ds else 0
        exceeds = mt.get("mem_temp_exceeds_threshold", False)
        tag = "WARNING" if exceeds else "OK"
        lines.append(f"    Mem Temp       : {tag} (max {ms['max']}C, delta to hotspot: {d_avg}C)")
    else:
        lines.append("    Mem Temp       : N/A")

    if mt.get("temp_still_rising"):
        rg = mt.get("rising_gpus", [])
        sl = ", ".join(f"GPU{g['gpu']}:+{g['slope_c_per_min']}C/min" for g in rg[:3])
        lines.append(f"    Temp Trend     : WARNING (still rising: {sl})")
    else:
        lines.append("    Temp Trend     : STABLE")

    pw = a.get("power_analysis", {})
    cu = pw.get("cap_utilization_pct", 0)
    cs = pw.get("cap_saturation_pct", 0)
    cw = pw.get("avg_power_cap_w", 0)
    if cu > 0:
        lines.append(f"    Power Cap      : {cu}% avg, {cs}% at cap ({cw:.0f}W cap)")
    else:
        lines.append("    Power Cap      : N/A")

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


def enrich_summary(path: str, analysis: Dict):
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
        if not isinstance(existing, dict):
            raise ValueError("summary.json root must be an object")

    existing["per_gpu"] = analysis.get("per_gpu", {})
    existing["gpu_limits"] = analysis.get("gpu_limits", {})
    existing["workload_pattern"] = analysis.get("workload_pattern", "unknown")
    existing["serial_profile"] = bool(analysis.get("serial_profile", False))
    existing["anomalies"] = {
        "throttle_events": analysis.get("throttle_events", []),
        # Keep legacy low-confidence exclusion records in this established
        # field during the schema transition. New consumers should prefer
        # ``imbalance_exclusions``; old consumers still see the diagnostics
        # they received before that sibling field existed.
        "imbalanced_gpus": analysis.get("imbalanced_gpus", []) + [
            {**entry, "low_confidence": True}
            for entry in analysis.get("imbalance_exclusions", [])
        ],
        "imbalance_exclusions": analysis.get("imbalance_exclusions", []),
        "imbalance_comparison_status": analysis.get(
            "imbalance_comparison_status", "compared",
        ),
        "power_cap_saturated_pct": analysis.get("power_analysis", {}).get("cap_saturation_pct", 0),
        "temp_still_rising": analysis.get("mem_temp_analysis", {}).get("temp_still_rising", False),
    }
    existing["power_analysis"] = analysis.get("power_analysis", {})
    mt = analysis.get("mem_temp_analysis", {})
    existing["mem_temp_analysis"] = {k: v for k, v in mt.items() if k != "rising_gpus"}
    existing["monitoring_validation"] = analysis.get("monitoring_validation", {})
    existing["steady_state"] = analysis.get("steady_state", {})
    existing["analysis_input"] = {
        "undecodable_rows": analysis.get("undecodable_rows", 0),
        "scan_aborted": bool(analysis.get("scan_aborted", False)),
        "analysis_unavailable": bool(
            analysis.get("analysis_unavailable", False)
        ),
    }

    temp_path = f"{path}.tmp"
    with open(temp_path, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(temp_path, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(run_dir: str, test_name: str) -> Dict:
    rows, csv_stats = load_csv_with_stats(
        os.path.join(run_dir, "power_temp.csv")
    )
    if not rows:
        if csv_stats["scan_aborted"]:
            return {
                "undecodable_rows": csv_stats["undecodable_rows"],
                "scan_aborted": True,
                "analysis_unavailable": True,
            }
        return {}

    by_gpu = group_by_gpu(rows)
    if not by_gpu:
        return {}

    steady = {}
    for gid, gr in by_gpu.items():
        steady[gid] = detect_steady_state(gr)

    pattern = detect_workload_pattern(by_gpu)
    serial_profile = bool((_get_workload_profile(test_name) or {}).get("serial"))
    per_gpu = compute_per_gpu(
        by_gpu, steady, pattern, serial_profile=serial_profile,
    )
    imbalanced = detect_imbalance(per_gpu, pattern, serial_profile=serial_profile)
    exclusions = find_imbalance_exclusions(
        per_gpu, pattern, serial_profile=serial_profile,
    )
    eligible_gpus = comparable_gpu_count(per_gpu, exclusions)
    comparison_status = (
        "not_applicable"
        if serial_profile or pattern == "serial"
        else "compared" if eligible_gpus >= 2 else "inconclusive"
    )
    throttle_events = detect_throttling(by_gpu, steady)
    power_info = analyze_power(by_gpu, steady)
    mem_info = analyze_mem_temp(by_gpu, steady)
    ss_metrics = compute_steady_state_metrics(by_gpu, steady)
    mon_val = validate_monitoring(test_name, per_gpu, pattern, run_dir)

    return {
        "gpu_limits": _get_gpu_limits(),
        "per_gpu": per_gpu,
        "workload_pattern": pattern,
        "serial_profile": serial_profile,
        "throttle_events": throttle_events,
        "imbalanced_gpus": imbalanced,
        "imbalance_exclusions": exclusions,
        "imbalance_comparison_status": comparison_status,
        "power_analysis": power_info,
        "mem_temp_analysis": mem_info,
        "steady_state": ss_metrics,
        "monitoring_validation": mon_val,
        # How much of power_temp.csv these numbers actually describe.
        "undecodable_rows": csv_stats["undecodable_rows"],
        "scan_aborted": csv_stats["scan_aborted"],
        "analysis_unavailable": False,
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
