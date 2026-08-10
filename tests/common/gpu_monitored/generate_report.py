# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Generate an HTML report with interactive SVG time-series charts from amd-smi monitoring CSV.

Charts have JavaScript crosshair + tooltip on hover showing exact values at each timestamp.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time as _time
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from typing import Any, Optional

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


def _get_system_tz() -> tuple:
    """Return (utc_offset_seconds, tz_abbreviation) for the system's local
    timezone. Uses timezone-aware ``datetime.fromtimestamp`` so it runs
    clean on Python 3.12+ (``datetime.utcfromtimestamp`` is deprecated
    there and emits a DeprecationWarning)."""
    now = _time.time()
    local = datetime.fromtimestamp(now).replace(tzinfo=None)
    utc = datetime.fromtimestamp(now, tz=timezone.utc).replace(tzinfo=None)
    offset_sec = int((local - utc).total_seconds())
    tz_name = _time.strftime("%Z") or "UTC"
    return offset_sec, tz_name


SYS_TZ_OFFSET, SYS_TZ_NAME = _get_system_tz()

GPU_COLORS = [
    "#3b82f6",
    "#ef4444",
    "#22c55e",
    "#f59e0b",
    "#8b5cf6",
    "#ec4899",
    "#06b6d4",
    "#f97316",
]

COMBINED_METRICS = [
    ("power_pct", "Power %", "#ef4444"),
    ("temp_pct", "Temperature %", "#f59e0b"),
    ("gfx_clk_pct", "GFX Clock %", "#3b82f6"),
    ("gfx_util", "GFX Util %", "#22c55e"),
    ("vram_pct", "VRAM Util %", "#8b5cf6"),
    ("mem_engine", "MEM Engine %", "#06b6d4"),
    ("mem_clk_pct", "MEM Clock %", "#ec4899"),
]

_chart_id_counter = 0


def _next_chart_id() -> str:
    global _chart_id_counter  # noqa: PLW0603
    _chart_id_counter += 1
    return f"chart{_chart_id_counter}"


def _load_csv(path: str) -> list[dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_command_metadata(run_dir: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    cmd_file = os.path.join(run_dir, "command.txt")
    if os.path.exists(cmd_file):
        with open(cmd_file) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
    return meta


def _sf(v, default=0.0):
    """Safe float conversion — returns default for empty/N/A/NaN/Inf/invalid."""
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


def _parse_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for r in rows:
        ts = r.get(_csv_schema.TIMESTAMP, "0")
        if not ts or ts.strip() in ("", "N/A"):
            continue
        vram_total = _sf(r.get(_csv_schema.VRAM_TOTAL), 0)
        vram_used = _sf(r.get(_csv_schema.VRAM_USED), 0)
        vram_pct_raw = r.get(_csv_schema.VRAM_PCT, "")
        vram_pct = (
            _sf(vram_pct_raw)
            if vram_pct_raw.strip() not in ("", "N/A")
            else ((vram_used / vram_total * 100) if vram_total > 0 else 0)
        )
        parsed.append(
            {
                "t": int(_sf(ts, 0)),
                "gpu": r.get(_csv_schema.GPU, "0"),
                "power": _sf(r.get(_csv_schema.POWER_USAGE)),
                "max_power": _sf(r.get(_csv_schema.MAX_POWER), 1000),
                "hotspot_temp": _sf(r.get(_csv_schema.HOTSPOT_TEMP)),
                "mem_temp": _sf(r.get(_csv_schema.MEM_TEMP)),
                "gfx_clk": _sf(r.get(_csv_schema.GFX_CLK)),
                "gfx_util": _sf(r.get(_csv_schema.GFX_UTIL)),
                "mem_clk": _sf(r.get(_csv_schema.MEM_CLK)),
                "mem_engine": _sf(r.get(_csv_schema.MEM_UTIL)),
                "vram_used": vram_used,
                "vram_total": vram_total,
                "vram_pct": vram_pct,
            }
        )
    return parsed


def _compute_pct_fields(rows: list[dict[str, Any]], gpu_limits: Optional[dict] = None) -> list[dict[str, Any]]:
    if not rows:
        return rows
    gl = gpu_limits or {}
    temp_ref = gl.get("throttle_temp_c", 100.0) or 100.0
    hw_gfx = gl.get("max_gfx_clk_mhz", 0)
    hw_mem = gl.get("max_mem_clk_mhz", 0)
    max_gfx_clk = hw_gfx if hw_gfx > 0 else (max((r["gfx_clk"] for r in rows), default=1) or 1)
    max_mem_clk = hw_mem if hw_mem > 0 else (max((r["mem_clk"] for r in rows), default=1) or 1)
    hw_power = gl.get("max_power_w", 0)
    for r in rows:
        mp = hw_power if hw_power > 0 else (r["max_power"] if r["max_power"] > 0 else 1000)
        r["power_pct"] = r["power"] / mp * 100
        r["temp_pct"] = r["hotspot_temp"] / temp_ref * 100
        r["gfx_clk_pct"] = r["gfx_clk"] / max_gfx_clk * 100
        r["mem_clk_pct"] = r["mem_clk"] / max_mem_clk * 100
    return rows


def _downsample(points: list, max_pts: int = 800) -> list:
    if len(points) <= max_pts:
        return points
    if max_pts <= 1:
        return [points[-1]] if points else []
    step = (len(points) - 1) / (max_pts - 1)
    indices = [int(i * step) for i in range(max_pts)]
    indices[-1] = len(points) - 1
    return [points[idx] for idx in indices]


def _stats(vals: list[float]) -> Optional[dict[str, float]]:
    if not vals:
        return None
    return {"min": min(vals), "max": max(vals), "avg": round(sum(vals) / len(vals), 1), "samples": len(vals)}


# ---------------------------------------------------------------------------
# SVG + interactive tooltip rendering
# ---------------------------------------------------------------------------


def _svg_polyline(
    pts: list[tuple[float, float]],
    color: str,
    w: int,
    h: int,
    pad: tuple[int, int, int, int],
    y_min: float,
    y_max: float,
) -> str:
    if len(pts) < 2:
        return ""
    pl, pr, _pt, _pb = pad
    t0, t1 = pts[0][0], pts[-1][0]
    span = max(1.0, t1 - t0)
    iw = max(1, w - pl - pr)
    ih = max(1, h - _pt - _pb)

    def _x(t):
        return pl + (t - t0) / span * iw

    def _y(v):
        rng = max(0.01, y_max - y_min)
        u = min(1.0, max(0.0, (v - y_min) / rng))
        return _pt + (1.0 - u) * ih

    coords = " ".join(f"{_x(t):.1f},{_y(v):.1f}" for t, v in pts)
    return f'  <polyline points="{coords}" fill="none" stroke="{color}" stroke-width="1.5"/>'


def _time_axis_ticks(t0: int, t1: int, w: int, h: int, pad: tuple[int, int, int, int]) -> str:
    pl, pr, _pt, _pb = pad
    span = max(1, t1 - t0)
    iw = w - pl - pr
    n_ticks = min(6, span // 60 + 1)
    ticks = ""
    for i in range(n_ticks + 1):
        frac = i / max(1, n_ticks)
        x = pl + frac * iw
        ts_val = t0 + frac * span
        lbl = datetime.fromtimestamp(ts_val).strftime("%H:%M:%S")
        ticks += (
            f'  <text x="{x}" y="{h - 5}" text-anchor="middle" font-size="10" fill="#94a3b8">{lbl} {SYS_TZ_NAME}</text>'
        )
    return ticks


def _y_axis_grid(
    w: int, h: int, pad: tuple[int, int, int, int], y_min: float, y_max: float, n_ticks: int, unit: str
) -> str:
    pl, pr, _pt, _pb = pad
    ih = h - _pt - _pb
    grid = ""
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        val = y_min + frac * (y_max - y_min)
        y = _pt + (1.0 - frac) * ih
        grid += f'  <line x1="{pl}" y1="{y}" x2="{w - pr}" y2="{y}" stroke="#334155" stroke-width="0.5"/>'
        label = f"{val:.0f}{unit}"
        grid += f'  <text x="{pl - 5}" y="{y + 4}" text-anchor="end" font-size="10" fill="#94a3b8">{label}</text>'
    return grid


def _render_combined_chart(  # noqa: C901
    data: dict[str, list[dict[str, Any]]], title: str, throttle_temp_c: float = 100.0, gpu_limits: Optional[dict] = None
) -> str:
    all_pts: list[dict[str, Any]] = []
    for gpu_rows in data.values():
        all_pts.extend(gpu_rows)
    if len(all_pts) < 2:
        return ""

    by_ts: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    metric_keys = [k for k, _, _ in COMBINED_METRICS]
    for r in all_pts:
        t = r["t"]
        for k in metric_keys:
            by_ts[t][k].append(r.get(k, 0))
        by_ts[t]["power_raw"].append(r["power"])
        by_ts[t]["temp_raw"].append(r["hotspot_temp"])
        by_ts[t]["gfx_clk_raw"].append(r["gfx_clk"])
        by_ts[t]["mem_clk_raw"].append(r["mem_clk"])
        by_ts[t]["vram_used_raw"].append(r["vram_used"])

    avg_pts: list[dict[str, Any]] = []
    for t in sorted(by_ts):
        d: dict[str, Any] = {"t": t}
        for k in metric_keys:
            vals = by_ts[t][k]
            d[k] = round(sum(vals) / len(vals), 1) if vals else 0
        for raw_k in ("power_raw", "temp_raw", "gfx_clk_raw", "mem_clk_raw", "vram_used_raw"):
            vals = by_ts[t].get(raw_k, [])
            d[raw_k] = round(sum(vals) / len(vals), 1) if vals else 0
        avg_pts.append(d)

    avg_pts = _downsample(avg_pts, 600)

    metric_keys = [k for k, _, _ in COMBINED_METRICS]
    y_max_pct = max((p.get(k, 0) for p in avg_pts for k in metric_keys), default=100)
    y_max_pct = max(100.0, math.ceil(y_max_pct / 10) * 10)

    w, h = 1100, 360
    pad = (55, 20, 30, 50)
    pl, pr, pt_pad, pb = pad
    ih = h - pt_pad - pb
    n_gridlines = 4 if y_max_pct <= 100 else 5
    grid = ""
    for i in range(n_gridlines + 1):
        pct = y_max_pct * i / n_gridlines
        y = pt_pad + (1.0 - pct / y_max_pct) * ih
        grid += f'  <line x1="{pl}" y1="{y}" x2="{w - pr}" y2="{y}" stroke="#334155" stroke-width="0.5"/>'
        grid += f'  <text x="{pl - 5}" y="{y + 4}" text-anchor="end" font-size="10" fill="#94a3b8">{pct:.0f}%</text>'
    if y_max_pct > 100:
        y100 = pt_pad + (1.0 - 100.0 / y_max_pct) * ih
        grid += f'  <line x1="{pl}" y1="{y100}" x2="{w - pr}" y2="{y100}" stroke="#ef4444" stroke-width="0.5" stroke-dasharray="4,4"/>'

    t0, t1 = avg_pts[0]["t"], avg_pts[-1]["t"]
    grid += _time_axis_ticks(t0, t1, w, h, pad)

    lines = ""
    for metric_key, _, color in COMBINED_METRICS:
        pts = [(r["t"], r.get(metric_key, 0)) for r in avg_pts]
        lines += _svg_polyline(pts, color, w, h, pad, 0.0, y_max_pct)

    all_flat = [r for rows in data.values() for r in rows]
    gl = gpu_limits or {}
    hw_power = gl.get("max_power_w", 0)
    max_power_cap = hw_power if hw_power > 0 else max((r["max_power"] for r in all_flat), default=1000)
    hw_gfx = gl.get("max_gfx_clk_mhz", 0)
    hw_mem = gl.get("max_mem_clk_mhz", 0)
    max_gfx_clk = hw_gfx if hw_gfx > 0 else (max((r["gfx_clk"] for r in all_flat), default=1))
    max_mem_clk = hw_mem if hw_mem > 0 else (max((r["mem_clk"] for r in all_flat), default=1))
    max_vram = max((r.get("vram_total", 0) for r in all_flat), default=0)

    ref_100 = {
        "power_pct": f"{max_power_cap:.0f}W",
        "temp_pct": f"{throttle_temp_c:.0f}\u00b0C",
        "gfx_clk_pct": f"{max_gfx_clk:.0f}MHz",
        "gfx_util": "100%",
        "vram_pct": f"{max_vram:.0f}MB",
        "mem_engine": "100%",
        "mem_clk_pct": f"{max_mem_clk:.0f}MHz",
    }

    legend = ""
    lx = pl
    for metric_key, lbl, color in COMBINED_METRICS:
        ref = ref_100.get(metric_key, "")
        display = f"{lbl} (100%={ref})" if ref else lbl
        legend += (
            f'  <rect x="{lx}" y="{h - 18}" width="8" height="8" fill="{color}"/>'
            + f'  <text x="{lx + 10}" y="{h - 10}" font-size="9" fill="#94a3b8">{display}</text>'
        )
        lx += len(display) * 5 + 16

    cid = _next_chart_id()

    tooltip_fields: list[tuple] = [
        ("power_pct", "Power", "%"),
        ("power_raw", "  \u21b3 actual", "W"),
        ("temp_pct", "Temperature", "%"),
        ("temp_raw", "  \u21b3 actual", "\u00b0C"),
        ("gfx_clk_pct", "GFX Clock", "%"),
        ("gfx_clk_raw", "  \u21b3 actual", " MHz"),
        ("gfx_util", "GFX Util", "%"),
        ("vram_pct", "VRAM Util", "%"),
        ("vram_used_raw", "  \u21b3 used", " MB"),
        ("mem_engine", "MEM Engine", "%"),
        ("mem_clk_pct", "MEM Clock", "%"),
        ("mem_clk_raw", "  \u21b3 actual", " MHz"),
    ]
    for i, (k, lbl, u) in enumerate(tooltip_fields):
        for mk, _, mc in COMBINED_METRICS:
            if k == mk or k.replace("_raw", "_pct") == mk:
                tooltip_fields[i] = (k, lbl, u, mc)
                break
        else:
            tooltip_fields[i] = (k, lbl, u, "#94a3b8")
    fields_for_js = []
    for k, lbl, u, *rest in tooltip_fields:
        c = rest[0] if rest else "#94a3b8"
        fields_for_js.append({"key": k, "label": lbl, "unit": u, "color": c})

    data_json = json.dumps(
        [
            {
                k: round(p.get(k, 0), 1) if isinstance(p.get(k, 0), float) else p.get(k, 0)
                for k in ["t"] + [f["key"] for f in fields_for_js]
            }
            for p in avg_pts
        ]
    )
    fields_json = json.dumps(fields_for_js)

    script = f"""
    <script>
(function() {{
    const cid = "{cid}";
    const data = {data_json};
    const fields = {fields_json};
    const svg = document.getElementById(cid + "_svg");
    const xline = document.getElementById(cid + "_xline");
    const tip = document.getElementById(cid + "_tip");
    if (!svg || !data.length) return;
    const pl={pl}, pr={pr}, pt={pt_pad}, pb={pb}, W={w}, H={h};
    const iw = W - pl - pr, t0 = data[0].t, t1 = data[data.length-1].t, span = Math.max(1, t1 - t0);
    function bisect(ts) {{ let lo=0,hi=data.length-1; while(lo<hi){{ const m=(lo+hi)>>1; if(data[m].t<ts)lo=m+1;else hi=m; }} if(lo>0&&Math.abs(data[lo-1].t-ts)<Math.abs(data[lo].t-ts))lo--; return lo; }}
    svg.addEventListener("mousemove", function(e) {{
        const rect=svg.getBoundingClientRect(), scaleX=W/rect.width, mx=(e.clientX-rect.left)*scaleX;
        if(mx<pl||mx>W-pr){{ xline.style.display="none";tip.style.display="none";return; }}
        const frac=(mx-pl)/iw, ts=t0+frac*span, idx=bisect(ts), d=data[idx], x=pl+(d.t-t0)/span*iw;
        xline.setAttribute("x1",x);xline.setAttribute("x2",x);xline.style.display="";
        const date=new Date((d.t+{SYS_TZ_OFFSET})*1000), timeStr=date.toISOString().substr(11,8)+' {SYS_TZ_NAME}';
        let html='<b>'+timeStr+'</b><br>';
        fields.forEach(f=>{{ const v=d[f.key]; if(v!==undefined&&v!==null) html+='<span style="color:'+f.color+'">\\u25cf</span> '+f.label+': '+(typeof v==="number"?v.toFixed(1):v)+f.unit+'<br>'; }});
        tip.innerHTML=html;tip.style.display="block";
        const tipX=(e.clientX-rect.left)+16, tipY=(e.clientY-rect.top)-10;
        tip.style.left=(tipX+tip.offsetWidth>rect.width?tipX-tip.offsetWidth-32:tipX)+"px";
        tip.style.top=tipY+"px";
    }});
    svg.addEventListener("mouseleave", function(){{ xline.style.display="none";tip.style.display="none"; }});
}})();
    </script>"""

    return f"""
    <div class="chart-card">
    <div class="chart-title">{escape(title)}</div>
    <div style="position:relative">
    <svg id="{cid}_svg" viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">
    {grid}
    {lines}
    {legend}
    </svg>
    <line id="{cid}_xline" x1="0" y1="0" x2="0" y2="{h}" stroke="#fff" stroke-width="0.5" style="display:none"/>
    <div id="{cid}_tip" class="chart-tooltip"></div>
    </div></div>
    {script}"""


def _render_metric_chart(
    data: dict[str, list[dict[str, Any]]], key: str, title: str, unit: str, y_max_override: float = 0
) -> str:
    all_vals = [r[key] for rows in data.values() for r in rows]
    if not all_vals:
        return ""

    y_min = 0.0
    y_max = y_max_override if y_max_override > 0 else (max(all_vals) * 1.1 or 1)

    w, h = 1100, 260
    pad = (55, 20, 25, 50)
    pl, pr, pt_pad, pb = pad

    grid = _y_axis_grid(w, h, pad, y_min, y_max, 5, unit)
    all_ts = sorted({r["t"] for rows in data.values() for r in rows})
    if len(all_ts) >= 2:
        grid += _time_axis_ticks(all_ts[0], all_ts[-1], w, h, pad)

    lines = ""
    legend = ""
    lx = pl + 10
    gpu_ids = sorted(data.keys(), key=lambda g: int(g) if g.isdigit() else 0)
    for gpu_id in gpu_ids:
        pts_raw = _downsample(data[gpu_id], 600)
        pts = [(r["t"], r[key]) for r in pts_raw]
        ci = int(gpu_id) % len(GPU_COLORS) if gpu_id.isdigit() else 0
        color = GPU_COLORS[ci]
        lines += _svg_polyline(pts, color, w, h, pad, y_min, y_max)
        legend += (
            f'  <rect x="{lx}" y="{h - 18}" width="8" height="8" fill="{color}"/>'
            + f'  <text x="{lx + 10}" y="{h - 10}" font-size="9" fill="#94a3b8">GPU {gpu_id}</text>'
        )
        lx += 62

    cid = _next_chart_id()

    tooltip_data: dict[int, dict] = defaultdict(dict)
    for gpu_id in gpu_ids:
        for r in _downsample(data[gpu_id], 600):
            ts = r["t"]
            if ts not in tooltip_data:
                tooltip_data[ts] = {"t": ts}
            tooltip_data[ts][f"g{gpu_id}"] = round(r[key], 1)
    tip_pts = [tooltip_data[t] for t in sorted(tooltip_data)]
    tip_pts = _downsample(tip_pts, 600)

    tip_fields = []
    for gpu_id in gpu_ids:
        ci = int(gpu_id) % len(GPU_COLORS) if gpu_id.isdigit() else 0
        tip_fields.append({"key": f"g{gpu_id}", "label": f"GPU {gpu_id}", "unit": unit, "color": GPU_COLORS[ci]})

    data_json = json.dumps(tip_pts)
    fields_json = json.dumps(tip_fields)

    script = f"""
    <script>
(function() {{
    const cid = "{cid}";
    const data = {data_json};
    const fields = {fields_json};
    const svg = document.getElementById(cid + "_svg");
    const xline = document.getElementById(cid + "_xline");
    const tip = document.getElementById(cid + "_tip");
    if (!svg || !data.length) return;
    const pl={pl}, pr={pr}, pt={pt_pad}, pb={pb}, W={w}, H={h};
    const iw = W - pl - pr, t0 = data[0].t, t1 = data[data.length-1].t, span = Math.max(1, t1 - t0);
    function bisect(ts) {{ let lo=0,hi=data.length-1; while(lo<hi){{ const m=(lo+hi)>>1; if(data[m].t<ts)lo=m+1;else hi=m; }} if(lo>0&&Math.abs(data[lo-1].t-ts)<Math.abs(data[lo].t-ts))lo--; return lo; }}
    svg.addEventListener("mousemove", function(e) {{
        const rect=svg.getBoundingClientRect(), scaleX=W/rect.width, mx=(e.clientX-rect.left)*scaleX;
        if(mx<pl||mx>W-pr){{ xline.style.display="none";tip.style.display="none";return; }}
        const frac=(mx-pl)/iw, ts=t0+frac*span, idx=bisect(ts), d=data[idx], x=pl+(d.t-t0)/span*iw;
        xline.setAttribute("x1",x);xline.setAttribute("x2",x);xline.style.display="";
        const date=new Date((d.t+{SYS_TZ_OFFSET})*1000), timeStr=date.toISOString().substr(11,8)+' {SYS_TZ_NAME}';
        let html='<b>'+timeStr+'</b><br>';
        fields.forEach(f=>{{ const v=d[f.key]; if(v!==undefined&&v!==null) html+='<span style="color:'+f.color+'">\\u25cf</span> '+f.label+': '+(typeof v==="number"?v.toFixed(1):v)+f.unit+'<br>'; }});
        tip.innerHTML=html;tip.style.display="block";
        const tipX=(e.clientX-rect.left)+16, tipY=(e.clientY-rect.top)-10;
        tip.style.left=(tipX+tip.offsetWidth>rect.width?tipX-tip.offsetWidth-32:tipX)+"px";
        tip.style.top=tipY+"px";
    }});
    svg.addEventListener("mouseleave", function(){{ xline.style.display="none";tip.style.display="none"; }});
}})();
    </script>"""

    return f"""
    <div class="chart-card">
    <div class="chart-title">{escape(title)}</div>
    <div style="position:relative">
    <svg id="{cid}_svg" viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">
    {grid}
    {lines}
    {legend}
    </svg>
    <line id="{cid}_xline" x1="0" y1="0" x2="0" y2="{h}" stroke="#fff" stroke-width="0.5" style="display:none"/>
    <div id="{cid}_tip" class="chart-tooltip"></div>
    </div></div>
    {script}"""


def _render_gpu_bar_chart(data: dict[str, list[dict[str, Any]]], metrics: list[tuple[str, str, str]]) -> str:
    gpu_ids = sorted(data.keys(), key=lambda g: int(g) if g.isdigit() else 0)
    if not gpu_ids:
        return ""
    avgs: dict[str, dict[str, float]] = {}
    for gpu_id in gpu_ids:
        avgs[gpu_id] = {}
        for key, _, _ in metrics:
            vals = [r[key] for r in data[gpu_id] if r.get(key, 0) > 0]
            avgs[gpu_id][key] = sum(vals) / len(vals) if vals else 0
    global_max = max((v for g in avgs.values() for v in g.values()), default=1) or 1

    w, h = 1100, 220
    pad_l, pad_b, pad_t, pad_r = 55, 40, 30, 20
    bar_area_w = w - pad_l - pad_r
    bar_area_h = h - pad_t - pad_b
    group_w = bar_area_w / max(1, len(gpu_ids))
    bar_w = group_w / (len(metrics) + 1)

    bars = ""
    for gi, gpu_id in enumerate(gpu_ids):
        gx = pad_l + gi * group_w
        bars += f'  <text x="{gx + group_w / 2}" y="{h - 10}" text-anchor="middle" font-size="11" fill="#94a3b8">GPU {gpu_id}</text>'
        for bi, (key, _label, color) in enumerate(metrics):
            val = avgs[gpu_id][key]
            bh = (val / global_max) * bar_area_h if global_max > 0 else 0
            bx = gx + (bi + 0.5) * bar_w
            by = pad_t + bar_area_h - bh
            bars += f'  <rect x="{bx}" y="{by}" width="{bar_w * 0.8}" height="{bh}" fill="{color}" rx="2"/>'
            if val > 0:
                bars += f'  <text x="{bx + bar_w * 0.4}" y="{by - 4}" text-anchor="middle" font-size="9" fill="#e2e8f0">{val:.0f}</text>'
    legend = ""
    lx = pad_l + 10
    for _, lbl, color in metrics:
        legend += (
            f'  <rect x="{lx}" y="{h - 18}" width="8" height="8" fill="{color}"/>'
            + f'  <text x="{lx + 10}" y="{h - 10}" font-size="9" fill="#94a3b8">{lbl}</text>'
        )
        lx += len(lbl) * 7 + 28

    return f"""
    <div class="chart-card">
    <div class="chart-title">Per-GPU Averages</div>
    <div style="position:relative">
    <svg viewBox="0 0 {w} {h}" width="100%" preserveAspectRatio="xMidYMid meet">
    {bars}
    {legend}
    </svg>
    </div></div>"""


# ---------------------------------------------------------------------------
# Health checks + per-GPU table from enriched summary.json
# ---------------------------------------------------------------------------


def _load_analysis(run_dir: str) -> dict[str, Any]:
    path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _render_health_checks(analysis: dict[str, Any]) -> str:
    anomalies = analysis.get("anomalies", {})
    power = analysis.get("power_analysis", {})
    mem = analysis.get("mem_temp_analysis", {})
    mon = analysis.get("monitoring_validation", {})
    pattern = analysis.get("workload_pattern", "")
    per_gpu = analysis.get("per_gpu", {})
    if not anomalies and not power and not per_gpu:
        return ""

    def _badge(ok: bool, ok_text: str, warn_text: str) -> str:
        if ok:
            return f'<span class="result-badge pass">{escape(ok_text)}</span>'
        return f'<span class="result-badge fail">{escape(warn_text)}</span>'

    rows = ""

    thr = anomalies.get("throttle_events", [])
    if thr:
        worst = max(e.get("throttle_pct", 0) for e in thr)
        rows += (
            f"<tr><td>Throttling</td><td>{_badge(False, '', f'WARNING ({len(thr)} GPU(s), up to {worst}%)')}</td></tr>"
        )
    else:
        rows += f"<tr><td>Throttling</td><td>{_badge(True, 'OK', '')}</td></tr>"

    imb = anomalies.get("imbalanced_gpus", [])
    if imb:
        mx = max(i.get("deviation_pct", 0) for i in imb)
        rows += f"<tr><td>GPU Imbalance</td><td>{_badge(False, '', f'WARNING (max deviation {mx}%)')}</td></tr>"
    elif pattern == "serial":
        rows += "<tr><td>GPU Imbalance</td><td>N/A (serial workload)</td></tr>"
    else:
        rows += f"<tr><td>GPU Imbalance</td><td>{_badge(True, 'OK', '')}</td></tr>"

    if pattern:
        rows += f"<tr><td>Workload</td><td>{escape(pattern)} ({len(per_gpu)} GPU(s))</td></tr>"

    ms = mem.get("mem_temp")
    if ms:
        exceeds = mem.get("mem_temp_exceeds_threshold", False)
        ds = mem.get("hotspot_mem_delta")
        d_avg = ds.get("avg", 0) if ds else 0
        max_mt = ms["max"]
        ok_txt = f"OK (max {max_mt}C, delta {d_avg}C)"
        warn_txt = f"WARNING (max {max_mt}C)"
        rows += f"<tr><td>Mem Temp</td><td>{_badge(not exceeds, ok_txt, warn_txt)}</td></tr>"

    rising = mem.get("temp_still_rising", False)
    rows += f"<tr><td>Temp Trend</td><td>{_badge(not rising, 'STABLE', 'WARNING (still rising)')}</td></tr>"

    cu = power.get("cap_utilization_pct", 0)
    cs = power.get("cap_saturation_pct", 0)
    cw = power.get("avg_power_cap_w", 0)
    if cu > 0:
        rows += f"<tr><td>Power Cap</td><td>{cu}% avg, {cs}% at cap ({cw:.0f}W)</td></tr>"

    mvs = mon.get("status", "SKIPPED")
    mvw = mon.get("warnings", [])
    if mvw:
        detail = "; ".join(mvw[:3])
        rows += f"<tr><td>Monitor Valid.</td><td>{_badge(False, '', f'{mvs}: {escape(detail)}')}</td></tr>"
    else:
        rows += f"<tr><td>Monitor Valid.</td><td>{_badge(mvs == 'OK', mvs, mvs)}</td></tr>"

    return f"""
    <div class="section-title">Health Checks</div>
    <table>
    <tr><th>Check</th><th>Status</th></tr>
    {rows}
    </table>"""


def _render_per_gpu_table(analysis: dict[str, Any]) -> str:
    per_gpu = analysis.get("per_gpu", {})
    if not per_gpu:
        return ""

    rows = ""
    for gid in sorted(per_gpu, key=lambda g: int(g) if g.isdigit() else 0):
        st = per_gpu[gid]

        def _v(key: str, field: str = "avg") -> str:
            s = st.get(key)
            if not s:
                return "N/A"
            return f"{s[field]:.0f}" if field != "avg" else f"{s['avg']:.1f}"

        rows += (
            f"<tr><td>GPU {gid}</td>"
            + f"<td>{_v('power')}</td>"
            + f"<td>{_v('hotspot_temp')}</td>"
            + f"<td>{_v('gfx_clk')}</td>"
            + f"<td>{_v('gfx_util')}</td>"
            + f"<td>{_v('vram_pct')}</td>"
            + f"<td>{st.get('active_samples', 'N/A')}</td>"
            + f"</tr>"
        )

    pattern = analysis.get("workload_pattern", "")
    note = ""
    if pattern == "serial":
        note = '<div class="hint">Serial workload detected \u2014 stats show active-window averages per GPU.</div>'

    return f"""
    <div class="section-title">Per-GPU Statistics (steady-state)</div>
    <table>
    <tr><th>GPU</th><th>Power (W)</th><th>Temp (\u00b0C)</th><th>GFX Clk (MHz)</th><th>GFX Util (%)</th><th>VRAM (%)</th><th>Active Samples</th></tr>
    {rows}
    </table>
    {note}"""


# ---------------------------------------------------------------------------
# Main HTML assembly
# ---------------------------------------------------------------------------

CSS = """
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
    .header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #334155; border-radius: 8px; padding: 24px; margin-bottom: 20px; }
    .header h1 { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
    .result-badge { display: inline-block; padding: 3px 12px; border-radius: 4px; font-weight: 600; font-size: 13px; }
    .result-badge.pass { background: #166534; color: #4ade80; }
    .result-badge.fail { background: #7f1d1d; color: #f87171; }
    .result-badge.skip { background: #374151; color: #9ca3af; }
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 8px; margin-top: 12px; font-size: 13px; color: #94a3b8; }
    .meta-grid span { color: #e2e8f0; }
    .chart-card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 16px; margin-bottom: 16px; position: relative; }
    .chart-title { font-size: 14px; font-weight: 600; margin-bottom: 8px; color: #f1f5f9; }
    .chart-tooltip { display: none; position: absolute; background: rgba(15,23,42,0.95); border: 1px solid #475569; border-radius: 6px; padding: 8px 12px; font-size: 12px; line-height: 1.5; pointer-events: none; z-index: 100; white-space: nowrap; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }
    .section-title { font-size: 16px; font-weight: 600; margin: 24px 0 12px; color: #f1f5f9; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 16px; }
    th { background: #1e293b; padding: 8px 12px; text-align: left; border-bottom: 2px solid #334155; color: #94a3b8; font-weight: 600; }
    td { padding: 6px 12px; border-bottom: 1px solid #1e293b; }
    tr:hover td { background: rgba(51, 65, 85, 0.3); }
    .hint { font-size: 12px; color: #64748b; margin-top: 4px; }
    svg { display: block; }
"""


def generate_report(run_dir: str, test_name: str, result: str, duration: int, output: str) -> None:
    global _chart_id_counter  # noqa: PLW0603
    _chart_id_counter = 0

    meta = _load_command_metadata(run_dir)
    raw_rows = _load_csv(os.path.join(run_dir, "power_temp.csv"))
    cu_rows = _load_csv(os.path.join(run_dir, "cu_occupancy.csv"))

    parsed = _parse_rows(raw_rows)

    analysis = _load_analysis(run_dir)
    gpu_limits = analysis.get("gpu_limits", {})
    throttle_temp = gpu_limits.get("throttle_temp_c", 100.0)

    parsed = _compute_pct_fields(parsed, gpu_limits=gpu_limits)

    by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in parsed:
        by_gpu[r["gpu"]].append(r)

    charts_html = ""
    charts_html += _render_combined_chart(
        by_gpu, "All Metrics (normalized to %)", throttle_temp_c=throttle_temp, gpu_limits=gpu_limits
    )
    charts_html += _render_metric_chart(by_gpu, "power", "Power (W)", "W")
    charts_html += _render_metric_chart(by_gpu, "hotspot_temp", "Hotspot Temperature (\u00b0C)", "\u00b0C")
    charts_html += _render_metric_chart(by_gpu, "gfx_clk", "GFX Clock (MHz)", " MHz")
    charts_html += _render_metric_chart(by_gpu, "gfx_util", "GFX Utilization (%)", "%", y_max_override=100.0)
    charts_html += _render_metric_chart(by_gpu, "vram_pct", "VRAM Utilization (%)", "%", y_max_override=100.0)
    charts_html += _render_metric_chart(by_gpu, "mem_engine", "MEM Engine Activity (%)", "%", y_max_override=100.0)
    charts_html += _render_metric_chart(by_gpu, "mem_clk", "Memory Clock (MHz)", " MHz")
    charts_html += _render_gpu_bar_chart(
        by_gpu,
        [
            ("power", "Avg Power (W)", "#ef4444"),
            ("hotspot_temp", "Avg Temp (\u00b0C)", "#f59e0b"),
            ("gfx_clk", "Avg GFX Clk (MHz)", "#3b82f6"),
        ],
    )

    all_vals = {
        "Power (W)": [r["power"] for r in parsed],
        "Hotspot Temp (\u00b0C)": [r["hotspot_temp"] for r in parsed],
        "Memory Temp (\u00b0C)": [r["mem_temp"] for r in parsed],
        "GFX Clock (MHz)": [r["gfx_clk"] for r in parsed],
        "GFX Util (%)": [r["gfx_util"] for r in parsed],
        "MEM Engine (%)": [r["mem_engine"] for r in parsed],
        "MEM Clock (MHz)": [r["mem_clk"] for r in parsed],
        "VRAM Used (MB)": [r["vram_used"] for r in parsed],
        "VRAM Util (%)": [r["vram_pct"] for r in parsed],
    }
    stats_rows = ""
    for lbl, vals in all_vals.items():
        s = _stats(vals)
        if s:
            stats_rows += (
                f"<tr><td>{escape(lbl)}</td><td>{s['min']:.0f}</td><td>{s['max']:.0f}</td>"
                + f"<td>{s['avg']:.1f}</td><td>{s['samples']}</td></tr>"
            )

    cu_table = ""
    if cu_rows:
        cu_data = [r for r in cu_rows if _sf(r.get("cu_occupancy", 0)) > 0]
        if cu_data:
            cu_table = (
                '<div class="section-title">CU Occupancy Samples</div>'
                "<table><tr><th>Time</th><th>GPU</th><th>PID</th><th>CU Occ</th><th>VRAM (MB)</th></tr>"
            )
            for r in cu_data[:50]:
                ts = datetime.fromtimestamp(int(r["timestamp"])).strftime(f"%H:%M:%S {SYS_TZ_NAME}")
                cu_table += (
                    f"<tr><td>{ts}</td><td>{r['gpu']}</td><td>{r['pid']}</td>"
                    + f"<td>{r['cu_occupancy']}</td><td>{r['vram_mb']}</td></tr>"
                )
            cu_table += "</table>"
            if len(cu_data) > 50:
                cu_table += f'<div class="hint">Showing 50 of {len(cu_data)} samples with CU_OCCUPANCY > 0</div>'

    health_html = _render_health_checks(analysis)
    per_gpu_html = _render_per_gpu_table(analysis)

    result_upper = result.upper()
    if "UNSUPPORTED" in result_upper or "SKIP" in result_upper:
        result_class = "skip"
    elif "PASS" in result_upper:
        result_class = "pass"
    else:
        result_class = "fail"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{escape(test_name)} \u2014 Monitoring Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="header">
<h1>{escape(test_name)} <span class="result-badge {result_class}">{escape(result)}</span></h1>
<div class="meta-grid">
<div>Duration: <span>{duration}s</span></div>
<div>Timestamp: <span>{escape(str(meta.get('timestamp', 'N/A')))}</span></div>
<div>Hostname: <span>{escape(str(meta.get('hostname', 'N/A')))}</span></div>
<div>ROCm: <span>{escape(str(meta.get('rocm_version', 'N/A')))}</span></div>
<div>GPU: <span>{escape(str(meta.get('gpu_model', 'N/A')))}</span></div>
<div>GPUs: <span>{escape(str(meta.get('num_gpus', 'N/A')))}</span></div>
<div>Samples: <span>{len(parsed)}</span></div>
<div>Interval: <span>{escape(str(meta.get('sample_interval', '1s')))}</span></div>
</div>
<div class="hint">
Workload: {escape(meta.get('workload_function', ''))} \u2014 {escape(meta.get('goal', ''))}
</div>
</div>

{charts_html}

<div class="section-title">Summary Statistics (all GPUs)</div>
<table>
<tr><th>Metric</th><th>Min</th><th>Max</th><th>Avg</th><th>Samples</th></tr>
{stats_rows}
</table>

{health_html}

{per_gpu_html}

{cu_table}

<div class="hint" style="margin-top:24px">
Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} \u2014 Hover over charts to see values
</div>
</body>
</html>"""

    with open(output, "w") as f:
        f.write(html)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--test-name", required=True)
    ap.add_argument("--result", default="UNKNOWN")
    ap.add_argument("--duration", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    generate_report(args.run_dir, args.test_name, args.result, args.duration, args.output)
