#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
Analyze results from rock_mps_test runs.

Reads health CSVs, monitor CSVs, profiler CSVs, stdout logs, and sysinfo.txt
from the results directory and produces a colored summary report with
throughput stats and an executive summary.

Supports both single-GPU directories (e.g., /tmp/.../gpu0) and multi-GPU
parent directories containing gpu0/, gpu1/, etc. subdirectories.

Usage:
    python3 scripts/check_results.py <results_dir>
    python3 scripts/check_results.py <results_dir> --json report.json
    python3 scripts/check_results.py <results_dir> --junit results.xml
    python3 scripts/check_results.py <results_dir> --json report.json --junit results.xml
"""

import argparse
import contextlib
import csv
import glob
import json
import os
import re
import sys
from xml.dom import minidom
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# ANSI color helpers — auto-disabled when stdout is not a terminal
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    if _USE_COLOR:
        return f"\033[{code}m{text}\033[0m"
    return text


def green(t):
    return _c("32", t)


def red(t):
    return _c("31", t)


def yellow(t):
    return _c("33", t)


def bold(t):
    return _c("1", t)


def dim(t):
    return _c("2", t)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

_DEFAULT_HEALTH_WARN_MB = 100
_MEMORY_MOVER_HEALTH_WARN_MB = 1024


def read_sysinfo(results_dir):
    for d in [results_dir, os.path.dirname(results_dir)]:
        path = os.path.join(d, "sysinfo.txt")
        if os.path.isfile(path):
            with open(path, errors="replace") as f:
                return f.read().strip()
    return None


def parse_sysinfo_fields(sysinfo_text):
    """Extract key fields from sysinfo.txt for the executive summary."""
    fields = {}
    if not sysinfo_text:
        return fields
    for line in sysinfo_text.splitlines():
        if line.startswith("ROCm:"):
            fields["rocm_version"] = line.split(":", 1)[1].strip()
        elif line.startswith("Kernel:"):
            fields["kernel"] = line.split(":", 1)[1].strip()
        elif line.startswith("Hostname:"):
            fields["hostname"] = line.split(":", 1)[1].strip()
        elif "Card series" in line:
            fields.setdefault("gpu_model", line.split(":", 1)[-1].strip())
        elif line.startswith("Count:"):
            fields["gpu_count"] = line.split(":", 1)[1].strip()
    return fields


def _is_monotonic_growth(samples, key, total_growth_mb, units_per_mb=1024 * 1024):
    """Check if a metric is growing continuously (leak) vs. one-time jump (pool init).

    Splits the samples into first half and second half. If the metric is still
    growing significantly in the second half, it's likely a real leak. If all
    growth happened in the first half and the second half is flat, it's just
    runtime pool initialization.

    units_per_mb: how many raw units make 1 MB (1024*1024 for bytes, 1024 for KB)."""
    if len(samples) < 10:
        return True  # too few samples to distinguish — assume worst case
    mid = len(samples) // 2
    try:
        vals = [int(s.get(key, 0)) for s in samples]
    except (ValueError, TypeError):
        return True
    second_half_growth_mb = (vals[-1] - vals[mid]) / units_per_mb
    # If second half grew by more than 20% of total growth or 50 MB, it's still climbing
    return second_half_growth_mb > max(total_growth_mb * 0.2, 50)


def analyze_health_csv(  # noqa: C901 — vendored analyzer; complexity is intentional
    path, rss_warn_mb=_DEFAULT_HEALTH_WARN_MB, vram_warn_mb=_DEFAULT_HEALTH_WARN_MB
):
    try:
        first = None
        last = None
        all_samples = []
        sticky_errors = 0
        count = 0

        with open(path, errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                count += 1
                all_samples.append(row)
                if first is None:
                    first = row
                last = row
                if row.get("hip_error_sticky") == "1":
                    sticky_errors += 1

        if count < 2 or first is None or last is None:
            return {"status": "SKIP", "reason": "too few samples"}

        def _safe_int(row, key):
            try:
                return int(row.get(key, 0))
            except (ValueError, TypeError):
                return 0

        issues = []

        rss_start = _safe_int(first, "host_rss_kb")
        rss_end = _safe_int(last, "host_rss_kb")
        rss_growth_mb = (rss_end - rss_start) / 1024.0
        if rss_growth_mb > rss_warn_mb:
            if _is_monotonic_growth(all_samples, "host_rss_kb", rss_growth_mb, units_per_mb=1024):
                issues.append(
                    f"Host RSS grew by {rss_growth_mb:.0f} MB — continuous growth detected (likely memory leak)"
                )
            else:
                issues.append(
                    f"Host RSS grew by {rss_growth_mb:.0f} MB"
                    " — early jump then stable (likely runtime init, not a leak)"
                )

        fd_start = _safe_int(first, "fd_count")
        fd_end = _safe_int(last, "fd_count")
        fd_growth = fd_end - fd_start
        if fd_growth > 20:
            issues.append(f"FD count grew by {fd_growth} (possible FD leak)")

        vram_start = _safe_int(first, "vram_used_bytes")
        vram_end = _safe_int(last, "vram_used_bytes")
        vram_growth_mb = (vram_end - vram_start) / (1024 * 1024)
        if vram_growth_mb > vram_warn_mb:
            if _is_monotonic_growth(all_samples, "vram_used_bytes", vram_growth_mb):
                issues.append(f"VRAM grew by {vram_growth_mb:.0f} MB — continuous growth detected (likely VRAM leak)")
            else:
                issues.append(
                    f"VRAM grew by {vram_growth_mb:.0f} MB"
                    " — early jump then stable (likely runtime pool init, not a leak)"
                )

        if sticky_errors > 0:
            issues.append(f"Sticky HIP errors detected in {sticky_errors} samples")

        # Separate real concerns from informational notes
        real_issues = [i for i in issues if "not a leak" not in i]
        info_notes = [i for i in issues if "not a leak" in i]

        if real_issues:
            return {"status": "WARN", "issues": issues}
        if info_notes:
            return {"status": "INFO", "issues": info_notes}
        try:
            dur = float(last.get("timestamp_sec", 0))
        except (ValueError, TypeError):
            dur = 0.0
        return {"status": "OK", "duration_sec": dur}
    except OSError as e:
        return {"status": "WARN", "issues": [f"Failed to read health CSV: {e}"]}


def analyze_monitor_csv(path):
    slow_queries = 0
    total = 0
    max_query_us = 0

    try:
        f = open(path, errors="replace")  # noqa: SIM115 — opened then immediately used as context manager below
    except OSError as e:
        return {"total_queries": 0, "slow_queries": 0, "max_query_us": 0, "error": str(e)}

    with f:
        header = f.readline()
        if not header:
            return {"total_queries": 0, "slow_queries": 0, "max_query_us": 0}
        cols = header.strip().split(",")
        try:
            us_idx = cols.index("query_us")
        except ValueError:
            return {"total_queries": 0, "slow_queries": 0, "max_query_us": 0}

        for line in f:
            fields = line.split(",")
            if len(fields) <= us_idx:
                continue
            total += 1
            try:
                query_us = int(fields[us_idx])
            except ValueError:
                continue
            if query_us > max_query_us:
                max_query_us = query_us
            if query_us > 100000:
                slow_queries += 1

    return {
        "total_queries": total,
        "slow_queries": slow_queries,
        "max_query_us": max_query_us,
    }


def analyze_profiler_csv(path):
    negative_times = 0
    total = 0
    ms_idx = -1

    try:
        with open(path, errors="replace") as f:
            header = f.readline()
            if not header:
                return {"total_kernels": 0, "negative_times": 0}
            cols = header.strip().split(",")
            with contextlib.suppress(ValueError):
                ms_idx = cols.index("kernel_ms")

            for line in f:
                total += 1
                if ms_idx >= 0:
                    fields = line.split(",")
                    if len(fields) > ms_idx:
                        try:
                            if float(fields[ms_idx]) <= 0:
                                negative_times += 1
                        except ValueError:
                            pass
    except OSError:
        pass

    return {"total_kernels": total, "negative_times": negative_times}


def _read_tail(path, tail_bytes=64 * 1024):
    """Read the last tail_bytes of a file. The Finished/Exit lines are at the end."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes)
            f.readline()  # skip partial first line
        return f.read().decode("utf-8", errors="replace")


def _read_head(path, head_bytes=16 * 1024):
    """Read the beginning of a log file where role identity and PID are printed."""
    with open(path, "rb") as f:
        return f.read(head_bytes).decode("utf-8", errors="replace")


def _read_stderr(stdout_path):
    """Read companion stderr log if it exists."""
    stderr_path = stdout_path.replace("_stdout.log", "_stderr.log")
    if os.path.isfile(stderr_path) and os.path.getsize(stderr_path) > 0:
        return _read_tail(stderr_path, 32 * 1024)
    return ""


def analyze_stdout_log(path):  # noqa: C901 — vendored analyzer; complexity is intentional
    result = {"role": os.path.basename(path).replace("_stdout.log", "")}

    stderr_content = _read_stderr(path)

    if os.path.getsize(path) == 0:
        result["iterations"] = 0
        result["errors"] = 0
        result["crashed"] = True
        result["termination"] = "CRASH (process produced no output — died during initialization)"
        sigs = (
            re.findall(
                r"(?i)segfault|segmentation fault|SIGSEGV|SIGABRT|SIGKILL|"
                r"Memory access fault|abort|out of memory|cannot allocate|"
                r"hip\w+ failed|amdsmi_\w+ failed|double free|"
                r"error:.*",
                stderr_content,
            )
            if stderr_content
            else []
        )
        result["crash_signatures"] = list(set(sigs))[:10]
        result["stderr_tail"] = (
            stderr_content.strip().splitlines()[-10:]
            if stderr_content.strip()
            else ["(no stderr captured — check dmesg for segfault/GPF)"]
        )
        result["failure_lines"] = []
        return result

    head_content = _read_head(path)
    content = _read_tail(path)
    combined_content = head_content + "\n" + content

    m_pid = re.search(r"\[[A-Z0-9_-]+\]\s+PID\s+(\d+)", combined_content)
    if m_pid:
        result["pid"] = int(m_pid.group(1))

    # All Finished lines now end with (Ns) — capture elapsed time
    # MEMORY_MOVER full: "Finished: N iterations, N checks, N errors, N alloc_failures, N ipc_rounds, N ipc_errors (Ns)"
    m = re.search(
        r"Finished: (\d+) iterations?, (\d+) checks?, (\d+) errors?, "
        r"(\d+) alloc_failures?, (\d+) ipc_rounds?, (\d+) ipc_errors?"
        r"(?: \(([0-9.]+)s\))?",
        content,
    )
    if m:
        result["iterations"] = int(m.group(1))
        result["checks"] = int(m.group(2))
        result["errors"] = int(m.group(3))
        result["alloc_failures"] = int(m.group(4))
        result["ipc_rounds"] = int(m.group(5))
        result["ipc_errors"] = int(m.group(6))
        if m.group(7):
            result["elapsed_sec"] = float(m.group(7))

    # MEMORY_MOVER older: "Finished: N iterations, N checks, N errors, N alloc_failures (Ns)"
    if "iterations" not in result:
        m = re.search(
            r"Finished: (\d+) iterations?, (\d+) checks?, (\d+) errors?, " r"(\d+) alloc_failures?(?: \(([0-9.]+)s\))?",
            content,
        )
        if m:
            result["iterations"] = int(m.group(1))
            result["checks"] = int(m.group(2))
            result["errors"] = int(m.group(3))
            result["alloc_failures"] = int(m.group(4))
            if m.group(5):
                result["elapsed_sec"] = float(m.group(5))

    # COMPUTE: "Finished: N iterations, N checks, N errors (Ns)"
    if "iterations" not in result:
        m = re.search(r"Finished: (\d+) iterations?, (\d+) checks?, (\d+) errors?" r"(?: \(([0-9.]+)s\))?", content)
        if m:
            result["iterations"] = int(m.group(1))
            result["checks"] = int(m.group(2))
            result["errors"] = int(m.group(3))
            if m.group(4):
                result["elapsed_sec"] = float(m.group(4))

    # LIBRARY/COMPILER: "Finished: N iterations, N errors (Ns)"
    if "iterations" not in result:
        m = re.search(r"Finished: (\d+) iterations?, (\d+) errors?" r"(?: \(([0-9.]+)s\))?", content)
        if m:
            result["iterations"] = int(m.group(1))
            result["errors"] = int(m.group(2))
            if m.group(3):
                result["elapsed_sec"] = float(m.group(3))

    # MONITOR: "Finished: N queries, N errors, avg_interval=... (Ns)"
    if "iterations" not in result:
        m = re.search(r"Finished: (\d+) queries?, (\d+) errors?.*?" r"(?: \(([0-9.]+)s\))?$", content, re.MULTILINE)
        if m:
            result["iterations"] = int(m.group(1))
            result["errors"] = int(m.group(2))
            if m.group(3):
                result["elapsed_sec"] = float(m.group(3))

    # PROFILER: "Finished: N iterations, N anomalies (>Rx), N severe (>Rx), N invalid, ...% anomaly rate (Ns)"
    # followed by "Verdict: PASS/FAIL (...)"
    # Anomalies are informational — the verdict determines pass/fail.
    if "iterations" not in result:
        m = re.search(
            r"Finished: (\d+) iterations?, (\d+) (?:timing )?anomal\S*"
            r"[^\n]*?(\d+) severe[^\n]*?(\d+) invalid[^\n]*?"
            r"\(([0-9.]+)s\)\s*$",
            content,
            re.MULTILINE,
        )
        if m:
            result["iterations"] = int(m.group(1))
            result["anomalies"] = int(m.group(2))
            result["severe_anomalies"] = int(m.group(3))
            result["invalid_times"] = int(m.group(4))
            if m.group(5):
                result["elapsed_sec"] = float(m.group(5))
            # Use the Verdict line to determine errors, not raw anomaly count
            v = re.search(r"Verdict:\s*(PASS|FAIL)", content)
            if v and v.group(1) == "FAIL":
                result["errors"] = 1
            else:
                result["errors"] = 0

    # IPC_XFER: "Finished: N iterations, N peer_copy_errors, N peer_direct_errors, N host_verify_errors (Ns)"
    if "iterations" not in result:
        m = re.search(
            r"Finished: (\d+) iterations?, (\d+) peer_copy_errors?, "
            r"(\d+) peer_direct_errors?, (\d+) host_verify_errors?"
            r"(?: \(([0-9.]+)s\))?",
            content,
        )
        if m:
            result["iterations"] = int(m.group(1))
            result["errors"] = int(m.group(2)) + int(m.group(3)) + int(m.group(4))
            result["peer_copy_errors"] = int(m.group(2))
            result["peer_direct_errors"] = int(m.group(3))
            result["host_verify_errors"] = int(m.group(4))
            if m.group(5):
                result["elapsed_sec"] = float(m.group(5))

    m = re.search(r"Exit code: (-?\d+)", content)
    if m:
        result["exit_code"] = int(m.group(1))

    signal_names = {
        -6: "SIGABRT (abort/assertion)",
        -11: "SIGSEGV (segmentation fault)",
        -9: "SIGKILL (killed)",
        -15: "SIGTERM (terminated)",
        -4: "SIGILL (illegal instruction)",
        -8: "SIGFPE (floating-point exception)",
        -7: "SIGBUS (bus error)",
    }

    has_finished = "Finished:" in content
    exit_code = result.get("exit_code")

    if not has_finished and exit_code is None:
        result["crashed"] = True
    elif not has_finished and exit_code is not None:
        result["crashed"] = True
        if exit_code < 0:
            sig = signal_names.get(exit_code, f"signal {-exit_code}")
            result["signal"] = sig
    elif has_finished and exit_code is not None and exit_code < 0:
        sig = signal_names.get(exit_code, f"signal {-exit_code}")
        result["signal"] = sig
        result["crashed"] = True
    else:
        result["crashed"] = False

    if exit_code is not None and exit_code < 0:
        result["signal"] = signal_names.get(exit_code, f"signal {-exit_code}")

    # Search stdout + stderr for crash evidence
    search_both = content + "\n" + stderr_content
    crash_patterns = [
        (r"Memory access fault by GPU node-\d+.*", "GPU memory access fault"),
        (r"(?i)segmentation fault", "Segmentation fault"),
        (r"(?i)SIGSEGV", "SIGSEGV"),
        (r"(?i)SIGABRT", "SIGABRT"),
        (r"(?i)SIGKILL", "SIGKILL"),
        (r"(?i)double free", "Double free"),
        (r"(?i)stack smash", "Stack smashing"),
        (r"(?i)heap.*corrupt", "Heap corruption"),
        (r"(?i)out of memory", "Out of memory"),
        (r"(?i)core dump", "Core dump"),
        (r"LLVM ERROR:.*", "LLVM fatal error"),
        (r"(?i)abort", "Abort"),
    ]
    crash_signatures = []
    for pattern, _label in crash_patterns:
        matches = re.findall(pattern, search_both)
        if matches:
            crash_signatures.append(matches[0].strip()[:120])
    result["crash_signatures"] = crash_signatures[:5]

    # Build a human-readable termination reason
    if result.get("crashed"):
        sig = result.get("signal")
        if crash_signatures:
            reason = crash_signatures[0]
        elif sig:
            reason = sig
        elif exit_code is not None and exit_code != 0:
            reason = f"exit code {exit_code}"
        else:
            reason = "process killed (no Finished line, no exit code — check dmesg)"
        result["termination"] = reason

    if stderr_content.strip() and result.get("crashed"):
        result["stderr_tail"] = stderr_content.strip().splitlines()[-10:]

    m_hip_err = re.search(r"Sticky HIP error detected", content)
    result["sticky_hip_error"] = bool(m_hip_err)

    fail_lines = re.findall(r"\*\*\*.*?\*\*\*", content)
    result["failure_lines"] = fail_lines[:10]

    # Grab last meaningful log line before crash (for context)
    if result.get("crashed"):
        log_lines = content.strip().splitlines()
        last_useful = [ln.strip() for ln in log_lines[-5:] if ln.strip() and not ln.startswith("=")]
        result["last_log_lines"] = last_useful[-3:] if last_useful else []

    return result


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_throughput(result):
    iters = result.get("iterations")
    elapsed = result.get("elapsed_sec")
    if iters is not None and elapsed and elapsed > 0:
        return f"{iters / elapsed:.1f}/s"
    return None


def print_report(top_dir, dirs_to_analyze, all_results):  # noqa: C901 — vendored analyzer; complexity is intentional
    sysinfo = read_sysinfo(top_dir)

    print(bold("=" * 70))
    print(bold("ROCK MPS Test — Results Analysis"))
    print(f"Directory: {top_dir}")
    print(f"GPUs found: {len(dirs_to_analyze)}")
    print(bold("=" * 70))

    if sysinfo:
        print(dim("\n  --- System Information ---"))
        for line in sysinfo.splitlines():
            print(dim(f"  {line}"))

    overall_pass = True
    total_iterations = 0
    total_checks = 0
    total_errors = 0
    total_ipc_rounds = 0
    total_ipc_errors = 0
    total_processes = 0
    total_crashed = 0
    total_alloc_failures = 0
    health_warnings = 0
    timing_anomaly_count = 0
    timing_total_count = 0
    crash_details = []

    for gpu_label, _results_dir in dirs_to_analyze:
        gpu_results = all_results[gpu_label]

        print(f"\n{bold('=' * 70)}")
        print(bold(f"  {gpu_label.upper()}"))
        print(bold("=" * 70))

        # Role results
        print(f"\n  {bold('--- Role Results ---')}")
        for r in gpu_results["roles"]:
            total_processes += 1
            crashed = r.get("crashed", False)
            if crashed:
                total_crashed += 1
                overall_pass = False
                status = "CRASH"
                crash_details.append(
                    {
                        "gpu": gpu_label,
                        "role": r["role"],
                        "cause": r.get("termination", "unknown"),
                    }
                )
            else:
                status = "PASS" if r.get("exit_code", 1) == 0 else "FAIL"
                if status == "FAIL":
                    overall_pass = False

            iters = r.get("iterations", "?")
            errors = r.get("errors", "?")
            checks = r.get("checks")

            if isinstance(iters, int):
                total_iterations += iters
            if isinstance(errors, int) and r.get("anomalies") is None:
                total_errors += errors
            if checks is not None:
                total_checks += checks

            anomalies = r.get("anomalies")
            if anomalies is not None:
                severe = r.get("severe_anomalies", 0)
                invalid = r.get("invalid_times", 0)
                detail = f"{iters} iterations, {anomalies} anomalies, {severe} severe, {invalid} invalid"
            elif checks is not None:
                detail = f"{iters} iterations, {checks} checks, {errors} errors"
            else:
                detail = f"{iters} iterations, {errors} errors"

            tp = format_throughput(r)
            if tp:
                detail += f" ({tp})"

            alloc_fails = r.get("alloc_failures")
            if alloc_fails is not None and alloc_fails > 0:
                total_alloc_failures += alloc_fails
                detail += f", {alloc_fails} alloc failures (handled)"
            ipc_rounds = r.get("ipc_rounds")
            ipc_errs = r.get("ipc_errors")
            if ipc_rounds is not None and ipc_rounds > 0:
                total_ipc_rounds += ipc_rounds
                total_ipc_errors += ipc_errs or 0
                ipc_tp = ""
                elapsed = r.get("elapsed_sec")
                if elapsed and elapsed > 0:
                    ipc_tp = f", {ipc_rounds / elapsed:.1f} IPC/s"
                detail += f", {ipc_rounds} IPC rounds ({ipc_errs} IPC errors{ipc_tp})"

            if status == "CRASH":
                tag = red("[CRASH]")
            elif status == "PASS":
                tag = green("[PASS] ")
            else:
                tag = red("[FAIL] ")
            print(f"    {tag} {r['role']:20s} — {detail}")

            if crashed:
                term = r.get("termination", "unknown")
                print(red(f"           Cause: {term}"))
                sig = r.get("signal")
                if sig and sig not in term:
                    print(red(f"           Signal: {sig}"))
                for sig_line in r.get("crash_signatures", [])[1:]:
                    print(red(f"           Also found: {sig_line}"))
                for log_line in r.get("last_log_lines", []):
                    print(dim(f"           Last output: {log_line}"))
                for stderr_line in r.get("stderr_tail", []):
                    print(dim(f"           stderr: {stderr_line.rstrip()}"))

            if r.get("sticky_hip_error"):
                print(yellow("           Sticky HIP error detected — GPU may be in bad state"))
                overall_pass = False

            # Diagnostic "*** ... ***" lines explain a role's own verdict. Do not
            # independently fail a clean role just because it logged bounded
            # stress/anomaly evidence; non-zero exits and parsed error counts
            # already make the role fail above.
            for line in r.get("failure_lines", [])[:3]:
                print(red(f"           {line.strip()}"))

        # Health
        print(f"\n  {bold('--- Health Monitor (Leak Detection) ---')}")
        for h in gpu_results["health"]:
            name = h["name"]
            result = h["result"]
            if result["status"] == "OK":
                print(f"    {green('[OK]  ')} {name} — {result.get('duration_sec', '?'):.0f}s, no leaks")
            elif result["status"] == "WARN":
                overall_pass = False
                health_warnings += 1
                print(f"    {yellow('[WARN]')} {name}:")
                for issue in result["issues"]:
                    print(yellow(f"             {issue}"))
            elif result["status"] == "INFO":
                print(f"    {dim('[INFO]')} {name}:")
                for issue in result["issues"]:
                    print(dim(f"             {issue}"))
            else:
                print(f"    {dim('[SKIP]')} {name} — {result.get('reason', '')}")

        # Monitor
        print(f"\n  {bold('--- Monitor (sysfs Query Latency) ---')}")
        for m in gpu_results["monitor"]:
            name = m["name"]
            r = m["result"]
            slow = r["slow_queries"]
            if slow > 0:
                overall_pass = False
                tag = yellow("[WARN]")
            else:
                tag = green("[OK]  ")
            print(
                f"    {tag} {name} — {r['total_queries']} queries, " f"{slow} slow (>100ms), max={r['max_query_us']}us"
            )

        # Profiler
        print(f"\n  {bold('--- Profiler (Timing Integrity) ---')}")
        for p in gpu_results["profiler"]:
            name = p["name"]
            r = p["result"]
            neg = r["negative_times"]
            timing_total_count += r["total_kernels"]
            timing_anomaly_count += neg
            if neg > 0:
                overall_pass = False
                tag = yellow("[WARN]")
            else:
                tag = green("[OK]  ")
            print(f"    {tag} {name} — {r['total_kernels']} kernels, " f"{neg} negative/zero times")

    # Executive summary
    sysinfo_fields = parse_sysinfo_fields(sysinfo)
    env_parts = []
    if sysinfo_fields.get("rocm_version"):
        env_parts.append(f"ROCm {sysinfo_fields['rocm_version']}")
    if sysinfo_fields.get("gpu_model"):
        cnt = sysinfo_fields.get("gpu_count", "?")
        env_parts.append(f"{sysinfo_fields['gpu_model']} x{cnt}")
    if sysinfo_fields.get("kernel"):
        env_parts.append(f"Linux {sysinfo_fields['kernel']}")
    env_line = " | ".join(env_parts) if env_parts else "unknown"

    anomaly_pct = (100.0 * timing_anomaly_count / timing_total_count) if timing_total_count > 0 else 0

    if total_processes == 0:
        overall_pass = False

    verdict = green(bold("PASS")) if overall_pass else red(bold("FAIL"))

    print(f"\n{bold('=' * 70)}")
    print(bold("  EXECUTIVE SUMMARY"))
    print(bold("=" * 70))
    if env_parts:
        print(f"  Environment:      {env_line}")
    if sysinfo_fields.get("hostname"):
        print(f"  Hostname:         {sysinfo_fields['hostname']}")
    print(f"  GPUs analyzed:    {len(dirs_to_analyze)}")
    if total_processes == 0:
        print(red("  Total processes:  0 (no logs found — did the test run?)"))
    else:
        print(f"  Total processes:  {total_processes}")
    print(f"  Result:           {verdict}")
    print("")

    if total_crashed > 0:
        print(red(f"  Process crashes:  {total_crashed}"))
        for cd in crash_details:
            print(red(f"    - {cd['gpu']}/{cd['role']}: {cd['cause']}"))
    else:
        print(green("  Process crashes:  0 (all processes completed normally)"))

    print(
        f"  Data integrity:   {total_checks:,} checks, "
        f"{green('0 errors') if total_errors == 0 else red(f'{total_errors:,} errors')}"
    )

    if total_ipc_rounds > 0:
        ipc_status = green("0 errors") if total_ipc_errors == 0 else red(f"{total_ipc_errors:,} errors")
        print(f"  IPC shared memory: {total_ipc_rounds:,} rounds, {ipc_status}")

    if total_alloc_failures > 0:
        print(f"  Alloc failures:   {total_alloc_failures:,} (all handled gracefully — not errors)")
    else:
        print("  Alloc failures:   0")

    leak_status = green("0 warnings") if health_warnings == 0 else yellow(f"{health_warnings} warning(s)")
    print(f"  Resource leaks:   {leak_status}")

    anomaly_str = f"{anomaly_pct:.3f}% ({timing_anomaly_count:,} / {timing_total_count:,} kernels)"
    print(f"  Timing anomalies: {anomaly_str}")
    print(bold("=" * 70))

    return overall_pass


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


def write_json(path, top_dir, dirs_to_analyze, all_results, overall_pass):
    sysinfo_fields = parse_sysinfo_fields(read_sysinfo(top_dir))
    output = {
        "tool": "rock_mps_test",
        "results_dir": top_dir,
        "overall_pass": overall_pass,
        "environment": sysinfo_fields,
        "gpus": {},
    }

    for gpu_label, _results_dir in dirs_to_analyze:
        gpu_data = all_results[gpu_label]
        gpu_out = {"roles": [], "health": [], "monitor": [], "profiler": []}

        for r in gpu_data["roles"]:
            entry = dict(r)
            entry.pop("failure_lines", None)
            entry.pop("crash_signatures", None)
            entry["pass"] = (not r.get("crashed", False)) and r.get("exit_code", 1) == 0
            entry["crashed"] = r.get("crashed", False)
            if r.get("termination"):
                entry["termination"] = r["termination"]
            if r.get("signal"):
                entry["signal"] = r["signal"]
            tp = format_throughput(r)
            if tp:
                entry["throughput"] = tp
            gpu_out["roles"].append(entry)

        for h in gpu_data["health"]:
            gpu_out["health"].append({"name": h["name"], **h["result"]})
        for m in gpu_data["monitor"]:
            gpu_out["monitor"].append({"name": m["name"], **m["result"]})
        for p in gpu_data["profiler"]:
            gpu_out["profiler"].append({"name": p["name"], **p["result"]})

        output["gpus"][gpu_label] = gpu_out

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nJSON report written to: {path}")


# ---------------------------------------------------------------------------
# JUnit XML output
# ---------------------------------------------------------------------------


def write_junit(  # noqa: C901 — vendored analyzer; complexity is intentional
    path, top_dir, dirs_to_analyze, all_results, overall_pass
):
    testsuites = ET.Element("testsuites", name="rock_mps_test")
    total_tests = 0
    total_failures = 0

    for gpu_label, _results_dir in dirs_to_analyze:
        gpu_data = all_results[gpu_label]
        tests = 0
        failures = 0
        suite_cases = []

        for r in gpu_data["roles"]:
            tests += 1
            crashed = r.get("crashed", False)
            passed = (not crashed) and r.get("exit_code", 1) == 0
            elapsed = r.get("elapsed_sec", 0)

            tc = ET.SubElement(
                ET.Element("_"),
                "testcase",
                name=r["role"],
                classname=f"rock_mps_test.{gpu_label}",
                time=f"{elapsed:.1f}",
            )

            iters = r.get("iterations", "?")
            errors = r.get("errors", "?")
            checks = r.get("checks")
            detail = f"{iters} iterations, {errors} errors"
            if checks is not None:
                detail = f"{iters} iterations, {checks} checks, {errors} errors"

            stdout = ET.SubElement(tc, "system-out")
            stdout.text = detail

            if crashed:
                failures += 1
                term = r.get("termination", "process crashed")
                err_el = ET.SubElement(tc, "error", message=term[:200], type="ProcessCrash")
                parts = [term]
                sig = r.get("signal")
                if sig and sig not in term:
                    parts.append(f"Signal: {sig}")
                for cs in r.get("crash_signatures", []):
                    parts.append(f"Crash signature: {cs}")
                for ll in r.get("last_log_lines", []):
                    parts.append(f"Last output: {ll}")
                for sl in r.get("stderr_tail", []):
                    parts.append(f"stderr: {sl}")
                err_el.text = "\n".join(parts)
            elif not passed:
                failures += 1
                fail_el = ET.SubElement(tc, "failure", message=f"{errors} errors in {iters} iterations")
                fail_lines = r.get("failure_lines", [])
                fail_el.text = "\n".join(fail_lines[:5]) if fail_lines else detail

            suite_cases.append(tc)

        # Health warnings as test cases
        for h in gpu_data["health"]:
            tests += 1
            tc = ET.SubElement(
                ET.Element("_"), "testcase", name=f"health_{h['name']}", classname=f"rock_mps_test.{gpu_label}"
            )
            if h["result"]["status"] == "WARN":
                failures += 1
                fail_el = ET.SubElement(tc, "failure", message="Resource leak detected")
                fail_el.text = "\n".join(h["result"].get("issues", []))
            suite_cases.append(tc)

        suite = ET.SubElement(testsuites, "testsuite", name=gpu_label, tests=str(tests), failures=str(failures))
        for tc in suite_cases:
            suite.append(tc)

        total_tests += tests
        total_failures += failures

    testsuites.set("tests", str(total_tests))
    testsuites.set("failures", str(total_failures))

    rough = ET.tostring(testsuites, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    # Remove extra XML declaration
    lines = pretty.split("\n")
    if lines and lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"JUnit XML report written to: {path}")


def _health_thresholds_for_role(role):
    """Return analyzer thresholds for a role's health CSV."""
    if role and role.startswith("memory_mover"):
        return _MEMORY_MOVER_HEALTH_WARN_MB, _MEMORY_MOVER_HEALTH_WARN_MB
    return _DEFAULT_HEALTH_WARN_MB, _DEFAULT_HEALTH_WARN_MB


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Analyze results from rock_mps_test runs.")
    parser.add_argument("results_dir", help="Results directory (single GPU or parent with gpu0/, gpu1/, ...)")
    parser.add_argument("--json", metavar="PATH", help="Write JSON report to PATH")
    parser.add_argument("--junit", metavar="PATH", help="Write JUnit XML report to PATH")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    args = parser.parse_args()

    global _USE_COLOR
    if args.no_color:
        _USE_COLOR = False

    top_dir = args.results_dir
    if not os.path.isdir(top_dir):
        print(f"Not a directory: {top_dir}")
        sys.exit(1)

    # Detect structure
    gpu_dirs = sorted(glob.glob(os.path.join(top_dir, "gpu*")))
    if gpu_dirs:
        dirs_to_analyze = [(os.path.basename(d), d) for d in gpu_dirs if os.path.isdir(d)]
    else:
        basename = os.path.basename(os.path.normpath(top_dir))
        label = basename if basename.startswith("gpu") else "gpu0"
        dirs_to_analyze = [(label, top_dir)]

    # Collect all results
    all_results = {}
    for gpu_label, results_dir in dirs_to_analyze:
        gpu = {"roles": [], "health": [], "monitor": [], "profiler": []}
        role_by_pid = {}

        for log_path in sorted(glob.glob(os.path.join(results_dir, "*_stdout.log"))):
            role_result = analyze_stdout_log(log_path)
            gpu["roles"].append(role_result)
            if role_result.get("pid") is not None:
                role_by_pid[role_result["pid"]] = role_result["role"]

        for csv_path in sorted(glob.glob(os.path.join(results_dir, "health_*.csv"))):
            pid_match = re.search(r"pid(\d+)", os.path.basename(csv_path))
            role = role_by_pid.get(int(pid_match.group(1))) if pid_match else None
            rss_warn_mb, vram_warn_mb = _health_thresholds_for_role(role)
            gpu["health"].append(
                {
                    "name": os.path.basename(csv_path),
                    "role": role,
                    "result": analyze_health_csv(csv_path, rss_warn_mb=rss_warn_mb, vram_warn_mb=vram_warn_mb),
                }
            )

        for csv_path in sorted(glob.glob(os.path.join(results_dir, "monitor_*.csv"))):
            gpu["monitor"].append(
                {
                    "name": os.path.basename(csv_path),
                    "result": analyze_monitor_csv(csv_path),
                }
            )

        for csv_path in sorted(glob.glob(os.path.join(results_dir, "profiler_*.csv"))):
            gpu["profiler"].append(
                {
                    "name": os.path.basename(csv_path),
                    "result": analyze_profiler_csv(csv_path),
                }
            )

        all_results[gpu_label] = gpu

    # Print report
    overall_pass = print_report(top_dir, dirs_to_analyze, all_results)

    # Optional outputs
    if args.json:
        write_json(args.json, top_dir, dirs_to_analyze, all_results, overall_pass)

    if args.junit:
        write_junit(args.junit, top_dir, dirs_to_analyze, all_results, overall_pass)

    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
