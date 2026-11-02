# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PTL runtime toggle under RCCL load.

System-level stress test for the PTL runtime toggle path. Drives the
per-GPU ptl_enable sysfs node at high frequency while RCCL collectives
are actively running across all GPUs in the node.

Pre-conditions:
  - amd-smi available on PATH
  - mpirun available on PATH
  - all_reduce_perf available under $ROCM_PATH/bin/ (or via $RCCL_BIN)
  - amdgpu loaded with ptl=1; ptl=2 causes a clean SKIP
  - rocprof NOT running (silently suppresses PTL)
  - Root or write access to /sys/class/drm/card*/device/ptl/ptl_enable

Exit codes: 0 = PASS, 1 = FAIL, 77 = SKIP.
"""

from __future__ import annotations

import csv
from datetime import datetime
import glob
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

# ============================================================
# Config (env-overridable)
# ============================================================

ROCM_PATH = os.environ.get("ROCM_PATH", "/opt/rocm")
RCCL_BIN = os.environ.get("RCCL_BIN", f"{ROCM_PATH}/bin/all_reduce_perf")
MPIRUN = os.environ.get("MPIRUN", "mpirun")
AMD_SMI = os.environ.get("AMD_SMI", "amd-smi")
RCCL_ITER_COUNT = int(os.environ.get("RCCL_ITER_COUNT", "5"))
TOGGLE_INTERVAL_SECS = float(os.environ.get("TOGGLE_INTERVAL_SECS", "0.5"))
AMD_SMI_STATIC_EVERY_N_TICKS = int(os.environ.get("AMD_SMI_STATIC_EVERY_N_TICKS", "20"))
WATCHDOG_SECS = int(os.environ.get("WATCHDOG_SECS", "0"))
RUN_DIR = pathlib.Path(os.environ.get("RUN_DIR", f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
SYSFS_PTL_GLOB = os.environ.get("SYSFS_PTL_GLOB", "/sys/class/drm/card*/device/ptl/ptl_enable")
RCCL_MIN_BYTES = os.environ.get("RCCL_MIN_BYTES", "8M")
RCCL_MAX_BYTES = os.environ.get("RCCL_MAX_BYTES", "1G")
RCCL_FACTOR = int(os.environ.get("RCCL_FACTOR", "2"))
RCCL_ITERS = int(os.environ.get("RCCL_ITERS", "20"))
RCCL_WARMUP = int(os.environ.get("RCCL_WARMUP", "5"))
MPIRUN_EXTRA: list[str] = shlex.split(os.environ.get("MPIRUN_EXTRA", "--allow-run-as-root --bind-to numa"))

_EXIT_FAIL = 1
_EXIT_SKIP = 77

# ============================================================
# Logging
# ============================================================

_log_path: pathlib.Path | None = None
_log_lock = threading.Lock()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _write_log(line: str) -> None:
    if _log_path:
        with _log_lock, _log_path.open("a") as f:
            f.write(line + "\n")


def log(msg: str) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    _write_log(line)


def warn(msg: str) -> None:
    line = f"[{_ts()}] WARN: {msg}"
    print(line, file=sys.stderr, flush=True)
    _write_log(line)


def die(msg: str) -> None:
    line = f"[{_ts()}] FAIL: {msg}"
    print(line, file=sys.stderr, flush=True)
    _write_log(line)
    sys.exit(_EXIT_FAIL)


def skip(msg: str) -> None:
    line = f"[{_ts()}] SKIP: {msg}"
    print(line, file=sys.stderr, flush=True)
    _write_log(line)
    sys.exit(_EXIT_SKIP)


# ============================================================
# PTL sysfs helpers
# ============================================================


def ptl_norm(val: str) -> str:
    """Normalize sysfs read form ('enabled'/'disabled') to 0/1."""
    if val in ("1", "enabled"):
        return "1"
    if val in ("0", "disabled"):
        return "0"
    return val


def read_ptl_node(node: str) -> str:
    try:
        return pathlib.Path(node).read_text().strip()
    except Exception:
        return "?"


def write_ptl_node(node: str, val: str) -> bool:
    """Write val to a PTL sysfs node; try direct write first, fall back to sudo tee."""
    try:
        if os.access(node, os.W_OK):
            pathlib.Path(node).write_text(val)
            return True
        result = subprocess.run(
            ["sudo", "tee", node],
            input=val,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# ============================================================
# Preflight
# ============================================================


def _get_asic_name() -> str:
    try:
        r = subprocess.run([AMD_SMI, "static"], capture_output=True, text=True, timeout=35)
        m = re.search(r"MARKET_NAME\s*:\s*(.+)", r.stdout, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return "unknown ASIC"


def preflight(ptl_nodes: list[str]) -> None:
    log("preflight: checking environment")

    for cmd in [MPIRUN, AMD_SMI]:
        if not shutil.which(cmd):
            skip(f"{cmd} not on PATH")

    rccl = RCCL_BIN
    if not os.access(rccl, os.X_OK):
        found = shutil.which(rccl)
        if found:
            globals()["RCCL_BIN"] = found
        else:
            skip(f"{rccl} not executable and not on PATH — set RCCL_BIN to the right path")

    if not ptl_nodes:
        asic = _get_asic_name()
        log(f"preflight: ASIC detected as '{asic}' — PTL sysfs nodes absent")
        skip(
            f"ASIC '{asic}' does not expose runtime PTL toggle "
            f"({SYSFS_PTL_GLOB} not found) — load amdgpu with ptl=1 to enable"
        )

    for node in ptl_nodes:
        if not os.access(node, os.R_OK):
            skip(f"cannot read {node} (need root or chmod 666)")
        if not os.access(node, os.W_OK):
            r = subprocess.run(["sudo", "-n", "true"], capture_output=True)
            if r.returncode != 0:
                skip(f"cannot write {node} — need direct write access or passwordless sudo")

    for proc in ("rocprof", "rocprofv2", "rocprof-sdk"):
        r = subprocess.run(["pgrep", "-x", proc], capture_output=True)
        if r.returncode == 0:
            skip(f"{proc} process running — PTL would be suppressed; abort to keep the test honest")

    log(f"preflight: ok ({len(ptl_nodes)} per-GPU PTL nodes, mpirun={MPIRUN}, rccl={RCCL_BIN})")


# ============================================================
# GPU / PTL discovery
# ============================================================


def detect_n_gpus(ptl_nodes: list[str]) -> int:
    try:
        r = subprocess.run([AMD_SMI, "list"], capture_output=True, text=True, timeout=35)
        n = len(re.findall(r"^GPU: \d+$", r.stdout, re.MULTILINE))
        if n >= 1:
            return n
    except Exception:
        pass
    n = len(ptl_nodes)
    if n < 1:
        die("could not detect any GPUs via amd-smi or PTL sysfs")
    return n


def snapshot_ptl_state(ptl_nodes: list[str], outfile: pathlib.Path) -> None:
    lines = []
    for node in ptl_nodes:
        m = re.search(r"/(card\d+)/", node)
        card = m.group(1) if m else "unknown"
        val = read_ptl_node(node)
        lines.append(f"{card} {node} {val}")
    outfile.write_text("\n".join(lines) + "\n")
    log(f"snapshot_ptl_state -> {outfile}")


def ensure_ptl_enabled(ptl_nodes: list[str]) -> None:
    log("ensure_ptl_enabled: forcing every GPU's ptl_enable to 1")

    ptl_param = pathlib.Path("/sys/module/amdgpu/parameters/ptl")
    if ptl_param.exists():
        mode = ptl_param.read_text().strip()
        if mode == "2":
            skip("amdgpu loaded with ptl=2 (permanent disable) — runtime PTL toggle unavailable")

    for node in ptl_nodes:
        if not write_ptl_node(node, "1"):
            die(f"failed to write 1 to {node} — amdgpu refused the change")
        rb = read_ptl_node(node)
        if ptl_norm(rb) != "1":
            die(
                f"wrote 1 to {node} but readback is {rb!r} "
                f"(normalized {ptl_norm(rb)!r}) — runtime PTL changes appear blocked"
            )

    log(f"ensure_ptl_enabled: ok (all {len(ptl_nodes)} GPUs enabled)")


def restore_ptl_state(ptl_nodes: list[str], orig_file: pathlib.Path, final_file: pathlib.Path) -> bool:
    log(f"restore_ptl_state: replaying {orig_file}")
    ok = True
    for line in orig_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        card, node, val = parts[0], parts[1], parts[2]
        if val == "?":
            continue
        want = ptl_norm(val)
        if not write_ptl_node(node, want):
            warn(f"restore: write '{want}' (from '{val}') -> {node} failed")
            ok = False
            continue
        rb_norm = ptl_norm(read_ptl_node(node))
        if rb_norm != want:
            warn(f"restore: {card} readback '{rb_norm}' != expected '{want}' (from '{val}')")
            ok = False
    snapshot_ptl_state(ptl_nodes, final_file)
    return ok


# ============================================================
# dmesg helpers
# ============================================================

_DMESG_CRITICAL_RE = re.compile(
    r"Kernel panic|panic|watchdog:.*(?:BUG|soft lockup|hard lockup)|"
    r"soft lockup|hard lockup|BUG:|kernel BUG|Oops:\s*[0-9a-fA-F]+|"
    r"Call Trace|call trace|blocked for more than|hung task|"
    r"gpu reset|amdgpu.*reset|ring .*timeout|GPU fault|"
    r"RAS.*error|IOMMU.*fault|DMAR.*fault|segfault|segmentation fault|"
    r"(?:amdgpu|drm|kfd).*\b[Pp][Tt][Ll]\b",
    re.IGNORECASE,
)


def read_dmesg() -> str:
    for cmd in [["dmesg", "--ctime"], ["sudo", "dmesg", "--ctime"]]:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                return r.stdout
        except Exception:
            pass
    warn("dmesg unreadable — dmesg oracle will be skipped")
    return ""


# ============================================================
# Background workers
# ============================================================


class _WorkerState:
    def __init__(self) -> None:
        self.stop = threading.Event()
        self.watchdog_fired = threading.Event()
        self._rccl_pgid: int | None = None
        self._lock = threading.Lock()

    def set_rccl_pgid(self, pgid: int | None) -> None:
        with self._lock:
            self._rccl_pgid = pgid

    def kill_rccl_tree(self, sig: int = signal.SIGTERM) -> None:
        with self._lock:
            pgid = self._rccl_pgid
        if pgid is None:
            return
        try:
            os.killpg(pgid, sig)  # type: ignore[attr-defined]
        except ProcessLookupError:
            pass
        except Exception as exc:
            warn(f"kill_rccl_tree: {exc}")


def _rccl_loop(n_gpus: int, rccl_log: pathlib.Path, state: _WorkerState) -> None:
    with rccl_log.open("a") as fh:
        for i in range(1, RCCL_ITER_COUNT + 1):
            if state.stop.is_set():
                break
            fh.write(f"rccl iter {i}/{RCCL_ITER_COUNT}: launching\n")
            fh.flush()
            cmd = [
                MPIRUN,
                "-np",
                str(n_gpus),
                *MPIRUN_EXTRA,
                RCCL_BIN,
                "-b",
                RCCL_MIN_BYTES,
                "-e",
                RCCL_MAX_BYTES,
                "-f",
                str(RCCL_FACTOR),
                "-g",
                "1",
                "-n",
                str(RCCL_ITERS),
                "-w",
                str(RCCL_WARMUP),
            ]
            try:
                # start_new_session=True puts mpirun in its own process group
                # so we can kill the entire mpirun tree cleanly via killpg.
                proc = subprocess.Popen(cmd, stdout=fh, stderr=fh, start_new_session=True)
                state.set_rccl_pgid(os.getpgid(proc.pid))  # type: ignore[attr-defined]
                rc = proc.wait()
                state.set_rccl_pgid(None)
            except Exception as exc:
                fh.write(f"rccl iter {i}: exception: {exc}\n")
                rc = -1
            fh.write(f"rccl iter {i}: rc={rc}\n")
            fh.flush()
            if i < RCCL_ITER_COUNT and not state.stop.is_set():
                time.sleep(1)
        fh.write(f"rccl loop: exit after {RCCL_ITER_COUNT} iter(s)\n")


def _ptl_toggler(
    ptl_nodes: list[str],
    rccl_thread: threading.Thread,
    trace_file: pathlib.Path,
    run_dir: pathlib.Path,
    state: _WorkerState,
) -> None:
    time.sleep(2)  # let mpirun launch the first iter before toggling
    tick = 0
    desired = 0
    with trace_file.open("a", newline="") as tf:
        writer = csv.writer(tf)
        while rccl_thread.is_alive() and not state.stop.is_set():
            tick += 1
            desired = 1 - desired
            ts = f"{time.time():.6f}"
            for node in ptl_nodes:
                m = re.search(r"/(card\d+)/", node)
                card = m.group(1) if m else "unknown"
                ok_write = write_ptl_node(node, str(desired))
                rb = read_ptl_node(node)
                ok = 1 if (ok_write and ptl_norm(rb) == str(desired)) else 0
                writer.writerow([ts, tick, card, node, desired, rb, ok])
            tf.flush()
            if tick % AMD_SMI_STATIC_EVERY_N_TICKS == 0:
                snap = run_dir / f"amd_smi_static_tick_{tick}.txt"
                try:
                    r = subprocess.run(
                        [AMD_SMI, "static", "-l"],
                        capture_output=True,
                        text=True,
                        timeout=35,
                    )
                    snap.write_text(r.stdout)
                except Exception:
                    pass
            time.sleep(TOGGLE_INTERVAL_SECS)
    (run_dir / "toggler_total_ticks").write_text(str(tick))
    log(f"toggler exit: total={tick}")


# ============================================================
# Watchdog
# ============================================================


def arm_watchdog(state: _WorkerState) -> threading.Timer | None:
    if WATCHDOG_SECS <= 0:
        log("watchdog: disabled (WATCHDOG_SECS=0)")
        return None
    log(f"watchdog: {WATCHDOG_SECS}s hard ceiling")

    def _fire() -> None:
        warn("WATCHDOG fired — terminating run")
        state.watchdog_fired.set()
        state.stop.set()
        state.kill_rccl_tree(signal.SIGTERM)
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Timer(WATCHDOG_SECS, _fire)
    timer.daemon = True
    timer.start()
    return timer


# ============================================================
# Verdict helpers
# ============================================================


def _count_toggle_stats(trace_file: pathlib.Path):
    total = ok_count = bad_count = 0
    if trace_file.exists():
        with trace_file.open(newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 7:
                    total += 1
                    if row[6].strip() == "1":
                        ok_count += 1
                    else:
                        bad_count += 1
    return total, ok_count, bad_count


def _count_rccl_stats(rccl_log: pathlib.Path):
    if not rccl_log.exists():
        return 0, 0
    text = rccl_log.read_text()
    runs = len(re.findall(r"^rccl iter \d+/\d+: launching", text, re.MULTILINE))
    # rc=0 is success; anything else is failure
    fails = len(re.findall(r"iter \d+: rc=(?!0\n)(\S+)", text))
    return runs, fails


# ============================================================
# Main
# ============================================================


def main() -> None:
    global _log_path

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _log_path = RUN_DIR / "run.log"

    log(f"ptl_runtime_toggle_rccl_workload: starting (run_dir={RUN_DIR})")

    ptl_nodes = sorted(glob.glob(SYSFS_PTL_GLOB))
    preflight(ptl_nodes)

    n_gpus = detect_n_gpus(ptl_nodes)
    log(f"detect_n_gpus: {n_gpus}")

    dmesg_before_file = RUN_DIR / "dmesg_before.txt"
    dmesg_after_file = RUN_DIR / "dmesg_after.txt"
    dmesg_delta_file = RUN_DIR / "dmesg_delta.txt"
    orig_ptl_file = RUN_DIR / "ptl_state_before.txt"
    final_ptl_file = RUN_DIR / "ptl_state_after.txt"
    rccl_log = RUN_DIR / "rccl.log"
    toggle_trace = RUN_DIR / "toggle_trace.csv"

    log("snapshotting dmesg baseline")
    dmesg_before_file.write_text(read_dmesg())

    snapshot_ptl_state(ptl_nodes, orig_ptl_file)
    ensure_ptl_enabled(ptl_nodes)

    with toggle_trace.open("w", newline="") as f:
        csv.writer(f).writerow(["tick_ts", "tick_idx", "card", "node", "desired", "readback", "ok"])

    state = _WorkerState()
    watchdog_timer = arm_watchdog(state)

    rccl_thread = threading.Thread(target=_rccl_loop, args=(n_gpus, rccl_log, state), daemon=True)
    rccl_thread.start()
    log(f"start_rccl_workload: launched {n_gpus}-rank {RCCL_BIN}, {RCCL_ITER_COUNT} iter(s)")

    toggler_thread = threading.Thread(
        target=_ptl_toggler,
        args=(ptl_nodes, rccl_thread, toggle_trace, RUN_DIR, state),
        daemon=True,
    )
    toggler_thread.start()
    log(f"start_toggler: interval={TOGGLE_INTERVAL_SECS}s")

    log(f"main: waiting for RCCL loop to complete {RCCL_ITER_COUNT} iter(s) ...")
    rccl_thread.join()

    log("main: waiting for toggler to finish ...")
    toggler_thread.join()

    if state.watchdog_fired.is_set():
        if watchdog_timer:
            watchdog_timer.cancel()
        die(f"watchdog fired — run exceeded {WATCHDOG_SECS}s ceiling")

    if watchdog_timer:
        watchdog_timer.cancel()

    log("snapshotting dmesg after")
    dmesg_after_text = read_dmesg()
    dmesg_after_file.write_text(dmesg_after_text)

    before_lines = set(dmesg_before_file.read_text().splitlines())
    delta_lines = [ln for ln in dmesg_after_text.splitlines() if ln not in before_lines]
    dmesg_delta_file.write_text("\n".join(delta_lines))

    restored_ok = restore_ptl_state(ptl_nodes, orig_ptl_file, final_ptl_file)

    # ============================================================
    # Verdict
    # ============================================================
    log("computing verdict")

    total_writes, ok_writes, bad_writes = _count_toggle_stats(toggle_trace)
    ticks_file = RUN_DIR / "toggler_total_ticks"
    total_ticks = int(ticks_file.read_text().strip()) if ticks_file.exists() else 0

    rccl_runs, rccl_fail = _count_rccl_stats(rccl_log)

    dmesg_hits = sum(1 for ln in delta_lines if _DMESG_CRITICAL_RE.search(ln))

    try:
        r = subprocess.run([AMD_SMI, "list"], capture_output=True, text=True, timeout=35)
        gpus_after = len(re.findall(r"^GPU: \d+$", r.stdout, re.MULTILINE))
    except Exception:
        gpus_after = 0

    verdict = "PASS"
    fail_reasons: list[str] = []
    if bad_writes != 0:
        verdict = "FAIL"
        fail_reasons.append(f"ptl_writes_bad={bad_writes}")
    if rccl_fail != 0:
        verdict = "FAIL"
        fail_reasons.append(f"rccl_runs_failed={rccl_fail}")
    if dmesg_hits != 0:
        verdict = "FAIL"
        fail_reasons.append(f"dmesg_critical={dmesg_hits}")
    if gpus_after != n_gpus:
        verdict = "FAIL"
        fail_reasons.append(f"gpu_count_changed_from_{n_gpus}_to_{gpus_after}")
    if not restored_ok:
        verdict = "FAIL"
        fail_reasons.append("ptl_state_not_restored")

    log("==================== VERDICT ====================")
    log(f"result                    : {verdict}")
    if fail_reasons:
        log(f"fail_reasons              : {' '.join(fail_reasons)}")
    log(f"n_gpus                    : {n_gpus} (after={gpus_after})")
    log(f"ptl_writes_total          : {total_writes}")
    log(f"ptl_writes_ok             : {ok_writes}")
    log(f"ptl_writes_bad            : {bad_writes}")
    log(f"toggler_ticks_total       : {total_ticks}")
    log(f"rccl_runs_attempted       : {rccl_runs} (of {RCCL_ITER_COUNT} requested)")
    log(f"rccl_runs_failed          : {rccl_fail}")
    log(f"dmesg_critical_delta      : {dmesg_hits}")
    log(f"ptl_state_restored        : {'yes' if restored_ok else 'no'}")
    log(f"run_dir                   : {RUN_DIR}")
    log("=================================================")

    sys.exit(0 if verdict == "PASS" else _EXIT_FAIL)


# ============================================================
# Entry point — cleanup on abort
# ============================================================

_global_state: _WorkerState | None = None


def _signal_handler(signum: int, frame: object) -> None:
    warn("signal caught — stopping workers and restoring PTL state")
    if _global_state is not None:
        _global_state.stop.set()
        _global_state.kill_rccl_tree(signal.SIGTERM)
        time.sleep(1)
        _global_state.kill_rccl_tree(signal.SIGKILL)  # type: ignore[attr-defined]


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    _global_state = _WorkerState()
    main()
