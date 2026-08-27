# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU monitoring context manager (amd-smi monitor subprocess + optional CU occupancy)."""

from __future__ import annotations

import contextlib
import csv
from dataclasses import dataclass, field
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from tests.common.gpu_monitored import csv_schema

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor
    from framework.executors.background_process import AbstractBackgroundProcess


_DYNAMIC_TELEMETRY_FIELDS = (
    csv_schema.POWER_USAGE,
    csv_schema.HOTSPOT_TEMP,
    csv_schema.MEM_TEMP,
    csv_schema.GFX_CLK,
    csv_schema.GFX_UTIL,
    csv_schema.MEM_UTIL,
    csv_schema.MEM_CLK,
    csv_schema.VRAM_USED,
    csv_schema.VRAM_PCT,
)


def resolve_amd_smi(rocm_root: Path | str | None = None) -> str:
    """Return amd-smi binary path (PATH first, then ``{rocm_root}/bin/amd-smi``).

    Mirrors ``health_plugin.GpuHealthChecker._resolve_amd_smi`` so gpu_monitored
    monitoring works when ROCm is only visible via ``--rock-dir``.
    """
    if shutil.which("amd-smi"):
        return "amd-smi"
    if rocm_root:
        candidate = Path(rocm_root) / "bin" / "amd-smi"
        if candidate.is_file():
            return str(candidate)
    return "amd-smi"


def _has_finite_telemetry(row: dict) -> bool:
    return any(_finite_field(row, metric) is not None for metric in _DYNAMIC_TELEMETRY_FIELDS)


@dataclass
class MonitoringEvidence:
    sample_count: int = 0
    active_sample_count: int = 0
    sampled_gpus: set[str] = field(default_factory=set)
    active_gpus: set[str] = field(default_factory=set)
    power_gpus: set[str] = field(default_factory=set)
    hotspot_gpus: set[str] = field(default_factory=set)
    # Rows dropped while scanning, kept so a caller can say *why* a sample
    # count came out lower than the run length implies. Dropping is the
    # right behaviour -- a garbled row must not fail an otherwise healthy
    # run -- but doing it without a trace turns amd-smi writing corrupt
    # output into a plausible-looking smaller sample count.
    undecodable_rows: int = 0  # csv reader could not decode the row at all
    unparsable_rows: int = 0  # row decoded but has no usable timestamp/GPU
    scan_aborted: bool = False  # hit _MAX_CORRUPT_ROWS; tail never read
    # Wall-clock extent of the telemetry, and the longest stretch between two
    # consecutive samples. The per-GPU coverage sets above say *which* GPUs
    # were seen but nothing about *when*, so a monitor that stops early still
    # satisfies them from whatever window it managed to record. These make the
    # run's telemetry auditable in time as well.
    first_timestamp: int | None = None
    last_timestamp: int | None = None
    # Worst per-GPU figures, not fleet-wide ones. A fleet-wide timestamp set
    # is dense as long as *any* GPU is still reporting, so a monitor that
    # left one GPU sampling while the rest went dark produced a full-looking
    # window: on an 8-GPU host with 7 dark for 540 s of a 600 s run, every
    # gate passed. Coverage has to hold for each GPU independently.
    max_sample_gap_sec: int = 0
    # ``{gpu: (span_sec, max_interior_gap_sec)}``, the source of truth for
    # legacy callers that do not provide run boundaries.
    gpu_windows: dict[str, tuple[int, int]] = field(default_factory=dict)
    # Explicit wall-clock run boundaries and each GPU's first/last sample.
    # These let ``worst_coverage`` keep leading and trailing gaps separate;
    # ``duration - span`` incorrectly adds them and can call two individually
    # acceptable edge gaps one over-budget continuous blackout.
    coverage_start_timestamp: int | None = None
    coverage_end_timestamp: int | None = None
    gpu_sample_bounds: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def dropped_rows(self) -> int:
        return self.undecodable_rows + self.unparsable_rows

    @property
    def span_sec(self) -> int:
        """Seconds between the first and last usable sample."""
        if self.first_timestamp is None or self.last_timestamp is None:
            return 0
        return max(0, self.last_timestamp - self.first_timestamp)

    def worst_coverage(self, duration_sec: int) -> tuple[int, str | None]:
        """``(blind_sec, gpu)`` for the worst-covered GPU.

        Per GPU, the blind stretch is the largest of its leading gap, trailing
        gap, and largest interior gap. The fleet's figure is the worst of
        those, with the GPU responsible named so triage knows which device
        stopped reporting.

        Measured per GPU rather than across the fleet because the fleet-wide
        timestamp set stays dense while even one GPU keeps reporting, which
        hid exactly the partial-monitor-death case this gate exists to catch.
        A GPU that never appears at all is left to the identity gate, which
        names it directly.
        """
        if not self.gpu_windows:
            return self.max_sample_gap_sec, None
        worst_blind = -1
        worst_gpu: str | None = None
        for gpu, (span, gap) in sorted(self.gpu_windows.items()):
            bounds = self.gpu_sample_bounds.get(gpu)
            if (
                bounds is not None
                and self.coverage_start_timestamp is not None
                and self.coverage_end_timestamp is not None
            ):
                first, last = bounds
                leading = max(0, first - self.coverage_start_timestamp)
                trailing = max(0, self.coverage_end_timestamp - last)
                blind = max(leading, trailing, gap)
            else:
                # Compatibility for evidence assembled without absolute run
                # boundaries. Production passes them; older callers retain
                # the conservative total-edge-shortfall behavior.
                outside = max(0, duration_sec - span) if duration_sec > 0 else 0
                blind = max(outside, gap)
            if blind > worst_blind:
                worst_blind, worst_gpu = blind, gpu
        return max(0, worst_blind), worst_gpu

    def blind_sec(self, duration_sec: int) -> int:
        """Longest stretch of the run with no telemetry, worst GPU."""
        return self.worst_coverage(duration_sec)[0]

    @property
    def worst_covered_gpu(self) -> str | None:
        """The worst-covered GPU ignoring run length (span only).

        ``worst_coverage`` is the duration-aware answer; this exists for
        callers that only have the evidence.
        """
        if not self.gpu_windows:
            return None
        return min(self.gpu_windows.items(), key=lambda kv: kv[1][0])[0]


def _finite_field(row: dict, field_name: str) -> float | None:
    """One telemetry field as a physically usable number, or ``None``.

    Negatives are rejected, not just non-finite values. Every field this is
    used for -- power draw, temperature, clock, utilisation, VRAM -- is a
    non-negative physical quantity, so a negative is a sentinel for "no
    reading" from the driver or a garbled write, never a measurement.
    Accepting them let a CSV of ``-1`` power and hotspot values satisfy the
    per-GPU sensor-coverage gate, which is supposed to prove those sensors
    were actually read.
    """
    try:
        value = float(str(row.get(field_name, "")).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


class Monitor:
    """Context manager that runs `amd-smi monitor` in the background.

    Use as:
        with Monitor(csv_file=..., sample_interval=1) as mon:
            ... run workload ...
    The monitor subprocess is cleanly killed on __exit__ (replaces bash
    `trap cleanup_monitors EXIT`).
    """

    def __init__(
        self,
        csv_file: Path,
        cu_csv: Path,
        sample_interval: int = 1,
        enable_cu_occupancy: bool = False,
        monitor_executor: AbstractExecutor | None = None,
        rocm_root: Path | str | None = None,
    ):
        self.csv_file = Path(csv_file)
        self.cu_csv = Path(cu_csv)
        self.sample_interval = sample_interval
        self.enable_cu_occupancy = enable_cu_occupancy
        self.monitor_executor = monitor_executor
        self._amd_smi = resolve_amd_smi(rocm_root)
        self._monitor_proc: subprocess.Popen | None = None
        self._monitor_bg: AbstractBackgroundProcess | None = None
        # amd-smi's own stderr, kept so a mid-run death can say why. Written
        # beside the CSV and removed unless the monitor actually failed.
        self._monitor_err = Path(f"{csv_file}.monitor_stderr")
        self._monitor_started: float = 0.0
        self._cu_thread: threading.Thread | None = None
        self._cu_stop = threading.Event()
        self._cu_sample_proc: subprocess.Popen | None = None
        self._cu_sample_lock = threading.Lock()

    def _use_direct_monitor_popen(self) -> bool:
        """Exec amd-smi directly when the monitor executor is local.

        ``CpuExecutor.start_background`` wraps commands in ``/bin/sh``; killing
        the shell can orphan the amd-smi child. Direct ``Popen`` matches the
        original gpu_monitored suite and lets ``_kill_monitor`` stop the real
        process. Remote ``SshExecutor`` paths still use ``start_background``.
        """
        if self.monitor_executor is None:
            return True
        from framework.executors.cpu_executor import CpuExecutor

        return isinstance(self.monitor_executor, CpuExecutor)

    def _monitor_subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        ex = self.monitor_executor
        if ex is not None:
            overrides = getattr(ex, "env_overrides", None)
            if overrides:
                env.update(overrides)
        return env

    def _start_monitor_proc(self) -> None:
        try:
            err_fh = self._monitor_err.open("wb")
        except OSError:
            err_fh = None
        self._monitor_proc = subprocess.Popen(
            [
                self._amd_smi,
                "monitor",
                "-p",
                "-t",
                "-u",
                "-m",
                "-v",
                "-w",
                str(self.sample_interval),
                "--csv",
                "--overwrite",
                "--file",
                str(self.csv_file),
            ],
            stdout=subprocess.DEVNULL,
            stderr=err_fh if err_fh is not None else subprocess.DEVNULL,
            env=self._monitor_subprocess_env(),
        )
        if err_fh is not None:
            err_fh.close()

    def __enter__(self) -> Monitor:
        # Start amd-smi monitor (writes CSV file directly). stderr goes to a
        # file rather than DEVNULL: when the monitor dies part-way through a
        # run its message is the only explanation for a short CSV, and
        # discarding it left the failure looking like "the GPUs were idle".
        self._monitor_started = time.monotonic()
        if self._use_direct_monitor_popen():
            self._start_monitor_proc()
        else:
            amd_smi = shlex.quote(self._amd_smi)
            monitor_cmd = (
                f"{amd_smi} monitor -p -t -u -m -v -w {self.sample_interval} "
                f"--csv --overwrite --file {self.csv_file}"
            )
            self._monitor_bg = self.monitor_executor.start_background(
                monitor_cmd,
                log_path=str(self._monitor_err),
                stream=False,
            )

        # Anything that runs after Popen must not leak the amd-smi
        # monitor. The with-statement protocol does NOT call ``__exit__``
        # if ``__enter__`` raises, so we need our own cleanup here. The
        # try/except has to wrap *everything* after Popen — the earlier
        # version started the try only after ``time.sleep(0.5)``, which
        # left a window where a Ctrl-C during that sleep would propagate
        # KeyboardInterrupt before the cleanup was wired up and orphan
        # the amd-smi child.
        try:
            # Briefly confirm the monitor didn't exit immediately
            # (missing binary, permission error, bad flags). A silent
            # early exit produces an empty CSV and every downstream
            # "N/A" with no obvious cause in the console log.
            time.sleep(0.5)
            rc = self._monitor_poll()
            if rc is not None:
                print(
                    f"  [monitor] WARNING: amd-smi monitor exited immediately "
                    f"(rc={rc}) — monitoring CSV will be empty; check that "
                    f"amd-smi is installed and has permission to read GPU "
                    f"metrics"
                )

            # Only touch the CU csv when the feature is enabled — avoids
            # leaving a 1-line "empty" CSV in every per-test directory.
            if self.enable_cu_occupancy:
                self.cu_csv.write_text("timestamp,gpu,pid,cu_occupancy,vram_mb\n")
                self._cu_thread = threading.Thread(target=self._cu_loop, daemon=True)
                self._cu_thread.start()
        except BaseException:
            self._kill_monitor()
            raise
        return self

    def _monitor_poll(self) -> int | None:
        if self._monitor_bg is not None:
            return self._monitor_bg.poll()
        if self._monitor_proc is not None:
            return self._monitor_proc.poll()
        return None

    def _kill_monitor(self) -> None:
        """Best-effort termination of the amd-smi monitor subprocess."""
        if self._monitor_bg is not None:
            with contextlib.suppress(Exception):
                self._monitor_bg.stop(timeout=5)
            self._monitor_bg = None
            return
        proc = self._monitor_proc
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
                return
            except subprocess.TimeoutExpired:
                pass
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=5)
        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop the CU loop, and if it's currently blocked in subprocess.run
        # (up to 10s), kill the in-flight amd-smi process so join() returns
        # promptly instead of padding every test's teardown by up to 20s.
        self._cu_stop.set()
        with self._cu_sample_lock:
            proc = self._cu_sample_proc
            if proc is not None and proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.terminate()
        if self._cu_thread is not None:
            self._cu_thread.join(timeout=5)
        self._report_monitor_death()
        self._kill_monitor()
        return False

    def _report_monitor_death(self) -> None:
        """Say so if amd-smi stopped on its own before the workload finished.

        A monitor that dies mid-run leaves a CSV covering only the seconds
        before it went, which downstream reads as "no GPU activity" -- the
        workload gets blamed for an environment problem. The ``__enter__``
        probe only catches an immediate exit, so this covers the rest of the
        run. The verdict still fails on missing evidence; this supplies the
        reason.
        """
        proc = self._monitor_proc
        if proc is None and self._monitor_bg is None:
            self._monitor_err.unlink(missing_ok=True)
            return
        if self._monitor_poll() is None:
            self._monitor_err.unlink(missing_ok=True)
            return
        alive_for = max(0.0, time.monotonic() - self._monitor_started)
        detail = ""
        try:
            # amd-smi always writes a "'CTRL' + 'C' to stop watching" banner to
            # stderr, so the last line is not necessarily the failure. Report
            # the last line that isn't that banner.
            lines = [
                ln.strip()
                for ln in self._monitor_err.read_text(errors="replace").splitlines()
                if ln.strip() and "to stop watching output" not in ln
            ]
            if lines:
                detail = f": {lines[-1][:200]}"
        except OSError:
            pass
        rc = self._monitor_poll()
        print(
            f"  [monitor] WARNING: amd-smi monitor exited on its own after "
            f"{alive_for:.0f}s (rc={rc}) — telemetry covers "
            f"only that window, so per-GPU activity and coverage checks may "
            f"fail for reasons unrelated to the workload{detail}"
        )
        if not detail:
            self._monitor_err.unlink(missing_ok=True)

    def _cu_loop(self) -> None:  # noqa: C901
        """Background loop: sample ``amd-smi process`` every 5s into cu_occupancy CSV."""
        from tests.common.gpu_monitored.executor_bridge import run_command_captured

        while not self._cu_stop.is_set():
            if self._cu_stop.wait(5):
                return
            out = ""
            if self.monitor_executor is not None:
                res = run_command_captured(
                    self.monitor_executor,
                    ["timeout", "10", self._amd_smi, "process", "--json"],
                    timeout=15,
                )
                if self._cu_stop.is_set():
                    return
                if res.exit_code not in (0, 124):
                    continue
                out = res.stdout
            else:
                try:
                    proc = subprocess.Popen(
                        [self._amd_smi, "process", "--json"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                    )
                except Exception:
                    continue
                with self._cu_sample_lock:
                    self._cu_sample_proc = proc
                    if self._cu_stop.is_set():
                        with contextlib.suppress(Exception):
                            proc.terminate()
                try:
                    try:
                        out, _ = proc.communicate(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        try:
                            out, _ = proc.communicate(timeout=2)
                        except Exception:
                            out = ""
                    except Exception:
                        out = ""
                finally:
                    with self._cu_sample_lock:
                        self._cu_sample_proc = None
            if self._cu_stop.is_set():
                return
            try:
                rows = self._parse_cu_occupancy(out or "")
                if rows:
                    with self.cu_csv.open("a") as f:
                        f.write("\n".join(rows) + "\n")
            except Exception:
                pass

    @staticmethod
    def _parse_cu_occupancy(out: str) -> list[str]:
        """Parse `amd-smi process --json` output; return CSV rows for active processes."""
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, ValueError):
            return []

        ts = str(int(time.time()))
        rows: list[str] = []

        if isinstance(data, list):
            gpus = data
        elif isinstance(data, dict):
            gpus = data.get("gpu_data", data.get("gpu", []))
        else:
            gpus = []

        for g in gpus:
            gid = g.get("gpu", "?")
            procs = g.get("process_list", g.get("PROCESS_INFO", []))
            for p in procs:
                cu = p.get("cu_occupancy", p.get("CU_OCCUPANCY", 0))
                pid = p.get("pid", p.get("PID", "?"))
                vm = p.get("memory_usage", p.get("MEMORY_USAGE", {}))
                vr = vm.get("vram_mem", vm.get("VRAM_MEM", "0")) if isinstance(vm, dict) else vm
                mb = Monitor._to_mb(vr)
                try:
                    cu_f = float(cu)
                except (ValueError, TypeError):
                    cu_f = 0.0
                if cu_f > 0 or mb > 10:
                    rows.append(f"{ts},{gid},{pid},{cu},{mb:.0f}")
        return rows

    @staticmethod
    def _to_mb(raw) -> float:
        """Convert an amd-smi memory value (scalar, string with optional
        unit suffix) to MB.

        Handles ``"1024"`` (assumed MB), ``"2.5 GB"``, ``"512 MB"``, and
        ``"1073741824 B"``. Earlier code stripped the unit suffix but
        never rescaled bytes → MB, so a byte-valued reading would be
        reported as millions of MB.
        """
        if raw is None:
            return 0.0
        s = str(raw).strip()
        if not s:
            return 0.0
        # Detect suffix first — match only a trailing " GB"/" MB"/" B" so
        # substrings like "GB" inside a device name can't trigger scaling.
        # The numeric part is spelled out rather than gathered in a character
        # class: "[0-9.+-eE]" reads "+" to "e" as a range, which swallows
        # letters and punctuation instead of just a signed decimal/exponent.
        m = re.match(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(GB|MB|KB|B)?\s*$", s, flags=re.IGNORECASE)
        if m is None:
            return 0.0
        try:
            v = float(m.group(1))
        except (ValueError, TypeError):
            return 0.0
        unit = (m.group(2) or "MB").upper()
        if unit == "GB":
            return v * 1024.0
        if unit == "MB":
            return v
        if unit == "KB":
            return v / 1024.0
        if unit == "B":
            return v / (1024.0 * 1024.0)
        return v


# Bound on rows the CSV reader itself cannot decode. High enough that a burst
# of corruption mid-file still leaves the surrounding valid samples usable, low
# enough that a reader stuck on an undecodable row cannot spin.
_MAX_CORRUPT_ROWS = 100

# Longest stretch of a run allowed to carry no telemetry at all.
#
# Calibrated against 51 archived runs across MI300A and MI325X covering all
# nine tests and durations from 42 s to 1663 s: every healthy run leaves
# 0-2 s untelemetered, independent of length, because the monitor brackets
# the workload and samples once a second. The three truncated runs in that
# set measured 332 s, 426 s and 1530 s -- two of which still reported PASS,
# because per-GPU coverage was satisfied inside the window the monitor did
# manage to record.
#
# 60 s therefore sits ~30x above the worst healthy observation and ~5x below
# the mildest real truncation. Scaled down for short tests (see
# ``telemetry_budget_sec``) so a 42 s run cannot hide a 40 s blackout.
MAX_TELEMETRY_GAP_SEC = 60


def telemetry_budget_sec(duration_sec: int) -> int:
    """Untelemetered seconds tolerated for a run of ``duration_sec``.

    Never more than half the run, so the allowance stays meaningful on
    short tests, and never below 5 s, so second-level rounding on a very
    short run cannot fail it.
    """
    if duration_sec <= 0:
        return MAX_TELEMETRY_GAP_SEC
    return min(MAX_TELEMETRY_GAP_SEC, max(5, duration_sec // 2))


def align_coverage_to_csv_clock(
    coverage_start_timestamp: int,
    duration_sec: int,
    evidence: MonitoringEvidence,
) -> tuple[int, int]:
    """Map run boundaries onto the CSV timestamp domain when clocks disagree.

    ``amd-smi monitor`` CSV timestamps can diverge from Python ``time.time()``
    (observed ~1 h skew on some hosts). When the gap exceeds the telemetry
    budget, anchor coverage to the CSV span plus the monotonic run duration.
    """
    if evidence.first_timestamp is None:
        return coverage_start_timestamp, coverage_start_timestamp + duration_sec
    skew = coverage_start_timestamp - evidence.first_timestamp
    budget = telemetry_budget_sec(duration_sec)
    if abs(skew) <= budget:
        return coverage_start_timestamp, coverage_start_timestamp + duration_sec
    return evidence.first_timestamp, evidence.first_timestamp + duration_sec


def collect_monitoring_evidence(  # noqa: C901
    csv_path: Path,
    *,
    coverage_start_timestamp: int | None = None,
    coverage_end_timestamp: int | None = None,
) -> MonitoringEvidence:
    """Collect valid sample and per-GPU sensor/activity evidence."""
    evidence = MonitoringEvidence(
        coverage_start_timestamp=coverage_start_timestamp,
        coverage_end_timestamp=coverage_end_timestamp,
    )
    sample_times: dict[str, set[int]] = {}
    if not csv_path.is_file():
        return evidence
    try:
        csv_file = csv_path.open(newline="")
    except OSError:
        return evidence
    with csv_file:
        reader = csv.DictReader(csv_file)
        # A row the reader cannot decode (a field beyond field_size_limit,
        # e.g. an interleaved or truncated monitor write) used to abort the
        # whole scan and return zero
        # evidence, discarding every valid sample read before it. That reads
        # downstream as "monitoring never ran" and fails an otherwise healthy
        # run, so treat it like the malformed rows already skipped below and
        # keep what was collected. Bounded in case a row cannot be consumed.
        while True:
            if evidence.undecodable_rows > _MAX_CORRUPT_ROWS:
                # Give up on the rest of the file rather than spin, but record
                # that the tail was never read: everything after this point is
                # missing from the counts below.
                evidence.scan_aborted = True
                break
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error:
                evidence.undecodable_rows += 1
                continue
            except OSError:
                break
            try:
                ts = int(str(row.get(csv_schema.TIMESTAMP, "")).strip())
            except (TypeError, ValueError):
                evidence.unparsable_rows += 1
                continue
            gpu = str(row.get(csv_schema.GPU) or "").strip()
            if not gpu:
                evidence.unparsable_rows += 1
                continue
            if _has_finite_telemetry(row):
                evidence.sample_count += 1
                evidence.sampled_gpus.add(gpu)
                sample_times.setdefault(gpu, set()).add(ts)
            if _finite_field(row, csv_schema.POWER_USAGE) is not None:
                evidence.power_gpus.add(gpu)
            if _finite_field(row, csv_schema.HOTSPOT_TEMP) is not None:
                evidence.hotspot_gpus.add(gpu)
            gfx_util = _finite_field(row, csv_schema.GFX_UTIL)
            if gfx_util is not None and gfx_util > 0:
                evidence.active_sample_count += 1
                evidence.active_gpus.add(gpu)
    if sample_times:
        every = sorted({t for times in sample_times.values() for t in times})
        evidence.first_timestamp = every[0]
        evidence.last_timestamp = every[-1]
        # Per GPU, because the fleet-wide view cannot distinguish "all GPUs
        # covered throughout" from "one GPU covered throughout while the
        # others went dark".
        for gpu, times in sorted(sample_times.items()):
            ordered = sorted(times)
            gap = max(
                (b - a for a, b in itertools.pairwise(ordered)),
                default=0,
            )
            evidence.gpu_windows[gpu] = (ordered[-1] - ordered[0], gap)
            evidence.gpu_sample_bounds[gpu] = (ordered[0], ordered[-1])
            evidence.max_sample_gap_sec = max(evidence.max_sample_gap_sec, gap)
    return evidence


def count_csv_samples(csv_path: Path) -> int:
    """Count parseable amd-smi telemetry rows."""
    return collect_monitoring_evidence(csv_path).sample_count


def count_active_gpu_samples(csv_path: Path) -> int:
    """Count valid samples whose GFX utilization is finite and positive."""
    return collect_monitoring_evidence(csv_path).active_sample_count


def count_sampled_gpus(csv_path: Path) -> int:
    """Count distinct GPU identifiers with a valid timestamped CSV row."""
    return len(collect_monitoring_evidence(csv_path).sampled_gpus)
