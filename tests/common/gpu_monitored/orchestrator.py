# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MonitoredTestOrchestrator: orchestrates a single test run (build is done separately)."""

from __future__ import annotations

import csv
import json
import math
import os
import re
import select
import socket
import sys
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor
    from framework.executors.executor_group import NodeExecutorGroup

from tests.common.gpu_monitored import analyze, csv_schema, report
from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.dmesg_capture import (
    DMESG_SNAPSHOT_UNAVAILABLE,
    capture_dmesg_text,
)
from tests.common.gpu_monitored.monitoring import (
    _MAX_CORRUPT_ROWS,
    Monitor,
    align_coverage_to_csv_clock,
    collect_monitoring_evidence,
    telemetry_budget_sec,
)
from tests.common.gpu_monitored.workloads.base import BuildStatus, RunContext, RunResult, Test
from tests.common.gpu_monitored.validation import ValidationResult, unsupported_reason_from_log, validate_result


@dataclass
class TestOutcome:
    test_name: str
    status: str                 # PASS / FAIL / UNSUPPORTED / BUILD_FAILED / SKIP
    exit_code: Optional[int]
    elapsed_seconds: int
    run_dir: Optional[Path]
    validation: str


class TerminationRequested(SystemExit):
    """Signal-triggered cancellation that must escape workload handling."""


class MonitoredTestOrchestrator:
    """Runs one or more tests, collects outcomes."""

    OUTPUT_PUMP_DRAIN_TIMEOUT_SEC = 10
    # How many consecutive pretest lines the dmesg delta will compare when
    # locating its anchor. Bounds the search on adversarial buffers; see
    # ``_write_dmesg_delta``.
    DMESG_ANCHOR_MAX_DEPTH = 256
    OUTPUT_PUMP_POLL_SEC = 0.1

    def __init__(
        self,
        config: Config,
        *,
        target_executor: NodeExecutorGroup | None = None,
        monitor_executor: AbstractExecutor | None = None,
    ):
        self.config = config
        self.target_executor = target_executor
        self.monitor_executor = monitor_executor

    def run_one(self, test: Test) -> TestOutcome:
        """Orchestrate a single test: monitor + workload + validate + analyze + report."""
        name = test.spec.name
        goal = test.spec.goal

        run_dir = self.config.log_root / name
        run_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        start_iso = datetime.now().isoformat(timespec="seconds")

        # Write command.txt header
        cmd_txt_lines = [
            f"test_name: {name}",
            f"goal: {goal}",
            f"workload_function: {type(test).__name__}",
            f"timestamp: {ts}",
            f"start_time: {start_iso}",
            f"hostname: {socket.gethostname()}",
            f"sample_interval: {self.config.sample_interval}s",
            f"rocm_version: {self.config.rocm_version}",
            f"rocm_root: {self.config.rocm_root}",
            f"gpu_arch: {self.config.gpu_arch or 'unknown'}",
            f"gpu_model: {self.config.gpu_model or 'unknown'} ({self.config.gpu_short_name or 'unknown'})",
            f"num_gpus: {self.config.num_gpus}",
            f"compiler: {self.config.clangxx}",
        ]
        (run_dir / "command.txt").write_text("\n".join(cmd_txt_lines) + "\n")

        # Pre-test dmesg snapshot
        self._capture_dmesg(run_dir / "dmesg_pretest.log")

        # Build RunContext
        ctx = RunContext(
            config=self.config,
            run_dir=run_dir,
            log_root=self.config.log_root,
            target_executor=self.target_executor,
            monitor_executor=self.monitor_executor,
            console_log=run_dir / "console.log",
        )

        # Run workload with monitoring. Use ``time.monotonic`` so an NTP
        # step during a long run can never produce a negative duration
        # or an effectively-infinite monitoring window.
        start_s = time.monotonic()
        coverage_start_timestamp = int(time.time())
        result = RunResult(exit_code=1)
        with Monitor(
            csv_file=run_dir / "power_temp.csv",
            cu_csv=run_dir / "cu_occupancy.csv",
            sample_interval=self.config.sample_interval,
            enable_cu_occupancy=self.config.enable_cu_occupancy,
            monitor_executor=self.monitor_executor,
            rocm_root=self.config.rocm_root,
        ):
            print(f"  [{name}] Running: {type(test).__name__}")
            result = self._run_workload(test, ctx, run_dir)
        duration = int(time.monotonic() - start_s)
        # No drain sleep here: ``Monitor.__exit__`` already runs
        # SIGTERM (5s wait) -> SIGKILL (5s wait) on amd-smi, so the
        # monitor child is reaped and the CSV's file descriptor is
        # closed by the OS before we get back to this line. The old
        # ``time.sleep(2)  # Let monitoring drain`` was a no-op that
        # added 20s of dead time per full suite run.

        # Post-test dmesg delta
        self._write_dmesg_delta(run_dir / "dmesg_pretest.log", run_dir / "dmesg.log")

        # An UNSUPPORTED test never ran its workload, so skip only Layer 2's
        # test-specific output contract. Generic crash and dmesg layers still
        # apply to the run window: a GPU reset during preflight must not be
        # hidden behind an environment classification.
        val = validate_result(
            test_name=name,
            log_file=run_dir / "console.log",
            exit_code=result.exit_code,
            dmesg_file=run_dir / "dmesg.log",
            skip_test_specific=result.unsupported,
            num_gpus=self.config.num_gpus,
        )
        if result.unsupported and not val.failed:
            val = ValidationResult(
                message=unsupported_reason_from_log(run_dir / "console.log", name),
                failed=False,
            )
        # If process exited 0 but validation failed, promote exit code to 1
        if result.exit_code == 0 and val.failed:
            result.exit_code = 1
        effective_unsupported = result.unsupported and not val.failed

        # Derive the wall-clock end from monotonic elapsed time instead of a
        # second wall-clock read. This keeps the edge comparison on the CSV's
        # epoch scale without letting an NTP step alter the workload duration.
        coverage_end_timestamp = coverage_start_timestamp + duration
        csv_path = run_dir / "power_temp.csv"
        prelim = collect_monitoring_evidence(csv_path)
        coverage_start_timestamp, coverage_end_timestamp = align_coverage_to_csv_clock(
            coverage_start_timestamp,
            duration,
            prelim,
        )
        monitoring = collect_monitoring_evidence(
            csv_path,
            coverage_start_timestamp=coverage_start_timestamp,
            coverage_end_timestamp=coverage_end_timestamp,
        )
        monitor_lines = monitoring.sample_count
        if monitoring.dropped_rows:
            # Dropping unusable rows keeps a garbled write from failing a
            # healthy run, but the sample count then under-reports the run
            # with no visible cause. Name it so a low count can be attributed
            # to corrupt telemetry rather than to an idle GPU. Not a verdict
            # of its own: the coverage and activity gates below still decide.
            print(f"  [monitor] WARNING: dropped {monitoring.dropped_rows} "
                  f"unusable row(s) from power_temp.csv "
                  f"({monitoring.undecodable_rows} undecodable, "
                  f"{monitoring.unparsable_rows} without a usable "
                  f"timestamp/GPU); {monitor_lines} sample(s) kept")
        if monitoring.scan_aborted:
            print(f"  [monitor] WARNING: stopped scanning power_temp.csv after "
                  f"more than {_MAX_CORRUPT_ROWS} undecodable rows — samples "
                  f"after that point are not counted, so coverage below may "
                  f"be incomplete for reasons unrelated to the workload")
        if not effective_unsupported and monitor_lines == 0:
            monitor_note = (
                "monitoring produced zero samples; cannot validate power/"
                "thermal behavior"
            )
            val = ValidationResult(
                message=(f"{val.message}; {monitor_note}"
                         if val.message else monitor_note),
                failed=True,
            )
            if result.exit_code == 0:
                result.exit_code = 1
        # Telemetry has to cover the run, not just name every GPU. The
        # per-GPU checks below are satisfied by any window in which all GPUs
        # appear, so a monitor that stops early -- or wedges while staying
        # alive, which the exit-based monitor-death checks cannot see -- left
        # a run reporting PASS on a fraction of its power/thermal history.
        # Observed twice: 122 s of a 1652 s power_band run and 686 s of a
        # 1018 s rvs_tst run, both PASS.
        blind_sec, worst_gpu = monitoring.worst_coverage(duration)
        telemetry_budget = telemetry_budget_sec(duration)
        if (not effective_unsupported
                and monitor_lines > 0
                and blind_sec > telemetry_budget):
            # Name the GPU responsible: the measurement is per-GPU, so a
            # partial monitor death points at specific devices rather than at
            # the run as a whole.
            blind_note = (
                f"monitoring did not continuously cover the {duration}s run"
                + (f" for GPU {worst_gpu}" if worst_gpu is not None else "")
                + f" (longest stretch without telemetry {blind_sec}s > "
                  f"{telemetry_budget}s budget); power/thermal evidence does "
                  f"not span the workload"
            )
            val = ValidationResult(
                message=(f"{val.message}; {blind_note}"
                         if val.message else blind_note),
                failed=True,
            )
            if result.exit_code == 0:
                result.exit_code = 1
        expected_gpus = {str(gpu) for gpu in range(self.config.num_gpus)}
        if (not effective_unsupported
                and monitoring.sampled_gpus != expected_gpus):
            coverage_note = (
                "monitoring GPU identity mismatch "
                f"(observed {sorted(monitoring.sampled_gpus)}, "
                f"expected {sorted(expected_gpus)})"
            )
            val = ValidationResult(
                message=(f"{val.message}; {coverage_note}"
                         if val.message else coverage_note),
                failed=True,
            )
            if result.exit_code == 0:
                result.exit_code = 1
        if (not effective_unsupported
                and (monitoring.power_gpus != expected_gpus
                     or monitoring.hotspot_gpus != expected_gpus)):
            sensor_note = (
                "monitoring sensor coverage incomplete "
                f"(power GPUs {sorted(monitoring.power_gpus)}, hotspot GPUs "
                f"{sorted(monitoring.hotspot_gpus)}, expected "
                f"{sorted(expected_gpus)})"
            )
            val = ValidationResult(
                message=(f"{val.message}; {sensor_note}"
                         if val.message else sensor_note),
                failed=True,
            )
            if result.exit_code == 0:
                result.exit_code = 1
        profile = test.spec.workload_profile or {}
        if (not effective_unsupported
                and profile.get("min_util", 0) > 0
                and monitoring.active_gpus != expected_gpus):
            activity_note = (
                "monitoring GPU activity incomplete "
                f"(active GPUs {sorted(monitoring.active_gpus)}, expected "
                f"{sorted(expected_gpus)})"
            )
            val = ValidationResult(
                message=(f"{val.message}; {activity_note}"
                         if val.message else activity_note),
                failed=True,
            )
            if result.exit_code == 0:
                result.exit_code = 1

        # Initial summary.json (matches shell's inline python -c block)
        summary_path = run_dir / "summary.json"
        self._write_initial_summary(
            summary_path,
            name=name,
            ts=ts,
            exit_code=result.exit_code,
            duration=duration,
            monitor_samples=monitor_lines,
            validation=val.message or "N/A",
            unsupported=effective_unsupported,
        )

        # Enhanced analysis (enriches summary.json, writes health_checks.txt)
        artifact_errors = []
        if not self._safe_analyze(run_dir, name):
            artifact_errors.append("analysis")
            artifact_note = "analysis artifact generation failed"
            val = ValidationResult(
                message=(f"{val.message}; {artifact_note}"
                         if val.message else artifact_note),
                failed=True,
            )
            result.exit_code = 1
            effective_unsupported = False
            self._update_summary_status(
                summary_path,
                result.exit_code,
                val.message,
                effective_unsupported,
                artifact_errors,
            )

        # HTML report
        if not self._safe_report(
            run_dir, name, result.exit_code, duration,
            unsupported=effective_unsupported,
        ):
            artifact_errors.append("report")
            artifact_note = "HTML report generation failed"
            val = ValidationResult(
                message=(f"{val.message}; {artifact_note}"
                         if val.message else artifact_note),
                failed=True,
            )
            result.exit_code = 1
            effective_unsupported = False
            self._update_summary_status(
                summary_path,
                result.exit_code,
                val.message,
                effective_unsupported,
                artifact_errors,
            )

        # Append final status only after artifact generation can vote.
        with (run_dir / "command.txt").open("a") as f:
            f.write(f"end_time: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"duration_seconds: {duration}\n")
            f.write(f"exit_code: {result.exit_code}\n")
            f.write(f"validation: {val.message or 'N/A'}\n")
            if result.reproduce_cmd:
                f.write("\n# Reproduce (test only, no monitoring):\n")
                f.write(result.reproduce_cmd + "\n")

        # Print per-test summary block
        self._print_test_summary(
            name=name,
            exit_code=result.exit_code,
            duration=duration,
            monitor_lines=monitor_lines,
            validation=val.message or "N/A",
            run_dir=run_dir,
            unsupported=effective_unsupported,
        )

        # Determine status
        if effective_unsupported:
            status = "UNSUPPORTED"
        elif result.exit_code == 0:
            status = "PASS"
        else:
            status = "FAIL"

        return TestOutcome(
            test_name=name,
            status=status,
            exit_code=result.exit_code,
            elapsed_seconds=duration,
            run_dir=run_dir,
            validation=val.message or "N/A",
        )

    # -------------------------------------------------------------------
    # Workload execution with stdout tee to console.log
    # -------------------------------------------------------------------
    def _run_workload(self, test: Test, ctx: RunContext, run_dir: Path) -> RunResult:
        """Invoke test.run(ctx), capturing all stdout/stderr to console.log.

        Workload fd 1/2 are redirected to a pipe whose reader thread writes
        ``console.log``. The live console shows only the per-test summary block
        printed afterwards. Subprocesses inherit fd 1/2, so redirection happens
        at the fd level rather than swapping ``sys.stdout``.
        """
        log_path = run_dir / "console.log"

        saved_stdout_fd: Optional[int] = None
        saved_stderr_fd: Optional[int] = None
        saved_sys_stdout = sys.stdout
        saved_sys_stderr = sys.stderr
        pipe_r: Optional[int] = None
        pipe_w: Optional[int] = None
        log_fh = None
        pump: Optional[threading.Thread] = None
        pump_stop = threading.Event()
        log_failed = threading.Event()
        result = RunResult(exit_code=1)

        try:
            # Flush anything buffered on the OLD fds before redirecting
            sys.stdout.flush()
            sys.stderr.flush()

            saved_stdout_fd = os.dup(1)
            saved_stderr_fd = os.dup(2)

            pipe_r, pipe_w = os.pipe()
            log_fh = open(log_path, "wb")

            def _pump(rfd=pipe_r, fh=log_fh):
                """Drain the capture pipe until EOF or cancellation."""
                try:
                    while not pump_stop.is_set():
                        try:
                            readable, _, _ = select.select(
                                [rfd], [], [], self.OUTPUT_PUMP_POLL_SEC,
                            )
                        except (OSError, ValueError):
                            log_failed.set()
                            break
                        if not readable:
                            continue
                        try:
                            chunk = os.read(rfd, 65536)
                        except OSError:
                            log_failed.set()
                            break
                        if not chunk:
                            break
                        try:
                            fh.write(chunk)
                            fh.flush()
                        except (OSError, ValueError):
                            log_failed.set()
                finally:
                    try:
                        os.close(rfd)
                    except OSError:
                        pass
                    try:
                        fh.close()
                    except (OSError, ValueError):
                        log_failed.set()

            pump = threading.Thread(target=_pump, daemon=True)
            pump.start()
            os.dup2(pipe_w, 1)
            os.dup2(pipe_w, 2)
            # Only fd 1/2 and any descendants now hold the write end.
            os.close(pipe_w)
            pipe_w = None

            # Pytest replaces sys.stdout with a capture object that does not
            # write to fd 1.  Subprocesses inherit fd 1 (the pipe above) but
            # Python ``print()`` would bypass console.log unless we rebind the
            # text streams too.
            sys.stdout = os.fdopen(1, "w", buffering=1, closefd=False)
            sys.stderr = os.fdopen(2, "w", buffering=1, closefd=False)

            try:
                result = test.run(ctx)
            except TerminationRequested:
                raise
            except SystemExit as e:
                # ``sys.exit()`` with no argument has ``e.code is None``,
                # which Python's process-exit machinery maps to status 0.
                # The earlier ``e.code if isinstance(e.code, int) else 1``
                # mistakenly mapped ``None`` to 1, reporting a successful
                # ``sys.exit()`` as FAIL. A *string* code (e.g.
                # ``sys.exit("error message")``) still maps to 1, matching
                # CPython's behaviour.
                if e.code is None:
                    code = 0
                elif isinstance(e.code, int):
                    code = e.code
                else:
                    code = 1
                result = RunResult(exit_code=code)
            except Exception:
                traceback.print_exc()
                result = RunResult(exit_code=1)

            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            try:
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sys.stderr.flush()
            except Exception:
                pass
            sys.stdout = saved_sys_stdout
            sys.stderr = saved_sys_stderr
            if saved_stdout_fd is not None:
                try:
                    os.dup2(saved_stdout_fd, 1)
                except OSError:
                    pass
            if saved_stderr_fd is not None:
                try:
                    os.dup2(saved_stderr_fd, 2)
                except OSError:
                    pass
            # Setup can fail after ``os.pipe()`` but before the normal close
            # (for example, opening console.log on a full filesystem). Close
            # the write end before joining so a started pump can observe EOF.
            if pipe_w is not None:
                try:
                    os.close(pipe_w)
                except OSError:
                    pass
                pipe_w = None
            if pump is not None:
                pump.join(timeout=self.OUTPUT_PUMP_DRAIN_TIMEOUT_SEC)
                if pump.is_alive():
                    pump_stop.set()
                    pump.join(timeout=max(1.0, self.OUTPUT_PUMP_POLL_SEC * 2))
                    print("  [runner] WARNING: output pump did not finish "
                          f"within {self.OUTPUT_PUMP_DRAIN_TIMEOUT_SEC}s; "
                          "console.log may be truncated")
                    # A cancelled drain abandons whatever is still in the
                    # pipe, so the authoritative log may be missing its tail.
                    log_failed.set()
            if log_failed.is_set() and result.exit_code == 0:
                result.exit_code = 1
            # ``pipe_r`` and ``log_fh`` belong to the pump once it starts.
            # If the pump is stuck in a hostile filesystem write, leave those
            # unique descriptors owned by the daemon rather than closing them
            # underneath it and risking descriptor-reuse corruption.
            for fd in (saved_stdout_fd, saved_stderr_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
            if pump is None:
                for fd in (pipe_r,):
                    if fd is not None:
                        try:
                            os.close(fd)
                        except OSError:
                            pass
                if log_fh is not None:
                    try:
                        log_fh.close()
                    except (OSError, ValueError):
                        pass

        return result

    # -------------------------------------------------------------------
    # dmesg capture
    # -------------------------------------------------------------------
    def _capture_dmesg(self, out_path: Path) -> None:
        # Some kernels emit a permission warning to *stderr* and exit
        # non-zero even when stdout is a complete dmesg dump. Treating
        # that as "capture failed" used to leave the pretest snapshot
        # empty, which then forced ``_write_dmesg_delta`` to print
        # the eviction banner and dump the full post-test ring buffer
        # — making real per-test deltas harder to read. Accept the
        # output whenever we got something non-empty back, even if rc
        # was non-zero. The next attempt (privileged) only runs when
        # the unprivileged path returned both rc!=0 *and* empty
        # stdout, which is the genuine "no permission" case.
        available, output = capture_dmesg_text(self.monitor_executor)
        # File emptiness is valid evidence when dmesg successfully reads an
        # empty/rotated ring. Use an explicit sentinel only for capture
        # failure so the delta step can distinguish those two states.
        out_path.write_text(
            output if available else DMESG_SNAPSHOT_UNAVAILABLE
        )

    def _write_dmesg_delta(self, pretest: Path, delta_out: Path) -> None:
        """Capture dmesg after test, diff against pretest snapshot.

        Uses ``splitlines()`` on both sides so trailing-newline differences
        can't produce an off-by-one. Also anchors on the last line of the
        pretest snapshot when it is still present in the post-test buffer;
        this makes the delta correct even if the kernel ring buffer grew
        past the pretest offset.
        """
        if not pretest.is_file():
            delta_out.write_text(
                "# [dmesg-delta] pretest snapshot unavailable; cannot "
                "validate kernel health for this workload\n"
            )
            return
        pretest_text = pretest.read_text()
        if pretest_text == DMESG_SNAPSHOT_UNAVAILABLE:
            delta_out.write_text(
                "# [dmesg-delta] pretest snapshot unavailable; cannot "
                "validate kernel health for this workload\n"
            )
            return
        pre_lines = pretest_text.splitlines()
        # Mirror ``_capture_dmesg``'s policy: accept ``stdout`` whenever
        # it's non-empty, even if rc was non-zero. Some kernels emit a
        # permission warning to stderr and exit non-zero with a complete
        # ring-buffer dump on stdout. Discarding that output used to
        # leave the post-test snapshot empty, the delta would collapse,
        # and Layer 3 dmesg gating would silently miss critical events
        # that landed during the test window.
        post_available, post = capture_dmesg_text(self.monitor_executor)
        if not post_available:
            delta_out.write_text(
                "# [dmesg-delta] posttest snapshot unavailable; cannot "
                "validate kernel health for this workload\n"
            )
            return

        post_lines = post.splitlines()
        delta: List[str] = []
        if pre_lines:
            # Find the earliest post position whose prefix ends with the
            # longest available suffix of the pretest snapshot. Choosing the
            # last byte-identical anchor can skip a new event when the same
            # dmesg line repeats later in the test window.
            # Any matching suffix, of any length, must end with the pretest
            # snapshot's last line -- so only the positions where that line
            # occurs can be anchors. Extending backwards from each of those
            # gives the same answer (longest suffix, earliest end among ties)
            # as scanning every suffix length against every position, without
            # the cost.
            #
            # That cost was not theoretical. The previous form tried each
            # suffix length from len(pre_lines) down to 1 and rescanned the
            # whole post list for each, slicing both sides to compare, so it
            # was O(pre x post x suffix_len). It exited immediately while the
            # pretest tail was still in the ring buffer, which is the common
            # case, and degenerated only when the anchor had been evicted --
            # exactly when a workload is verbose enough to overwrite it. A
            # power-band run emitting ~12,700 messages against a 12,764-line
            # pretest snapshot took long enough to look like a hang, and the
            # eviction branch below could not be reached to report it.
            matched_at: Optional[int] = None
            max_suffix = min(len(pre_lines), len(post_lines))
            anchor = pre_lines[-1]
            best_len = 0
            # Depth cap so no buffer can make this expensive. Without it the
            # cost is O(candidates x depth), and a buffer whose lines are all
            # identical -- plausible because ``dmesg -T`` stamps only to the
            # second, so a message repeating within one second is byte-identical
            # -- makes every position a candidate extending the full length:
            # measured at 22 s for ~12.7k lines, and quadratic beyond that.
            # Agreement over this many consecutive kernel lines already
            # identifies the anchor beyond doubt; if the cap ever does bite, it
            # can only choose an earlier position, which over-reports the delta
            # rather than fabricating an offset -- the same trade the
            # evicted-anchor branch below already makes.
            max_depth = min(max_suffix, self.DMESG_ANCHOR_MAX_DEPTH)
            for end in range(1, len(post_lines) + 1):
                if post_lines[end - 1] != anchor:
                    continue
                # How far back this candidate agrees with the pretest tail.
                limit = min(end, max_depth)
                run = 0
                while (run < limit
                       and post_lines[end - 1 - run] == pre_lines[-1 - run]):
                    run += 1
                if run > best_len:
                    best_len = run
                    matched_at = end
                    if best_len == max_depth:
                        break   # good enough; earliest such end wins
            if matched_at is not None:
                delta = post_lines[matched_at:]
            else:
                # Pretest anchor was evicted from the ring buffer by
                # events during the test. A length-based offset
                # (``post_lines[len(pre_lines):]``) would splice in the
                # middle of the new content and report a random suffix,
                # so we instead log *all* of post_lines with a clear
                # banner — the caller can still see kernel events, just
                # not filtered to "new since pretest". Over-reporting is
                # preferable to silently fabricating an offset.
                banner = (
                    "# [dmesg-delta] pretest anchor evicted from ring "
                    "buffer; showing full post-test dmesg (may include "
                    "pre-existing entries)"
                )
                delta = [banner] + post_lines
        else:
            # A successfully captured empty pretest ring is a valid baseline:
            # every posttest line appeared during this workload. This differs
            # from an unavailable capture, represented by the sentinel above.
            delta = post_lines
        delta_out.write_text("\n".join(delta) + ("\n" if delta else ""))

    # -------------------------------------------------------------------
    # Summary + analyze + report
    # -------------------------------------------------------------------
    def _write_initial_summary(self, path: Path, *, name: str, ts: str, exit_code: int,
                               duration: int, monitor_samples: int, validation: str,
                               unsupported: bool = False) -> None:
        """Write the base summary.json that analyze.enrich_summary will extend."""
        csv_file = path.parent / "power_temp.csv"
        metrics = {}
        if csv_file.is_file():
            metrics = {
                "power": _csv_stats(csv_file, csv_schema.POWER_USAGE),
                "hotspot_temp": _csv_stats(csv_file, csv_schema.HOTSPOT_TEMP),
                "gfx_clk": _csv_stats(csv_file, csv_schema.GFX_CLK),
                "gfx_util": _csv_stats(csv_file, csv_schema.GFX_UTIL),
                "vram_pct": _csv_stats(csv_file, csv_schema.VRAM_PCT),
            }
        if unsupported:
            result_str = "UNSUPPORTED"
        elif exit_code == 0:
            result_str = "PASS"
        else:
            result_str = "FAIL"
        data = {
            "test_name": name,
            "timestamp": ts,
            "result": result_str,
            "exit_code": exit_code,
            "unsupported": unsupported,
            "duration_seconds": duration,
            "monitor_samples": monitor_samples,
            "validation": validation,
            "metrics": metrics,
        }
        # Propagate pre-test health probe annotation, if any. Only
        # emit the keys when dirty so passing-clean runs keep the
        # exact summary.json shape they had before this feature.
        if self.config.pretest_kernel_dirty:
            data["pretest_kernel_dirty"] = True
            data["inherited_critical_categories"] = list(
                self.config.inherited_critical_categories
            )
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w") as f:
            json.dump(data, f, indent=2)
        os.replace(temp_path, path)

    @staticmethod
    def _update_summary_status(
        path: Path,
        exit_code: int,
        validation: str,
        unsupported: bool,
        artifact_errors: List[str],
    ) -> None:
        data = json.loads(path.read_text())
        data["result"] = "UNSUPPORTED" if unsupported else (
            "PASS" if exit_code == 0 else "FAIL"
        )
        data["exit_code"] = exit_code
        data["unsupported"] = unsupported
        data["validation"] = validation
        data["artifacts_complete"] = False
        data["artifact_errors"] = list(artifact_errors)
        temp_path = path.with_name(f"{path.name}.tmp")
        with temp_path.open("w") as summary_file:
            json.dump(data, summary_file, indent=2)
        os.replace(temp_path, path)

    def _safe_analyze(self, run_dir: Path, name: str) -> bool:
        try:
            analyze.analyze_and_write(run_dir, name)
            return True
        except Exception as e:
            try:
                (run_dir / "analysis.stderr.log").write_text(
                    f"analysis error: {e}\n{traceback.format_exc()}"
                )
            except OSError:
                pass
            return False

    def _safe_report(self, run_dir: Path, name: str, exit_code: int, duration: int,
                     *, unsupported: bool = False) -> bool:
        try:
            report.write_report(run_dir, name, exit_code, duration,
                                unsupported=unsupported)
            return True
        except Exception as e:
            try:
                (run_dir / "report.stderr.log").write_text(
                    f"report error: {e}\n{traceback.format_exc()}"
                )
            except OSError:
                pass
            return False

    def _print_test_summary(self, *, name: str, exit_code: int, duration: int,
                            monitor_lines: int, validation: str, run_dir: Path,
                            unsupported: bool = False) -> None:
        """Mirror the shell's per-test text summary block."""
        if unsupported:
            verdict = "UNSUPPORTED"
        elif exit_code == 0:
            verdict = "PASS"
        else:
            verdict = f"FAIL (exit {exit_code})"

        mins = duration // 60
        secs = duration % 60

        print("============================================================")
        print(f"  {name} — {verdict}")
        print("============================================================")
        print(f"  Duration: {duration}s  Samples: {monitor_lines}")
        print(f"  Validation: {validation}")
        print("")

        csv_file = run_dir / "power_temp.csv"
        if csv_file.is_file() and monitor_lines > 0:
            metric_map = [
                ("Power(W)", csv_schema.POWER_USAGE),
                ("Temp(C)", csv_schema.HOTSPOT_TEMP),
                ("GFXClk(MHz)", csv_schema.GFX_CLK),
                ("GFXUtil(%)", csv_schema.GFX_UTIL),
                ("VRAM(%)", csv_schema.VRAM_PCT),
            ]
            print(f"  {'Metric':<15}{'Min':>8}{'Max':>8}{'Avg':>8}")
            print(f"  {'-'*15}{'-'*8}{'-'*8}{'-'*8}")
            for label, col in metric_map:
                s = _csv_stats(csv_file, col)
                if s:
                    print(f"  {label:<15}{s['min']:>8.0f}{s['max']:>8.0f}{s['avg']:>8.1f}")

        health = run_dir / "health_checks.txt"
        if health.is_file():
            print("")
            print(health.read_text().rstrip())
        print("============================================================")

        # Per-test line summary (shell prints this after `run_test` returns).
        # Suppressed when a single test was selected: the run then closes with
        # ``summary.print_and_write_summary``'s compact block, which carries
        # exactly these fields, so printing them here only duplicates them. A
        # matrix run keeps the footer -- there it terminates each test's
        # section and the fields are not repeated until the final table.
        if self.config.single_test_run:
            print("")
            return

        if verdict == "PASS":
            status = "PASS"
        elif verdict == "UNSUPPORTED":
            status = "UNSUPPORTED"
        else:
            status = "FAIL"
        print(f"  Result: {status} (exit={exit_code}, {mins}m {secs}s)")
        print(f"  Validation: {validation}")
        print(f"  Dir:    {run_dir}")
        print("")


def _csv_stats(csv_file: Path, column: str) -> Optional[dict]:
    """Min/max/avg/samples for a numeric CSV column.

    Skips empty / ``N/A`` / ``NaN`` cells. ``float("nan")`` parses
    successfully, and since ``min``/``max`` of a list containing NaN
    returns an implementation-defined result (and NaN poisons any later
    mean), every transient-out sample from amd-smi would otherwise leak
    into ``summary.json`` as a ``NaN`` and break downstream dashboards.
    """
    if not csv_file.is_file():
        return None
    values: List[float] = []
    try:
        with csv_file.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                v = row.get(column)
                if v is None:
                    continue
                s = v.strip()
                if not s or s.upper() in ("N/A", "NA", "NAN"):
                    continue
                try:
                    fv = float(s)
                except (ValueError, TypeError):
                    continue
                if math.isnan(fv) or math.isinf(fv):
                    continue
                values.append(fv)
    except Exception:
        return None
    if not values:
        return None
    # Round min/max/avg to 1 decimal to match ``analyze_monitoring._stats``
    # — the two pipelines write overlapping fields into summary.json and
    # dashboards joining on min/max between them previously couldn't
    # trust equality because only ``avg`` was rounded here.
    return {
        "min": round(min(values), 1),
        "max": round(max(values), 1),
        "avg": round(sum(values) / len(values), 1),
        "samples": len(values),
    }
