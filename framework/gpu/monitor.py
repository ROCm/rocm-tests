# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
monitor.py -- GPU metric snapshots and continuous background monitoring.

Provides two independent features:

``GpuMonitor``
    Point-in-time snapshot of GPU state, used by ``--gpu-health-metrics`` to
    capture a reading **before** and **after** each test.  Callers pass an
    already-built ``AbstractExecutor`` so the same class works for local
    (``CpuExecutor``) and remote (``SshExecutor``) nodes without modification.

``GpuBackgroundMonitor``
    Continuous background poller that runs in a daemon thread while a test
    executes.  Activated by ``--monitor-gpu``.  Samples every
    ``monitor_interval_secs`` seconds (from ``rocm-test.toml``) and writes
    timestamped rows to ``<artifact_dir>/executor-logs/<test>_gpu_monitor.log``.
    Stops immediately when the test's fixture ``finally`` block calls
    ``stop()`` — irrespective of any configured duration cap.

Executor selection (enforced by the fixture, not this module):
    Local node  → ``CpuExecutor``  (local subprocess, **no** ROCR injection)
    Remote node → ``SshExecutor``  (existing SSH session, **no** ROCR injection)
    ``LocalExecutor`` is intentionally NOT used — it injects
    ``ROCR_VISIBLE_DEVICES`` which restricts ``amd-smi``'s device view.

Health-snapshot output format (``[pre-health]`` / ``[post-health]``)::

    [pre-health] GPU-0  temp=47C  vram=1024/32768MB(3%)  util=0%  ECC=0  clk=auto
    [post-health] GPU-0  temp=72C  vram=8192/32768MB(25%)  util=85%  ECC=0  clk=auto
    [health-delta]  GPU-0  temp=+25C  vram=+7168MB  util=peak≈85%

Background monitor log format (``<test>_gpu_monitor.log``)::

    # GPU continuous monitor — test: test_llvm_mem_intrinsic_stress
    # GPUs: [0]  interval: 5.0s  duration: test-bounded
    # Cols: timestamp | GPU-N | temp=...  vram=...  util=...  ECC=...  clk=...
    2026-05-11 15:40:01 | GPU-0 | temp=47C  vram=1024/32768MB(3%)  util=0%  ECC=0  clk=auto
    2026-05-11 15:40:06 | GPU-0 | temp=52C  vram=8192/32768MB(25%)  util=85%  ECC=0  clk=auto
    # Monitor stopped (test completed)
"""

from __future__ import annotations

from dataclasses import dataclass
import datetime
import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

_ALL_METRICS: frozenset[str] = frozenset({"temp", "vram", "util", "ecc", "clock"})


# ---------------------------------------------------------------------------
# GpuMetrics — snapshot data class
# ---------------------------------------------------------------------------


@dataclass
class GpuMetrics:
    """Snapshot of GPU state captured via ``amd-smi metric``.

    All fields except ``gpu_index`` are ``None`` when the metric could not be
    read (e.g., ``amd-smi`` not on PATH, GPU not accessible).

    Attributes:
        gpu_index:     Zero-based GPU ordinal.
        temp_c:        Hot-spot temperature in Celsius.
        vram_used_mb:  VRAM currently used in MB.
        vram_free_mb:  VRAM currently free in MB.
        vram_total_mb: Total VRAM in MB.
        compute_pct:   GFX compute utilization percentage (0-100).
        ecc_errors:    Total correctable ECC error count.
        clock_state:   Performance level string (e.g. ``"auto"``, ``"high"``).
    """

    gpu_index: int
    temp_c: int | None = None
    vram_used_mb: int | None = None
    vram_free_mb: int | None = None
    vram_total_mb: int | None = None
    compute_pct: int | None = None
    ecc_errors: int | None = None
    clock_state: str | None = None


# ---------------------------------------------------------------------------
# GpuMonitor — point-in-time snapshot (--gpu-health-metrics)
# ---------------------------------------------------------------------------


class GpuMonitor:
    """Capture a single GPU metric snapshot and format it for console / log output.

    Used by ``target_executor`` and multi-GPU fixtures to emit ``[pre-health]`` /
    ``[post-health]`` / ``[health-delta]`` lines when ``--gpu-health-metrics``
    is active.

    The caller is responsible for supplying the right ``executor``:
        - ``CpuExecutor`` for local nodes (no ROCR env injection)
        - ``SshExecutor`` for remote nodes (reuses the pool's existing session)

    Args:
        executor: Pre-built executor that can run ``amd-smi metric --gpu N --json``.
        metrics:  Set of metric names to collect.  Valid: ``"temp"``, ``"vram"``,
                  ``"util"``, ``"ecc"``, ``"clock"``.  ``None`` = all.
    """

    def __init__(
        self,
        executor: AbstractExecutor,
        metrics: set[str] | None = None,
    ) -> None:
        self._executor = executor
        self._metrics: frozenset[str] = frozenset(metrics) if metrics is not None else _ALL_METRICS

    def snapshot(self, gpu_index: int) -> GpuMetrics:
        """Capture metric snapshot for *gpu_index*.

        Runs ``amd-smi metric --gpu N --json`` exactly **once** via
        ``self._executor`` and extracts only the metrics in ``self._metrics``
        from the single JSON response.  Never raises — all failures are
        silently suppressed.

        Args:
            gpu_index: Zero-based GPU ordinal to inspect.

        Returns:
            ``GpuMetrics`` with requested fields populated.
        """
        result = GpuMetrics(gpu_index=gpu_index)
        try:
            from framework.rocm.libs.amd_smi import (  # pylint: disable=import-outside-toplevel
                _parse_clock,
                _parse_ecc,
                _parse_temp,
                _parse_util,
                _parse_vram,
                _run_metric_json,
            )

            entry = _run_metric_json(self._executor, gpu_index)
            if entry is None:
                return result

            if "temp" in self._metrics:
                result.temp_c = _parse_temp(entry)
            if "vram" in self._metrics:
                vram = _parse_vram(entry, gpu_index)
                if vram is not None:
                    result.vram_used_mb = vram.used_mb
                    result.vram_free_mb = vram.free_mb
                    result.vram_total_mb = vram.total_mb
            if "util" in self._metrics:
                result.compute_pct = _parse_util(entry)
            if "ecc" in self._metrics:
                result.ecc_errors = _parse_ecc(entry)
            if "clock" in self._metrics:
                result.clock_state = _parse_clock(entry)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("GpuMonitor.snapshot GPU %d: %s", gpu_index, exc)
        return result

    def summary_line(self, metrics: GpuMetrics, phase: str) -> str:
        """One-line metric summary for console / session.log.

        Only includes fields that were configured in ``self._metrics``.
        Fields that were collected but returned ``None`` appear as ``n/a``.
        Fields that were not collected are omitted entirely.

        Args:
            metrics: Snapshot from ``snapshot()``.
            phase:   ``"pre"`` or ``"post"``.

        Returns:
            Single-line string, e.g.:
            ``[pre-health]  GPU-0  temp=47C  vram=1024/32768MB(3%)  util=0%  ECC=0``
        """
        tag = f"[{phase}-health]"
        label = f"GPU-{metrics.gpu_index}"
        parts: list[str] = []

        if metrics.temp_c is not None:
            parts.append(f"temp={metrics.temp_c}C")
        elif "temp" in self._metrics:
            parts.append("temp=n/a")

        if metrics.vram_used_mb is not None and metrics.vram_total_mb is not None:
            pct = int(metrics.vram_used_mb * 100 / metrics.vram_total_mb) if metrics.vram_total_mb else 0
            parts.append(f"vram={metrics.vram_used_mb}/{metrics.vram_total_mb}MB({pct}%)")
        elif "vram" in self._metrics:
            parts.append("vram=n/a")

        if metrics.compute_pct is not None:
            parts.append(f"util={metrics.compute_pct}%")
        elif "util" in self._metrics:
            parts.append("util=n/a")

        if metrics.ecc_errors is not None:
            parts.append(f"ECC={metrics.ecc_errors}")
        elif "ecc" in self._metrics:
            parts.append("ECC=n/a")

        if metrics.clock_state:
            parts.append(f"clk={metrics.clock_state}")
        elif "clock" in self._metrics:
            parts.append("clk=n/a")

        return f"{tag:<14} {label}  {'  '.join(parts)}"

    def delta_line(self, pre: GpuMetrics, post: GpuMetrics) -> str:
        """One-line delta between pre and post snapshots.

        Args:
            pre:  Snapshot taken before the test ran.
            post: Snapshot taken after the test ran.

        Returns:
            Single-line string, e.g.:
            ``[health-delta]  GPU-0  temp=+25C  vram=+7168MB  util=peak≈85%``
        """
        label = f"GPU-{pre.gpu_index}"
        parts: list[str] = []

        if pre.temp_c is not None and post.temp_c is not None:
            delta_t = post.temp_c - pre.temp_c
            sign = "+" if delta_t >= 0 else ""
            parts.append(f"temp={sign}{delta_t}C")
        elif post.temp_c is not None:
            parts.append(f"temp=>{post.temp_c}C")

        if pre.vram_used_mb is not None and post.vram_used_mb is not None:
            delta_v = post.vram_used_mb - pre.vram_used_mb
            sign = "+" if delta_v >= 0 else ""
            parts.append(f"vram={sign}{delta_v}MB")
        elif post.vram_used_mb is not None:
            parts.append(f"vram=>{post.vram_used_mb}MB")

        if post.compute_pct is not None:
            parts.append(f"util=peak≈{post.compute_pct}%")

        if not parts:
            parts.append("no delta (metrics unavailable)")

        return f"[health-delta]  {label}  {'  '.join(parts)}"


# ---------------------------------------------------------------------------
# GpuBackgroundMonitor — continuous background poller (--monitor-gpu)
# ---------------------------------------------------------------------------


class GpuBackgroundMonitor:
    """Continuous background GPU metric poller writing to a per-test log file.

    Runs as a daemon thread so it never blocks pytest shutdown.  The fixture
    calls ``start()`` just before yielding the executor to the test and
    ``stop()`` in the ``finally`` block — the thread terminates regardless
    of whether ``duration_secs`` has elapsed.

    The caller is responsible for supplying the right ``executor``:
        - ``CpuExecutor`` for local nodes (no ROCR env injection)
        - ``SshExecutor`` for remote nodes (reuses the pool's existing session)

    Log format (timestamped text rows)::

        # GPU continuous monitor — test: test_foo
        # GPUs: [0, 1]  interval: 5.0s  duration: test-bounded
        # Cols: timestamp | GPU-N | temp=...  vram=...  util=...  ECC=...  clk=...
        2026-05-11 15:40:01 | GPU-0 | temp=47C  vram=1024/32768MB(3%)  util=0%  ECC=0  clk=auto
        2026-05-11 15:40:01 | GPU-1 | temp=49C  vram=512/32768MB(2%)   util=0%  ECC=0  clk=auto
        ...
        # Monitor stopped (test completed)

    Args:
        executor:      Pre-built ``AbstractExecutor`` — ``CpuExecutor`` for
                       local nodes, ``SshExecutor`` for remote.
        metrics:       Set of metric names to collect per sample.
        interval_secs: Seconds between ``amd-smi`` polls.
        duration_secs: Optional additional cap in seconds (0 = test-bounded only).
    """

    def __init__(
        self,
        executor: AbstractExecutor,
        metrics: set[str],
        interval_secs: float,
        duration_secs: float,
    ) -> None:
        self._executor = executor
        self._metrics: frozenset[str] = frozenset(metrics) if metrics else _ALL_METRICS
        self._interval = max(interval_secs, 1.0)  # floor at 1s to avoid amd-smi hammering
        self._duration = duration_secs
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, gpu_indices: list[int], log_path: str) -> None:
        """Write log header and spawn the background polling thread.

        Args:
            gpu_indices: Physical GPU ordinals to monitor (one row per GPU per sample).
            log_path:    Absolute path to the ``_gpu_monitor.log`` file.
        """
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(gpu_indices, log_path),
            daemon=True,
            name=f"gpu-monitor-{','.join(str(i) for i in gpu_indices)}",
        )
        self._thread.start()
        logger.debug(
            "GpuBackgroundMonitor: started for GPU(s) %s, interval=%.1fs, log=%s",
            gpu_indices,
            self._interval,
            log_path,
        )

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to finish (up to 10 s).

        Safe to call even if ``start()`` was never called or the thread has already
        exited (e.g., due to ``duration_secs`` expiry).
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=10)
        self._thread = None
        logger.debug("GpuBackgroundMonitor: stopped")

    def _format_row(self, ts: str, snap: GpuMetrics) -> str:
        """Format one timestamped sample row for the log file."""
        label = f"GPU-{snap.gpu_index}"
        parts: list[str] = []

        if snap.temp_c is not None:
            parts.append(f"temp={snap.temp_c}C")
        elif "temp" in self._metrics:
            parts.append("temp=n/a")

        if snap.vram_used_mb is not None and snap.vram_total_mb is not None:
            pct = int(snap.vram_used_mb * 100 / snap.vram_total_mb) if snap.vram_total_mb else 0
            parts.append(f"vram={snap.vram_used_mb}/{snap.vram_total_mb}MB({pct}%)")
        elif "vram" in self._metrics:
            parts.append("vram=n/a")

        if snap.compute_pct is not None:
            parts.append(f"util={snap.compute_pct}%")
        elif "util" in self._metrics:
            parts.append("util=n/a")

        if snap.ecc_errors is not None:
            parts.append(f"ECC={snap.ecc_errors}")
        elif "ecc" in self._metrics:
            parts.append("ECC=n/a")

        if snap.clock_state:
            parts.append(f"clk={snap.clock_state}")
        elif "clock" in self._metrics:
            parts.append("clk=n/a")

        return f"{ts} | {label} | {'  '.join(parts)}"

    def _run(self, gpu_indices: list[int], log_path: str) -> None:  # pylint: disable=too-many-locals
        """Main polling loop — runs in the daemon thread.

        ``amd-smi metric --gpu N --json`` is called exactly **once per GPU per
        sample interval**.  All configured metrics are extracted from that single
        JSON response — no repeated invocations for different metric fields.
        """
        from framework.rocm.libs.amd_smi import (  # pylint: disable=import-outside-toplevel
            _parse_clock,
            _parse_ecc,
            _parse_temp,
            _parse_util,
            _parse_vram,
            _run_metric_json,
        )

        duration_str = f"{self._duration:.0f}s cap" if self._duration > 0 else "test-bounded"
        deadline = time.monotonic() + self._duration if self._duration > 0 else None

        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(
                    f"# GPU continuous monitor — GPUs: {gpu_indices}"
                    f"  interval: {self._interval}s  duration: {duration_str}\n"
                )
                fh.write(f"# Cols: timestamp | GPU-N |" f" {', '.join(sorted(self._metrics))}\n")
                fh.flush()

                while not self._stop_event.is_set():
                    if deadline is not None and time.monotonic() >= deadline:
                        fh.write("# Monitor stopped (duration cap reached)\n")
                        fh.flush()
                        break

                    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    for gpu_idx in gpu_indices:
                        snap = GpuMetrics(gpu_index=gpu_idx)
                        try:  # pylint: disable=too-many-nested-blocks
                            # Single amd-smi call per GPU per sample — parse all metrics from it.
                            entry = _run_metric_json(self._executor, gpu_idx)
                            if entry is not None:
                                if "temp" in self._metrics:
                                    snap.temp_c = _parse_temp(entry)
                                if "vram" in self._metrics:
                                    vram = _parse_vram(entry, gpu_idx)
                                    if vram is not None:
                                        snap.vram_used_mb = vram.used_mb
                                        snap.vram_free_mb = vram.free_mb
                                        snap.vram_total_mb = vram.total_mb
                                if "util" in self._metrics:
                                    snap.compute_pct = _parse_util(entry)
                                if "ecc" in self._metrics:
                                    snap.ecc_errors = _parse_ecc(entry)
                                if "clock" in self._metrics:
                                    snap.clock_state = _parse_clock(entry)
                        except Exception as exc:  # pylint: disable=broad-exception-caught
                            logger.debug("GpuBackgroundMonitor: sample failed for GPU %d: %s", gpu_idx, exc)
                        fh.write(self._format_row(ts, snap) + "\n")

                    fh.flush()
                    self._stop_event.wait(timeout=self._interval)

                fh.write("# Monitor stopped (test completed)\n")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("GpuBackgroundMonitor: log write failed: %s", exc)
