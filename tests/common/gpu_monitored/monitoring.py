# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU monitoring context manager (amd-smi monitor subprocess + optional CU occupancy).

Uses the framework's command execution utilities where applicable. The main
monitoring subprocess (`amd-smi monitor --csv --file`) writes directly to a
file via its own --file flag, so subprocess.Popen is used for lifecycle management.
"""

import json
import logging
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import List, Optional

from framework.executors.local_executor import run_cmd_get_stdout_stderr
from framework.rocm.libs.amd_smi import _get

logger = logging.getLogger(__name__)


class Monitor:
    """Context manager that runs `amd-smi monitor` in the background.

    Use as:
        with Monitor(csv_file=..., sample_interval=2) as mon:
            ... run workload ...
    The monitor subprocess is cleanly killed on __exit__.
    """

    def __init__(self, csv_file: Path, cu_csv: Path, sample_interval: int = 2,
                 enable_cu_occupancy: bool = False, amd_smi_path: str = "amd-smi"):
        self.csv_file = Path(csv_file)
        self.cu_csv = Path(cu_csv)
        self.sample_interval = sample_interval
        self.enable_cu_occupancy = enable_cu_occupancy
        self._amd_smi = amd_smi_path
        self._monitor_proc: Optional[subprocess.Popen] = None
        self._cu_thread: Optional[threading.Thread] = None
        self._cu_stop = threading.Event()

    def __enter__(self) -> "Monitor":
        self._monitor_proc = subprocess.Popen(
            [self._amd_smi, "monitor", "-p", "-t", "-u", "-m", "-v",
             "-w", str(self.sample_interval), "--csv", "--file", str(self.csv_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        try:
            time.sleep(0.5)
            if self._monitor_proc.poll() is not None:
                rc = self._monitor_proc.returncode
                logger.warning(
                    "amd-smi monitor exited immediately (rc=%d) — "
                    "monitoring CSV will be empty", rc,
                )

            if self.enable_cu_occupancy:
                self.cu_csv.write_text(
                    "timestamp,gpu,pid,cu_occupancy,vram_mb\n"
                )
                self._cu_thread = threading.Thread(target=self._cu_loop, daemon=True)
                self._cu_thread.start()
        except BaseException:
            self._kill_monitor()
            raise
        return self

    def _kill_monitor(self) -> None:
        """Best-effort termination of the amd-smi monitor subprocess."""
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
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cu_stop.set()
        if self._cu_thread is not None:
            self._cu_thread.join(timeout=5)
        self._kill_monitor()
        return False

    def _cu_loop(self) -> None:
        """Background loop: sample `amd-smi process --json` every 5s into cu_occupancy CSV."""
        while not self._cu_stop.is_set():
            if self._cu_stop.wait(5):
                return
            try:
                rc, stdout, _stderr = run_cmd_get_stdout_stderr(
                    self._amd_smi, "process", "--json",
                    timeout=10, quiet=True,
                )
                if rc != 0 or not stdout.strip():
                    continue
            except Exception:
                continue

            if self._cu_stop.is_set():
                return
            try:
                rows = self._parse_cu_occupancy(stdout)
                if rows:
                    with self.cu_csv.open("a") as f:
                        f.write("\n".join(rows) + "\n")
            except Exception:
                pass

    @staticmethod
    def _parse_cu_occupancy(out: str) -> List[str]:
        """Parse `amd-smi process --json` output; return CSV rows for active processes."""
        try:
            data = json.loads(out)
        except (json.JSONDecodeError, ValueError):
            return []

        ts = str(int(time.time()))
        rows: List[str] = []

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
                cu = _get(p, ("cu_occupancy",), ("CU_OCCUPANCY",), default=0)
                pid = _get(p, ("pid",), ("PID",), default="?")
                vm = _get(p, ("memory_usage",), ("MEMORY_USAGE",), default={})
                if isinstance(vm, dict):
                    vr = _get(vm, ("vram_mem",), ("VRAM_MEM",), default="0")
                else:
                    vr = vm
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
        """Convert an amd-smi memory value to MB."""
        if raw is None:
            return 0.0
        s = str(raw).strip()
        if not s:
            return 0.0
        m = re.match(r"^\s*([0-9.+-eE]+)\s*(GB|MB|KB|B)?\s*$", s,
                     flags=re.IGNORECASE)
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


def count_csv_samples(csv_path: Path) -> int:
    """Count data rows (not headers) in a monitoring CSV."""
    if not csv_path.is_file():
        return 0
    try:
        with csv_path.open() as f:
            lc = sum(1 for _line in f)
        return max(0, lc - 1)
    except Exception:
        return 0
