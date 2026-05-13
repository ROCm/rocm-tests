# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
drain.py -- GPU memory drain checker for graceful slot release.

After a test finishes and its subprocess exits, the AMD driver typically frees
device VRAM within milliseconds.  However, in back-to-back parallel executions
the next test may briefly observe residual VRAM usage from the previous test's
dying process.  ``GpuDrainChecker`` polls ``amd-smi`` to wait until used VRAM
drops below a configurable threshold before the slot is declared free.

Usage
-----
Sequential tests (no xdist):
    A short fixed sleep (``--gpu-drain-secs``, default 0.5 s) is used instead
    of active polling — amd-smi is not invoked.

Parallel tests (xdist active):
    ``GpuDrainChecker.wait_for_drain()`` is called; it polls every
    ``poll_interval_secs`` until VRAM drops below ``threshold_mb`` or
    ``timeout_secs`` is reached.  If ``amd-smi`` is unavailable the checker
    returns immediately (graceful degradation — no test is blocked).

Integration
-----------
Called from ``target_executor``, ``multi_gpu_fixture``, ``multi_node_fixture``
``finally`` blocks in ``remote_node_plugin.py`` when xdist is active.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

# Default threshold: consider GPU drained when used VRAM < 512 MB.
_DEFAULT_THRESHOLD_MB: int = 512
_DEFAULT_POLL_INTERVAL: float = 1.0
_DEFAULT_TIMEOUT: float = 30.0


class GpuDrainChecker:
    """Poll ``amd-smi`` until GPU VRAM usage drops below a threshold.

    Designed for the release path in GPU slot fixtures — ensures the next test
    allocated to the same GPU does not observe stale device memory from its
    predecessor.

    Graceful degradation:
        When ``amd-smi`` is absent or returns an unexpected format, the method
        logs a warning and returns ``True`` immediately so that the fixture
        teardown is never blocked by an external tool failure.

    Attributes:
        rock_dir: Optional path to a TheRock/ROCm install tree.  When set,
                  ``{rock_dir}/bin/amd-smi`` is tried as a fallback if the
                  system ``amd-smi`` is not found on PATH.
    """

    def __init__(self, rock_dir: str | None = None) -> None:
        self.rock_dir = rock_dir

    def wait_for_drain(
        self,
        gpu_index: int,
        threshold_mb: int = _DEFAULT_THRESHOLD_MB,
        poll_interval_secs: float = _DEFAULT_POLL_INTERVAL,
        timeout_secs: float = _DEFAULT_TIMEOUT,
    ) -> bool:
        """Wait until used VRAM on *gpu_index* drops below *threshold_mb*.

        Args:
            gpu_index:         GPU ordinal to check (``ROCR_VISIBLE_DEVICES`` index).
            threshold_mb:      Consider drained when used VRAM < this value (MB).
                               Default 512 MB.
            poll_interval_secs: Seconds between ``amd-smi`` calls (default 1 s).
            timeout_secs:      Give up and return ``False`` after this many seconds
                               (default 30 s).  The fixture will still release the
                               slot — the next test just starts sooner.

        Returns:
            ``True`` when VRAM drained within timeout or amd-smi is unavailable.
            ``False`` when timeout elapsed before VRAM reached the threshold.
        """
        amd_smi = self._find_amd_smi()
        if amd_smi is None:
            logger.debug(
                "GpuDrainChecker: amd-smi not found — skipping drain check for GPU-%d",
                gpu_index,
            )
            return True

        deadline = time.monotonic() + timeout_secs
        attempt = 0
        while True:
            used_mb = self._query_used_vram_mb(amd_smi, gpu_index)
            if used_mb is None:
                # Couldn't parse — degrade gracefully
                logger.debug(
                    "GpuDrainChecker: could not parse VRAM for GPU-%d, skipping drain",
                    gpu_index,
                )
                return True

            attempt += 1
            if used_mb < threshold_mb:
                if attempt > 1:
                    logger.info(
                        "GpuDrainChecker: GPU-%d drained to %d MB after %d poll(s)",
                        gpu_index,
                        used_mb,
                        attempt,
                    )
                return True

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "GpuDrainChecker: GPU-%d still has %d MB used after %.0fs timeout — releasing slot anyway",
                    gpu_index,
                    used_mb,
                    timeout_secs,
                )
                return False

            logger.debug(
                "GpuDrainChecker: GPU-%d used=%d MB >= threshold=%d MB; waiting %.1fs (%.1fs remaining)",
                gpu_index,
                used_mb,
                threshold_mb,
                poll_interval_secs,
                remaining,
            )
            time.sleep(min(poll_interval_secs, remaining))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_amd_smi(self) -> str | None:
        """Return the path to an ``amd-smi`` binary, or ``None`` if not found."""
        # Prefer system PATH entry
        path = shutil.which("amd-smi")
        if path:
            return path

        # Fallback: TheRock/ROCm install
        if self.rock_dir:
            candidate = os.path.join(self.rock_dir, "bin", "amd-smi")
            if os.path.isfile(candidate):
                return candidate

        return None

    def _query_used_vram_mb(self, amd_smi: str, gpu_index: int) -> int | None:
        """Run ``amd-smi memory -g <N> --json`` and return used VRAM in MB.

        Args:
            amd_smi:   Path to the ``amd-smi`` binary.
            gpu_index: GPU ordinal to query.

        Returns:
            Used VRAM in MB, or ``None`` if parsing fails.
        """
        try:
            proc = subprocess.run(
                [amd_smi, "memory", "-g", str(gpu_index), "--json"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode != 0:
                logger.debug(
                    "GpuDrainChecker: amd-smi exited %d: %s",
                    proc.returncode,
                    proc.stderr[:200],
                )
                return None
            return self._parse_used_mb(proc.stdout)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.debug("GpuDrainChecker: amd-smi call failed: %s", exc)
            return None

    @staticmethod
    def _parse_used_mb(json_str: str) -> int | None:
        """Parse ``amd-smi memory --json`` output to extract used VRAM in MB.

        Handles several amd-smi JSON schema variants:

        - amd-smi 6.x: ``[{"gpu": 0, "mem_info": {"vram_used": {"value": N, "unit": "MB"}}}]``
        - amd-smi 5.x: ``[{"gpu": 0, "vram_used_mb": N}]``
        - Flat dict (some firmware): ``{"vram_used_mb": N}``

        Args:
            json_str: Raw JSON string from ``amd-smi memory --json``.

        Returns:
            Used VRAM in MB (integer), or ``None`` if the format is unrecognised.
        """
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            return None

        # Normalise to a single device dict
        if isinstance(data, list) and data:
            data = data[0]
        if not isinstance(data, dict):
            return None

        # Schema variant 1: nested mem_info → vram_used
        mem_info = data.get("mem_info", {})
        if mem_info:
            vram_used = mem_info.get("vram_used", {})
            if isinstance(vram_used, dict):
                raw = vram_used.get("value", 0)
                unit = vram_used.get("unit", "MB").upper()
                mb = int(raw) if unit == "MB" else int(raw) // (1024 * 1024)
                return mb
            if isinstance(vram_used, (int, float)):
                return int(vram_used)

        # Schema variant 2: flat vram_used_mb
        flat = data.get("vram_used_mb")
        if flat is not None:
            return int(flat)

        # Schema variant 3: nested vram → used
        vram = data.get("vram", {})
        if isinstance(vram, dict):
            used = vram.get("used", {})
            if isinstance(used, dict):
                raw = used.get("value", 0)
                unit = used.get("unit", "MB").upper()
                return int(raw) if unit == "MB" else int(raw) // (1024 * 1024)
            if isinstance(used, (int, float)):
                return int(used)

        return None
