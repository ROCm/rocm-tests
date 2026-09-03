# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""transferbench: GPU-to-GPU bandwidth sweep using TransferBench rsweep.

TransferBench (https://github.com/ROCm/TransferBench) benchmarks
simultaneous copies between CPUs and GPUs. The ``rsweep`` mode runs a
bandwidth sweep across all GPU-to-GPU transfer paths.

The ``TransferBench`` binary is shipped with ROCm Validation Suite (RVS)
-- same ``amdrocm7-rvs`` package / preinstall as ``rvs`` -- and is
expected at ``<rocm_root>/bin/TransferBench`` before the suite runs.
This test drives that preinstalled binary directly and does **not**
install a package or build from source.
"""

from __future__ import annotations

import os
from pathlib import Path

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.workloads.base import BuildContext, BuildStatus, RunContext, RunResult, Test, TestSpec


class TransferBench(Test):
    spec = TestSpec(
        name="transferbench",
        goal="TransferBench GPU-to-GPU bandwidth sweep with monitoring",
        # TransferBench is a data-movement benchmark, not a compute one: the
        # bulk of an ``rsweep`` is executed by SDMA engines and the CPU (a
        # representative 8x MI325X run logged 104 DMA and 24 CPU executors vs
        # 96 GPU ones), and even the GPU-executed transfers are lightweight
        # copy kernels. Average GFX utilisation therefore sits near zero on a
        # perfectly healthy run, so a compute-style ``min_util`` bar only
        # manufactures warnings. Health here is judged by the bandwidth
        # validator (per-transfer + aggregate results), not by GFX util.
        workload_profile={"min_util": 0, "min_vram_pct": 0},
    )

    DEFAULT_SWEEP_TIME_LIMIT = 300
    DEFAULT_SWEEP_MIN = 8
    DEFAULT_SWEEP_MAX = 8

    def build(self, ctx: BuildContext) -> BuildStatus:
        tb = self._find_bin(ctx.config)
        if tb is not None:
            print(f"  [build] TransferBench: found at {tb}")
            return BuildStatus.OK

        print(
            f"  [build] TransferBench: not found under {ctx.rocm_root} "
            f"(expected {ctx.rocm_root}/bin/TransferBench or "
            f"ROCM_TEST_TRANSFERBENCH_BIN from pytest fixtures). TransferBench "
            f"is shipped with ROCm Validation Suite; preinstall it under the "
            f"ROCm root or let tests/e2e/rvs/conftest.py build it from source."
        )
        return BuildStatus.BUILD_FAILED

    def available(self, config: Config) -> bool:
        return self._find_bin(config) is not None

    def run(self, ctx: RunContext) -> RunResult:
        tb_bin = self._find_bin(ctx.config)
        if tb_bin is None:
            print("  [transferbench] TransferBench not found")
            return RunResult(exit_code=1)

        # Read env-var overrides.
        # Validate them here rather than forwarding raw strings: a typo
        # like ``SWEEP_MAX=8m`` otherwise falls through to whatever the
        # TransferBench binary decides to do; instead we warn and fall
        # back to the documented default.
        sweep_time_limit = self._positive_env_int(
            "SWEEP_TIME_LIMIT",
            self.DEFAULT_SWEEP_TIME_LIMIT,
        )
        sweep_min = self._positive_env_int("SWEEP_MIN", self.DEFAULT_SWEEP_MIN)
        sweep_max = self._positive_env_int("SWEEP_MAX", self.DEFAULT_SWEEP_MAX)

        env = {
            "SWEEP_TIME_LIMIT": sweep_time_limit,
            "SWEEP_MIN": sweep_min,
            "SWEEP_MAX": sweep_max,
        }
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        rocm_lib = f"{ctx.rocm_root}/lib"
        env["LD_LIBRARY_PATH"] = f"{rocm_lib}:{existing_ld}" if existing_ld else rocm_lib
        env["PATH"] = f"{ctx.rocm_root}/bin:{os.environ.get('PATH', '')}"

        # ``SWEEP_TIME_LIMIT`` is the binary's own self-bound — rsweep
        # stops issuing new transfers once the time elapses. A
        # harness-side watchdog is opt-in via ``--per-iter-watchdog``.
        wd = ctx.config.per_iter_watchdog or None
        timeout_part = f"timeout {wd} " if wd else ""

        reproduce = (
            f"SWEEP_TIME_LIMIT={sweep_time_limit} "
            f"SWEEP_MIN={sweep_min} SWEEP_MAX={sweep_max} "
            f"{timeout_part}{tb_bin} rsweep"
        )

        print(f"  [transferbench] Running: {tb_bin} rsweep")
        print(f"  [transferbench] SWEEP_TIME_LIMIT={sweep_time_limit} " f"SWEEP_MIN={sweep_min} SWEEP_MAX={sweep_max}")

        rc = ctx.exec(
            [str(tb_bin), "rsweep"],
            env=env,
            timeout=wd,
        )
        if rc == 124:
            print(
                f"  [transferbench] FAIL: watchdog timeout — rsweep did "
                f"not complete within --per-iter-watchdog {wd}s"
            )
            return RunResult(exit_code=1, reproduce_cmd=reproduce)
        if rc == 0:
            print("  [transferbench] Completed rsweep successfully (rc=0)")
        return RunResult(exit_code=rc, reproduce_cmd=reproduce)

    @classmethod
    def _installed_bin(cls, rocm_root: Path) -> Path | None:
        """Return TransferBench binary when present and executable."""
        override = os.environ.get("ROCM_TEST_TRANSFERBENCH_BIN", "").strip()
        if override:
            p = Path(override)
            if p.is_file() and os.access(p, os.X_OK):
                return p
        installed = rocm_root / "bin" / "TransferBench"
        if installed.is_file() and os.access(installed, os.X_OK):
            return installed
        return None

    @classmethod
    def _find_bin(cls, config: Config) -> Path | None:
        return cls._installed_bin(config.rocm_root)

    @staticmethod
    def _positive_env_int(name: str, default: int) -> str:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return str(default)
        try:
            value = int(raw)
            if value <= 0:
                raise ValueError("must be positive")
        except ValueError as e:
            print(f"  [transferbench] WARNING: ignoring invalid " f"{name}={raw!r} ({e}); using default {default}")
            return str(default)
        return str(value)
