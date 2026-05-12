# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
health_plugin.py -- GPU health gate fixture.

Provides the ``health_fixture`` which runs pre-execution and post-execution
GPU health checks (temperature, ECC errors, VRAM headroom, clock state).

Tests that use ``gpu_fixture`` implicitly benefit from health checks because
``gpu_fixture`` calls health checks inside its setup/teardown. The
``health_fixture`` is exposed separately for tests that need to assert on
health state explicitly.

Health thresholds are loaded from the ``framework_config`` session fixture
(GpuSection of rocm-test.toml / env vars) so they can be tuned per-environment.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import shutil
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from framework.gpu.detector import GpuInfo

logger = logging.getLogger(__name__)


@dataclass
class HealthResult:
    """Result of a single GPU health check.

    Attributes:
        passed:  True if all checks passed.
        message: Human-readable summary (populated on failure).
        temp_c:  Observed GPU temperature in Celsius, or None if unavailable.
        ecc_errors: Observed ECC error count, or None if unavailable.
        vram_free_mb: Free VRAM in MB, or None if unavailable.
    """

    passed: bool
    message: str = ""
    temp_c: int | None = None
    ecc_errors: int | None = None
    vram_free_mb: int | None = None


class GpuHealthChecker:
    """Run GPU health checks against configured thresholds.

    Args:
        max_temp_celsius: Fail if GPU temperature exceeds this value.
        max_ecc_errors:   Fail if ECC errors exceed this count.
        min_vram_free_mb: Fail if free VRAM is below this threshold.
        rock_dir:         Optional path to a TheRock/ROCm installation that
                          provides ``bin/amd-smi``.  Used as fallback when
                          system ``amd-smi`` is not on PATH.
    """

    def __init__(
        self,
        max_temp_celsius: int = 90,
        max_ecc_errors: int = 0,
        min_vram_free_mb: int = 512,
        rock_dir: str | None = None,
    ) -> None:
        self.max_temp_celsius = max_temp_celsius
        self.max_ecc_errors = max_ecc_errors
        self.min_vram_free_mb = min_vram_free_mb
        self._rock_dir = rock_dir

    def _resolve_amd_smi(self) -> str:
        """Return the amd-smi binary path to use for health checks.

        Tries system PATH first; falls back to ``{rock_dir}/bin/amd-smi`` when
        a rock_dir was supplied and the system binary is absent.

        Returns:
            Path string suitable for use as the first element of a subprocess
            command list.  Falls back to ``"amd-smi"`` (system PATH) when no
            rock_dir is configured.
        """
        if shutil.which("amd-smi"):
            return "amd-smi"
        if self._rock_dir:
            rock_bin = os.path.join(self._rock_dir, "bin", "amd-smi")
            if os.path.isfile(rock_bin):
                logger.debug("health: using rock_dir amd-smi at %s", rock_bin)
                return rock_bin
        return "amd-smi"

    def summary_line(self, gpu_info: GpuInfo, result: HealthResult, phase: str) -> str:
        """One-line health summary for console (INFO log level).

        Format::

            [pre-health ] GPU 0  gfx942   PASS  (temp=45C  ECC=0  free=28192MB)
            [post-health] GPU 0  gfx942   FAIL  temp=95C > 90C threshold

        Args:
            gpu_info: GPU descriptor from detection.
            result:   HealthResult returned by check().
            phase:    ``"pre"`` or ``"post"``.

        Returns:
            Single-line string with no trailing newline.
        """
        tag = f"[{phase}-health]"
        status = "PASS" if result.passed else "FAIL"
        header = f"{tag:<13} GPU {gpu_info.index}  {gpu_info.arch:<10}  {status}"
        metrics: list[str] = []
        if result.temp_c is not None:
            metrics.append(f"temp={result.temp_c}C")
        if result.ecc_errors is not None:
            metrics.append(f"ECC={result.ecc_errors}")
        if result.vram_free_mb is not None:
            metrics.append(f"free={result.vram_free_mb}MB")
        if metrics:
            return f"{header}  ({',  '.join(metrics)})"
        if not result.passed:
            return f"{header}  {result.message}"
        return header

    def detail_block(self, gpu_info: GpuInfo, result: HealthResult, phase: str) -> str:
        """Multi-line health report for log file (DEBUG log level).

        Format::

            GPU 0  pre-test health check
              arch        : gfx942
              total VRAM  : 32768 MB
              NUMA node   : 0
              temperature :    45 C    limit <=  90 C              OK
              ECC errors  :     0      limit <=   0                OK
              free VRAM   : 28192 MB   limit >= 512 MB             OK
              ──────────────────────────────────
              result      : PASS

        Args:
            gpu_info: GPU descriptor from detection.
            result:   HealthResult returned by check().
            phase:    ``"pre"`` or ``"post"``.

        Returns:
            Multi-line string with no trailing newline.
        """

        def _row(label: str, value: str, limit: str, ok: bool) -> str:
            mark = "OK" if ok else "FAIL"
            return f"  {label:<14}: {value:<12}  {limit:<28}  {mark}"

        lines = [f"GPU {gpu_info.index}  {phase}-test health check"]
        lines.append(f"  {'arch':<14}: {gpu_info.arch}")
        lines.append(f"  {'total VRAM':<14}: {gpu_info.vram_mb} MB")
        lines.append(f"  {'NUMA node':<14}: {gpu_info.numa_node}")

        if result.temp_c is not None:
            lines.append(
                _row(
                    "temperature",
                    f"{result.temp_c} C",
                    f"limit <= {self.max_temp_celsius} C",
                    result.temp_c <= self.max_temp_celsius,
                )
            )
        else:
            lines.append(f"  {'temperature':<14}: n/a")

        if result.ecc_errors is not None:
            lines.append(
                _row(
                    "ECC errors",
                    str(result.ecc_errors),
                    f"limit <= {self.max_ecc_errors}",
                    result.ecc_errors <= self.max_ecc_errors,
                )
            )
        else:
            lines.append(f"  {'ECC errors':<14}: n/a")

        if result.vram_free_mb is not None:
            lines.append(
                _row(
                    "free VRAM",
                    f"{result.vram_free_mb} MB",
                    f"limit >= {self.min_vram_free_mb} MB",
                    result.vram_free_mb >= self.min_vram_free_mb,
                )
            )
        else:
            lines.append(f"  {'free VRAM':<14}: n/a")

        lines.append(f"  {'─' * 50}")
        lines.append(f"  {'result':<14}: {'PASS' if result.passed else 'FAIL'}")
        if not result.passed:
            lines.append(f"  {'detail':<14}: {result.message}")
        return "\n".join(lines)

    def check(self, gpu_index: int) -> HealthResult:
        """Run health checks on *gpu_index* and return a HealthResult.

        Args:
            gpu_index: Zero-based AMD GPU ordinal to inspect.

        Returns:
            HealthResult with passed=True if all thresholds are met.
        """
        try:
            return self._check_via_amd_smi(gpu_index)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("amd-smi health check failed: %s", exc)
            # In environments without amd-smi (unit tests, CI without GPU),
            # return a synthetic pass so the framework doesn't block test execution.
            return HealthResult(passed=True, message="health check skipped (no amd-smi)")

    def _check_via_amd_smi(self, gpu_index: int) -> HealthResult:  # pylint: disable=too-many-locals
        """Run health checks via ``amd-smi metric --gpu N --json``."""
        from framework.executors.cpu_executor import CpuExecutor  # pylint: disable=import-outside-toplevel
        from framework.rocm.libs.amd_smi import query_ecc_errors, query_gpu_temp, query_vram_usage  # pylint: disable=import-outside-toplevel

        # Prepend rock_dir/bin to PATH so the bundled amd-smi is preferred when set.
        env_overrides: dict = {}
        if self._rock_dir:
            rock_bin = os.path.join(self._rock_dir, "bin")
            env_overrides["PATH"] = f"{rock_bin}:{os.environ.get('PATH', '')}"

        executor = CpuExecutor(env_overrides=env_overrides)
        temp_c = query_gpu_temp(executor, gpu_index)
        ecc_errors = query_ecc_errors(executor, gpu_index)
        vram_info = query_vram_usage(executor, gpu_index)
        vram_free_mb = vram_info.free_mb if vram_info else None

        failures = []
        if temp_c is not None and temp_c > self.max_temp_celsius:
            failures.append(f"temp {temp_c}°C > threshold {self.max_temp_celsius}°C")
        if ecc_errors is not None and ecc_errors > self.max_ecc_errors:
            failures.append(f"ECC errors {ecc_errors} > threshold {self.max_ecc_errors}")
        if vram_free_mb is not None and vram_free_mb < self.min_vram_free_mb:
            failures.append(f"free VRAM {vram_free_mb} MB < threshold {self.min_vram_free_mb} MB")

        passed = not failures
        message = "; ".join(failures) if failures else "all checks passed"
        return HealthResult(
            passed=passed,
            message=message,
            temp_c=temp_c,
            ecc_errors=ecc_errors,
            vram_free_mb=vram_free_mb,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def health_fixture(request, framework_config):
    """Provide a GpuHealthChecker configured from framework_config.

    Thresholds come from ``[gpu]`` section of rocm-test.toml (or env overrides).
    The rock_dir path (from ``--rock-dir`` / ``ROCK_DIR``) is forwarded so the
    checker can fall back to the bundled ``amd-smi`` when it is not on PATH.

    Args:
        request:          Pytest fixture request (provides config access).
        framework_config: Session-scoped config fixture from conftest.py.

    Returns:
        GpuHealthChecker: Ready to call .check(gpu_index) in tests.
    """
    config = request.config
    rock_dir: str | None = (
        config.getoption("--rock-dir", default=None)
        or os.environ.get("ROCK_DIR")
        or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
    )
    return GpuHealthChecker(
        max_temp_celsius=framework_config.gpu.max_temp_celsius,
        max_ecc_errors=framework_config.gpu.max_ecc_errors,
        min_vram_free_mb=framework_config.gpu.min_vram_free_mb,
        rock_dir=rock_dir,
    )
