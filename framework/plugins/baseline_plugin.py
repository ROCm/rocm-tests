# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
baseline_plugin.py -- Performance regression detection against YAML baselines.

Provides the ``baseline_fixture`` which compares float metrics emitted by tests
(in ``KEY=<value>`` format on stdout) against per-arch YAML baseline files.

Baseline files live in ``tests/performance/baselines/<arch>/<benchmark>.yaml``.
The path is configurable via the ``[baselines]`` section of ``rocm-test.toml``
(or ``ROCM_TEST_BASELINES_BASELINE_DIR`` env var).

Baseline YAML format:
    THROUGHPUT_TFLOPS:
      value: 12.5
      tolerance_pct: 5.0   # Optional; falls back to [baselines].regression_pct

Usage in tests:
    def test_matmul(target_executor, baseline_fixture):
        result = target_executor.run("python3 -c 'print(\"THROUGHPUT_TFLOPS=12.8\")'")
        assert result.ok
        baseline_fixture.compare(result.stdout, benchmark="matmul_fp32")
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import pathlib

import pytest

logger = logging.getLogger(__name__)


@dataclass
class BaselineComparison:
    """Result of comparing a metric against its baseline.

    Attributes:
        metric:    Metric key name.
        observed:  Value reported by the test.
        baseline:  Expected value from the YAML file, or None if not found.
        delta_pct: Percentage deviation from baseline (negative = regression).
        passed:    True if within tolerance.
        message:   Human-readable summary.
    """

    metric: str
    observed: float
    baseline: float | None
    delta_pct: float | None
    passed: bool
    message: str


class BaselineChecker:
    """Compare test-reported metrics against YAML baseline files.

    Args:
        baseline_dir:   Root directory containing per-arch YAML baseline files.
        regression_pct: Default tolerance percentage (e.g. 5.0 means ±5%).
        arch:           GPU architecture string (e.g. ``"gfx942"``).
    """

    def __init__(
        self,
        baseline_dir: str,
        regression_pct: float = 5.0,
        arch: str = "unknown",
    ) -> None:
        self.baseline_dir = pathlib.Path(baseline_dir)
        self.regression_pct = regression_pct
        self.arch = arch

    def compare(self, output: str, benchmark: str) -> list[BaselineComparison]:
        """Parse KEY=value metrics from *output* and compare to baseline YAML.

        Args:
            output:    Stdout from gpu_fixture.run() containing KEY=value lines.
            benchmark: Baseline file name (without .yaml), e.g. ``"matmul_fp32"``.

        Returns:
            List of BaselineComparison, one per metric found in output.
        """
        baseline_data = self._load_baseline(benchmark)
        comparisons: list[BaselineComparison] = []

        for line in output.splitlines():
            line = line.strip()
            if "=" not in line:
                continue
            key, raw = line.split("=", 1)
            try:
                observed = float(raw.strip())
            except ValueError:
                continue

            entry = baseline_data.get(key)
            if entry is None:
                comparisons.append(
                    BaselineComparison(
                        metric=key,
                        observed=observed,
                        baseline=None,
                        delta_pct=None,
                        passed=True,
                        message=f"{key}: no baseline entry — recorded as {observed:.4g}",
                    )
                )
                continue

            expected = float(entry.get("value", 0))
            tolerance = float(entry.get("tolerance_pct", self.regression_pct))
            delta_pct = ((observed - expected) / expected * 100) if expected else 0.0
            passed = abs(delta_pct) <= tolerance
            status = "OK" if passed else "REGRESSION"
            comparisons.append(
                BaselineComparison(
                    metric=key,
                    observed=observed,
                    baseline=expected,
                    delta_pct=delta_pct,
                    passed=passed,
                    message=(
                        f"{key}: {status} observed={observed:.4g} "
                        f"baseline={expected:.4g} "
                        f"delta={delta_pct:+.1f}% (tol±{tolerance:.1f}%)"
                    ),
                )
            )
            if not passed:
                logger.warning("REGRESSION: %s", comparisons[-1].message)

        return comparisons

    def _load_baseline(self, benchmark: str) -> dict[str, dict]:
        """Load a YAML baseline file for the current arch and benchmark.

        Args:
            benchmark: Baseline file stem (e.g. ``"matmul_fp32"``).

        Returns:
            Dict of metric-name → {value, tolerance_pct}. Empty if file missing.
        """
        yaml_path = self.baseline_dir / self.arch / f"{benchmark}.yaml"
        if not yaml_path.exists():
            logger.debug("No baseline file found at %s", yaml_path)
            return {}
        import yaml  # pylint: disable=import-outside-toplevel

        with yaml_path.open() as fh:
            return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_fixture(framework_config, gpu_fixture):
    """Provide a BaselineChecker configured from framework_config and gpu_fixture.

    The checker is pre-configured with the correct arch (from the allocated GPU)
    and the baseline directory from rocm-test.toml.

    Args:
        framework_config: Session config with baseline_dir and regression_pct.
        gpu_fixture:      Provides gpu_info.arch for arch-specific baselines.

    Returns:
        BaselineChecker: Call .compare(result.stdout, benchmark="name").
    """
    return BaselineChecker(
        baseline_dir=framework_config.baselines.baseline_dir,
        regression_pct=framework_config.baselines.regression_pct,
        arch=gpu_fixture.gpu_info.arch,
    )
