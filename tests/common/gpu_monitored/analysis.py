# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Adapter: wraps analyze_monitoring with a file-writing helper.

The underlying `analyze_monitoring` module lives in the same package
(`tests.common.gpu_monitored.analyze_monitoring`).
"""

from __future__ import annotations

from pathlib import Path

from tests.common.gpu_monitored.analyze_monitoring import (
    enrich_summary,
    health_checks_text,
    run_analysis,
)


def analyze_and_write(run_dir: Path, test_name: str) -> None:
    """Run analysis, enrich summary.json, write health_checks.txt.

    Errors are non-fatal -- caller logs via `analysis.stderr.log`.
    """
    analysis = run_analysis(str(run_dir), test_name)
    if not analysis:
        return
    enrich_summary(str(run_dir / "summary.json"), analysis)
    (run_dir / "health_checks.txt").write_text(health_checks_text(analysis))
