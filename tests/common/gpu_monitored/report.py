# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Adapter: maps internal exit codes to display strings and writes HTML report.

The underlying `generate_report` module lives in the same package
(``tests.common.gpu_monitored.generate_report``) and can be invoked as a script via
``python3 -m tests.common.gpu_monitored.generate_report``.
"""

from __future__ import annotations

from pathlib import Path

from tests.common.gpu_monitored.generate_report import generate_report


def write_report(run_dir: Path, test_name: str, exit_code: int, duration: int, *, unsupported: bool = False) -> None:
    """Generate HTML report at `run_dir/report.html`.

    ``unsupported`` is a classification flag set by the test (e.g. RVS
    when no per-GPU config is available); it takes priority over the
    process exit code for display purposes.
    """
    if unsupported:
        result_str = "UNSUPPORTED"
    elif exit_code == 0:
        result_str = "PASS"
    else:
        result_str = f"FAIL (exit {exit_code})"
    generate_report(
        run_dir=str(run_dir),
        test_name=test_name,
        result=result_str,
        duration=int(duration),
        output=str(run_dir / "report.html"),
    )
