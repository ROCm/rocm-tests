# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored TransferBench rsweep (GPU-to-GPU bandwidth sweep).

Locates or builds TransferBench, runs ``rsweep`` mode under continuous
amd-smi monitoring, then performs 5-layer validation, anomaly detection,
and generates an HTML report.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import time

import pytest

from framework.builder.binary_builder import clone_repo
from framework.executors.local_executor import run_cmd_get_stdout_stderr
from tests.common.gpu_monitored.analysis import analyze_and_write
from tests.common.gpu_monitored.monitoring import count_csv_samples
from tests.common.gpu_monitored.report import write_report
from tests.common.gpu_monitored.validation import (
    capture_dmesg,
    dmesg_delta,
    pretest_health_probe,
    validate_transferbench_result,
)

logger = logging.getLogger(__name__)

_REPO_URL = "https://github.com/ROCm/TransferBench.git"
_DEFAULT_SWEEP_TIME_LIMIT = 300
_DEFAULT_SWEEP_MIN = 8
_DEFAULT_SWEEP_MAX = 8


def _positive_env_int(name: str, default: int) -> int:
    """Parse a positive integer from env var, falling back to default."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value <= 0:
            raise ValueError("must be positive")
    except ValueError as e:
        logger.warning(
            "Ignoring invalid %s=%r (%s); using default %d",
            name,
            raw,
            e,
            default,
        )
        return default
    return value


def _find_transferbench(rock_dir: str, build_dir: Path | None = None) -> Path | None:
    """Locate TransferBench binary: ROCm install first, then source build."""
    installed = Path(rock_dir) / "bin" / "TransferBench"
    if installed.is_file() and os.access(installed, os.X_OK):
        return installed
    if build_dir:
        built = build_dir / "TransferBench" / "TransferBench"
        if built.is_file() and os.access(built, os.X_OK):
            return built
    return None


def _build_transferbench(rock_dir: str, build_dir: Path) -> Path | None:
    """Build TransferBench from source using make.

    Returns path to the built binary, or None on failure.
    """
    src_dir = build_dir / "TransferBench"
    binary = src_dir / "TransferBench"

    if binary.is_file() and os.access(binary, os.X_OK):
        logger.info("TransferBench already built: %s", binary)
        return binary

    logger.info("Building TransferBench from source in %s", src_dir)

    try:
        clone_repo(_REPO_URL, src_dir)
    except Exception as e:
        logger.error("TransferBench: git clone failed: %s", e)
        return None

    build_env = dict(os.environ)
    build_env["ROCM_PATH"] = rock_dir
    build_env["HIP_PATH"] = rock_dir
    build_env["HIP_CLANG_PATH"] = os.path.join(rock_dir, "lib", "llvm", "bin")

    cpu_count = os.cpu_count() or 4
    build_rc, _build_out, build_err = run_cmd_get_stdout_stderr(
        "make",
        "-j",
        str(cpu_count),
        cwd=str(src_dir),
        timeout=600,
        quiet=True,
        env=build_env,
    )

    if not (binary.is_file() and os.access(binary, os.X_OK)):
        logger.error(
            "TransferBench: build FAILED (make rc=%d); stderr:\n%s",
            build_rc,
            build_err[:2000],
        )
        return None

    logger.info("TransferBench: build OK -> %s", binary)
    return binary


@pytest.mark.runtime.medium
def test_gpu_transferbench_monitored(
    gpu_monitor,
    run_dir,
    rock_dir,
    ld_path,
    kernel_health_probe,
):
    """Run TransferBench rsweep with GPU monitoring and full validation."""
    kernel_health_probe(lookback_sec=300, strict=False)

    _clean, health_summary = pretest_health_probe(lookback_min=5)
    with open(run_dir / "pretest_health.json", "w") as f:
        json.dump(health_summary, f, indent=2)

    # Locate or build TransferBench
    bld_env = os.environ.get("TRANSFERBENCH_BUILD_DIR", "")
    bld = Path(bld_env) if bld_env else (run_dir / "build")

    bin_path = _find_transferbench(rock_dir, bld)
    if bin_path is None:
        bld.mkdir(parents=True, exist_ok=True)
        bin_path = _build_transferbench(rock_dir, bld)
    if bin_path is None:
        pytest.skip("TransferBench binary not found and build from source failed")

    # Sweep configuration
    sweep_time_limit = _positive_env_int("SWEEP_TIME_LIMIT", _DEFAULT_SWEEP_TIME_LIMIT)
    sweep_min = _positive_env_int("SWEEP_MIN", _DEFAULT_SWEEP_MIN)
    sweep_max = _positive_env_int("SWEEP_MAX", _DEFAULT_SWEEP_MAX)

    rocm_lib = os.path.join(rock_dir, "lib")
    ld = ld_path["LD_LIBRARY_PATH"]
    if rocm_lib not in ld:
        ld = f"{rocm_lib}:{ld}"

    # Set env vars in the current process so they're inherited without
    # passing a full env dict (which causes verbose logging of all vars).
    os.environ["LD_LIBRARY_PATH"] = ld
    os.environ["SWEEP_TIME_LIMIT"] = str(sweep_time_limit)
    os.environ["SWEEP_MIN"] = str(sweep_min)
    os.environ["SWEEP_MAX"] = str(sweep_max)
    rocm_bin = os.path.join(rock_dir, "bin")
    if rocm_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{rocm_bin}:{os.environ.get('PATH', '')}"

    per_iter_timeout = int(os.environ.get("TRANSFERBENCH_TIMEOUT", "0")) or None

    logger.info(
        "Running TransferBench rsweep: SWEEP_TIME_LIMIT=%d " "SWEEP_MIN=%d SWEEP_MAX=%d, timeout %s",
        sweep_time_limit,
        sweep_min,
        sweep_max,
        f"{per_iter_timeout}s" if per_iter_timeout else "none (default 3600s)",
    )

    start_time = time.time()
    dmesg_before = capture_dmesg()

    with gpu_monitor(cu_occupancy=False):
        rc, stdout, stderr = run_cmd_get_stdout_stderr(
            str(bin_path),
            "rsweep",
            timeout=per_iter_timeout or 3600,
            quiet=True,
        )

    duration = time.time() - start_time

    # Log output
    (run_dir / "transferbench_output.log").write_text(
        f"=== TransferBench rsweep (exit_code={rc}) ===\n" f"--- stdout ---\n{stdout}\n" f"--- stderr ---\n{stderr}\n"
    )
    logger.info(
        "[transferbench] rsweep finished (rc=%d, %.1fs)",
        rc,
        duration,
    )

    dmesg_after = capture_dmesg()
    dmesg_new = dmesg_delta(dmesg_before, dmesg_after)

    if dmesg_before:
        (run_dir / "dmesg_pretest.log").write_text(dmesg_before)
    if dmesg_new:
        (run_dir / "dmesg.log").write_text(dmesg_new)

    # Handle watchdog timeout
    timed_out = (rc == 124 or rc == -9) and per_iter_timeout is not None
    if timed_out:
        stdout += (
            f"\n  [transferbench] FAIL: watchdog timeout — rsweep did " f"not complete within {per_iter_timeout}s."
        )

    exit_code = rc if rc >= 0 else 1

    validation_failed, validation_msg = validate_transferbench_result(
        stdout,
        stderr,
        exit_code,
        dmesg_new,
        timed_out=timed_out,
    )

    (run_dir / "validation.txt").write_text(validation_msg + "\n")

    timeout_part = f"timeout {per_iter_timeout} " if per_iter_timeout else ""
    reproduce_cmd = (
        f"SWEEP_TIME_LIMIT={sweep_time_limit} "
        f"SWEEP_MIN={sweep_min} SWEEP_MAX={sweep_max} "
        f"{timeout_part}{bin_path} rsweep"
    )

    test_name = "gpu_transferbench_monitored"
    summary = {
        "test_name": test_name,
        "passed": not validation_failed,
        "exit_code": exit_code,
        "duration_sec": round(duration, 2),
        "timed_out": timed_out,
        "sweep_time_limit": sweep_time_limit,
        "sweep_min": sweep_min,
        "sweep_max": sweep_max,
        "samples_collected": count_csv_samples(run_dir / "power_temp.csv"),
        "validation": validation_msg,
        "reproduce_cmd": reproduce_cmd,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    analyze_and_write(run_dir, test_name)

    write_report(
        run_dir=run_dir,
        test_name=test_name,
        exit_code=exit_code,
        duration=int(duration),
    )

    logger.info("Validation:\n%s", validation_msg)
    logger.info("Report: %s", run_dir / "report.html")

    assert not validation_failed, f"TransferBench validation failed:\n{validation_msg}"
