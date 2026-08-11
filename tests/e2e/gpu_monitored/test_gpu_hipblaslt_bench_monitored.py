# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored hipBLASLt GEMM benchmark sweep with full analysis pipeline.

Runs hipblaslt-bench shapes under continuous amd-smi monitoring, then performs
validation, anomaly detection, and generates an HTML report.

Shapes are derived from ROCmTestInternal's hipBLASLt_GEMM.sh.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time

import pytest

from tests.common.gpu_monitored.analysis import analyze_and_write
from tests.common.gpu_monitored.monitoring import count_csv_samples
from tests.common.gpu_monitored.report import write_report
from tests.common.gpu_monitored.validation import (
    capture_dmesg,
    dmesg_delta,
    pretest_health_probe,
    validate_hipblaslt_result,
)

logger = logging.getLogger(__name__)

# GEMM shapes: (M, N, K, batch_count)
NN_SHAPES: list[tuple[int, int, int, int]] = [
    (8192, 320, 320, 1),
    (2048, 640, 640, 1),
    (512, 1280, 1280, 1),
    (8192, 320, 1280, 1),
    (512, 10240, 1280, 1),
    (2048, 5120, 640, 1),
    (8192, 2560, 320, 1),
    (512, 1280, 5120, 1),
    (2048, 640, 2560, 1),
    (154, 320, 768, 1),
    (154, 1280, 768, 1),
    (4096, 40, 4096, 16),
    (1024, 80, 1024, 16),
    (1024, 80, 77, 16),
]

NT_SHAPES: list[tuple[int, int, int, int]] = [
    (4096, 4096, 40, 16),
    (1024, 1024, 80, 16),
    (4096, 77, 40, 16),
    (256, 77, 160, 16),
    (1024, 77, 80, 16),
]

ITERS = 600
COLD_ITERS = 10


def _find_hipblaslt_bench(rock_dir: str) -> Path | None:
    """Locate hipblaslt-bench binary from the ROCm install."""
    installed = Path(rock_dir) / "bin" / "hipblaslt-bench"
    if installed.is_file() and os.access(installed, os.X_OK):
        return installed
    return None


def _run_shape(
    bench: Path,
    trans_a: str,
    trans_b: str,
    m: int,
    n: int,
    k: int,
    batch: int,
    env: dict,
    shape_num: int,
    total_shapes: int,
    print_header: bool,
    cwd: Path | None = None,
    timeout: int | None = None,
) -> tuple[bool, str]:
    """Run a single GEMM shape. Returns (success, output_line).

    Leading dimensions follow BLAS convention:
        transA == "N" -> A stored MxK -> lda = M
        transA == "T" -> A stored KxM -> lda = K
        transB == "N" -> B stored KxN -> ldb = K
        transB == "T" -> B stored NxK -> ldb = N
    """
    lda = m if trans_a == "N" else k
    ldb = k if trans_b == "N" else n
    common_args = [
        "--precision",
        "f16_r",
        "--compute_type",
        "f32_r",
        "--activation_type",
        "none",
        "--iters",
        str(ITERS),
        "--cold_iters",
        str(COLD_ITERS),
        "--alpha",
        "1",
        "--beta",
        "0",
    ]
    cmd = [
        str(bench),
        "-v",
        "--transA",
        trans_a,
        "--transB",
        trans_b,
        "-m",
        str(m),
        "-n",
        str(n),
        "-k",
        str(k),
        "--lda",
        str(lda),
        "--stride_a",
        str(m * k),
        "--ldb",
        str(ldb),
        "--stride_b",
        str(k * n),
        "--ldc",
        str(m),
        "--stride_c",
        str(m * n),
        "--ldd",
        str(m),
        "--stride_d",
        str(m * n),
        *common_args,
        "--batch_count",
        str(batch),
    ]
    try:
        res = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        msg = (
            f"NO DATA: shape {shape_num}/{total_shapes} "
            f"({trans_a}{trans_b} {m}x{n}x{k}x{batch}) produced no data row"
        )
        logger.warning("[hipblaslt] %s", msg)
        if e.stdout:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else e.stdout
            logger.warning("%s", out[-1000:])
        if e.stderr:
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr
            logger.warning("%s", err[-1000:])
        return False, msg

    if res.returncode != 0:
        msg = (
            f"NO DATA: shape {shape_num}/{total_shapes} "
            f"({trans_a}{trans_b} {m}x{n}x{k}x{batch}) produced no data row"
        )
        logger.warning("[hipblaslt] %s", msg)
        out_lines = (res.stdout + res.stderr).splitlines()[-5:]
        for line in out_lines:
            logger.warning("  %s", line)
        return False, msg

    header_line = None
    data_line = None
    for line in (res.stdout + res.stderr).splitlines():
        if header_line is None and re.match(r"^\[\d+\]:transA", line):
            header_line = line
        if data_line is None and re.match(r"^\s+[NT],[NT],\d", line):
            data_line = line

    if print_header and header_line:
        logger.info("%s", header_line)
    if data_line:
        logger.info("%s", data_line)
        return True, data_line.strip()

    msg = f"NO DATA: shape {shape_num}/{total_shapes} " f"({trans_a}{trans_b} {m}x{n}x{k}x{batch}) produced no data row"
    logger.warning("[hipblaslt] %s", msg)
    return False, msg


@pytest.mark.runtime.medium
def test_gpu_hipblaslt_bench_monitored(
    gpu_monitor,
    run_dir,
    rock_dir,
    ld_path,
    kernel_health_probe,
):
    """Run hipBLASLt GEMM sweep with GPU monitoring and full validation."""
    kernel_health_probe(lookback_sec=300, strict=False)

    # Write pretest_health.json artifact
    _clean, health_summary = pretest_health_probe(lookback_min=5)
    with open(run_dir / "pretest_health.json", "w") as f:
        json.dump(health_summary, f, indent=2)

    bench = _find_hipblaslt_bench(rock_dir)
    if bench is None:
        pytest.skip("hipblaslt-bench binary not found")

    rocm_lib = os.path.join(rock_dir, "lib")
    ld = ld_path["LD_LIBRARY_PATH"]
    if rocm_lib not in ld:
        ld = f"{rocm_lib}:{ld}"

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ld

    shape_timeout = int(os.environ.get("HIPBLASLT_SHAPE_TIMEOUT", "0")) or None

    total_shapes = len(NN_SHAPES) + len(NT_SHAPES)
    logger.info(
        "Running %d GEMM shapes (%d NN + %d NT), %d iters each",
        total_shapes,
        len(NN_SHAPES),
        len(NT_SHAPES),
        ITERS,
    )

    # Build reproduce command for first shape
    timeout_part = f"timeout {shape_timeout} " if shape_timeout else ""
    reproduce_cmd = (
        f"{timeout_part}{bench} -v --transA N --transB N "
        f"-m 8192 -n 320 -k 320 "
        f"--precision f16_r --compute_type f32_r --activation_type none "
        f"--iters {ITERS} --cold_iters {COLD_ITERS} "
        f"--alpha 1 --beta 0 --batch_count 1"
    )

    start_time = time.time()
    passed = 0
    failed = 0
    results_log: list[str] = []
    all_stdout: list[str] = []
    all_stderr: list[str] = []

    # Layer 3 prep: capture dmesg before workload
    dmesg_before = capture_dmesg()

    with gpu_monitor(cu_occupancy=False):
        shape_num = 0

        for m, n, k, batch in NN_SHAPES:
            shape_num += 1
            ok, line = _run_shape(
                bench,
                "N",
                "N",
                m,
                n,
                k,
                batch,
                env,
                shape_num,
                total_shapes,
                print_header=(shape_num == 1),
                cwd=run_dir,
                timeout=shape_timeout,
            )
            results_log.append(f"[{shape_num}/{total_shapes}] {line}")
            all_stdout.append(line)
            if ok:
                passed += 1
            else:
                failed += 1
                all_stderr.append(line)

        for m, n, k, batch in NT_SHAPES:
            shape_num += 1
            ok, line = _run_shape(
                bench,
                "N",
                "T",
                m,
                n,
                k,
                batch,
                env,
                shape_num,
                total_shapes,
                print_header=False,
                cwd=run_dir,
                timeout=shape_timeout,
            )
            results_log.append(f"[{shape_num}/{total_shapes}] {line}")
            all_stdout.append(line)
            if ok:
                passed += 1
            else:
                failed += 1
                all_stderr.append(line)

    duration = time.time() - start_time

    # Layer 3: capture dmesg after workload
    dmesg_after = capture_dmesg()
    dmesg_new = dmesg_delta(dmesg_before, dmesg_after)

    # Store dmesg snapshots as artifacts
    if dmesg_before:
        (run_dir / "dmesg_pretest.log").write_text(dmesg_before)
    if dmesg_new:
        (run_dir / "dmesg.log").write_text(dmesg_new)

    # 5-layer validation on combined output
    combined_stdout = "\n".join(all_stdout)
    combined_stderr = "\n".join(all_stderr)
    exit_code = 1 if failed > 0 else 0

    validation_failed, validation_msg = validate_hipblaslt_result(
        combined_stdout,
        combined_stderr,
        exit_code,
        dmesg_new,
        shapes_passed=passed,
        shapes_failed=failed,
    )

    (run_dir / "hipblaslt_results.txt").write_text("\n".join(results_log) + "\n")
    (run_dir / "validation.txt").write_text(validation_msg + "\n")

    test_name = "gpu_hipblaslt_bench_monitored"
    summary = {
        "test_name": test_name,
        "passed": not validation_failed,
        "exit_code": exit_code,
        "duration_sec": round(duration, 2),
        "shapes_total": total_shapes,
        "shapes_passed": passed,
        "shapes_failed": failed,
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
    logger.info("Completed: %d/%d shapes passed, %d failed (%.1fs)", passed, total_shapes, failed, duration)
    logger.info("Report: %s", run_dir / "report.html")

    assert not validation_failed, f"hipBLASLt GEMM sweep validation failed:\n{validation_msg}\n" + "\n".join(
        line for line in results_log if "FAIL" in line or "TIMEOUT" in line or "NO DATA" in line
    )
