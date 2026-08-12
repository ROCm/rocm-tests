# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GPU-monitored cudamemtest with full analysis pipeline.

Runs cuda_memtest sub-tests (1..10) under continuous amd-smi monitoring,
then performs 5-layer validation, anomaly detection, and generates an
HTML report.

The cuda_memtest binary is built from source (cloned, hipified, compiled
with hipcc) if not already present.
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
    validate_cudamemtest_result,
)

logger = logging.getLogger(__name__)

SUBTESTS = list(range(1, 11))
NUM_PASSES = 1
_REPO_URL = "https://github.com/ComputationalRadiationPhysics/cuda_memtest.git"
_PINNED_COMMIT = "0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d"


def _build_cuda_memtest(rock_dir: str, build_dir: Path) -> Path | None:
    """Build cuda_memtest from source using hipify + hipcc.

    Returns path to the built binary, or None on failure.
    """
    src_dir = build_dir / "cuda_memtest"
    binary = src_dir / "cuda_memtest"

    if binary.is_file():
        logger.info("cuda_memtest binary already built: %s", binary)
        return binary

    logger.info("Building cuda_memtest from source in %s", src_dir)

    try:
        clone_repo(_REPO_URL, src_dir, ref=_PINNED_COMMIT)
    except Exception as e:
        logger.error("cuda_memtest: git clone failed: %s", e)
        return None

    # Reset to pinned commit (clone_repo may have checked out ref already,
    # but be defensive for shallow/partial clones)
    rc, _out, _err = run_cmd_get_stdout_stderr(
        "git",
        "reset",
        "--hard",
        _PINNED_COMMIT,
        cwd=str(src_dir),
        timeout=30,
        quiet=True,
    )
    if rc != 0:
        logger.warning(
            "cuda_memtest: git reset to %s failed (rc=%d); " "building against current HEAD",
            _PINNED_COMMIT[:12],
            rc,
        )

    binary.unlink(missing_ok=True)

    # Hipify .cu / .cpp files
    hipify = os.path.join(rock_dir, "bin", "hipify-perl")
    for pattern in ("cuda_memtest.*", "misc.*", "tests.cu"):
        for f in src_dir.glob(pattern):
            if not f.is_file():
                continue
            tmp = src_dir / f"hip_{f.name}"
            hip_rc, hip_stdout, _hip_err = run_cmd_get_stdout_stderr(
                hipify,
                str(f),
                timeout=60,
                quiet=True,
            )
            if hip_rc == 0 and hip_stdout:
                tmp.write_text(hip_stdout)
                f.unlink()
                tmp.rename(f)

    # Patch header for HIP build
    header = src_dir / "cuda_memtest.h"
    if header.is_file():
        content = header.read_text()
        new = content.replace(
            "MEMTEST_PP_CONCAT_DO(cuda, name)",
            "MEMTEST_PP_CONCAT_DO(hip, name)",
        )
        if new != content:
            header.write_text(new)

    # Patch hipHostGetDevicePointer call (requires void** cast)
    for src_name in ("cuda_memtest.cu", "cuda_memtest.cpp"):
        src = src_dir / src_name
        if src.is_file():
            content = src.read_text()
            new = content.replace(
                "hipHostGetDevicePointer(",
                "hipHostGetDevicePointer((void **)",
            )
            if new != content:
                src.write_text(new)
            break

    # Compile with hipcc — set ROCM_PATH so hipcc resolves the correct
    # clang++ and rocm_agent_enumerator from this rock_dir install.
    hipcc = os.path.join(rock_dir, "bin", "hipcc")
    build_env = dict(os.environ)
    build_env["ROCM_PATH"] = rock_dir
    build_env["HIP_PATH"] = rock_dir
    build_env["HIP_CLANG_PATH"] = os.path.join(rock_dir, "lib", "llvm", "bin")

    cu_file = src_dir / "cuda_memtest.cu"
    if cu_file.is_file():
        srcs = "cuda_memtest.cu misc.cpp tests.cu"
    else:
        srcs = "cuda_memtest.cpp misc.cpp tests.cpp"

    build_cmd = f"{hipcc} -DENABLE_NVML=0 {srcs} -o cuda_memtest"
    build_rc, _build_out, build_err = run_cmd_get_stdout_stderr(
        "bash",
        "-c",
        build_cmd,
        cwd=str(src_dir),
        timeout=600,
        quiet=True,
        env=build_env,
    )

    if not binary.is_file():
        logger.error(
            "cuda_memtest: build FAILED (hipcc rc=%d); stderr:\n%s",
            build_rc,
            build_err[:2000],
        )
        return None

    logger.info("cuda_memtest: build OK -> %s", binary)
    return binary


def _run_subtest(
    bin_path: Path,
    test_num: int,
    extra_args: list[str] | None = None,
    timeout: int | None = None,
) -> tuple[int, str, str]:
    """Run a single cuda_memtest sub-test.

    Returns (exit_code, stdout, stderr).
    """
    cmd_parts = [
        str(bin_path),
        "--disable_all",
        "--enable_test",
        str(test_num),
        "--num_passes",
        str(NUM_PASSES),
    ]
    if extra_args:
        cmd_parts.extend(extra_args)

    rc, stdout, stderr = run_cmd_get_stdout_stderr(
        *cmd_parts,
        timeout=timeout or 3600,
        quiet=True,
    )
    return rc, stdout, stderr


@pytest.mark.runtime.medium
def test_gpu_cudamemtest_monitored(
    gpu_monitor,
    run_dir,
    rock_dir,
    ld_path,
    kernel_health_probe,
):
    """Run cuda_memtest sub-tests with GPU monitoring and full validation."""
    kernel_health_probe(lookback_sec=300, strict=False)

    _clean, health_summary = pretest_health_probe(lookback_min=5)
    with open(run_dir / "pretest_health.json", "w") as f:
        json.dump(health_summary, f, indent=2)

    bld_env = os.environ.get("CUDA_MEMTEST_BUILD_DIR", "")
    bld = Path(bld_env) if bld_env else (run_dir / "build")
    bld.mkdir(parents=True, exist_ok=True)

    bin_path = _build_cuda_memtest(rock_dir, bld)
    if bin_path is None:
        pytest.skip("cuda_memtest binary not found and build from source failed")

    rocm_lib = os.path.join(rock_dir, "lib")
    ld = ld_path["LD_LIBRARY_PATH"]
    if rocm_lib not in ld:
        ld = f"{rocm_lib}:{ld}"

    os.environ["LD_LIBRARY_PATH"] = ld

    per_iter_timeout = int(os.environ.get("CUDAMEMTEST_PER_ITER_TIMEOUT", "0")) or None
    memtest_duration = int(os.environ.get("CUDAMEMTEST_DURATION", "3600"))
    extra_args: list[str] = []
    memtest_blocks = os.environ.get("CUDAMEMTEST_MAX_BLOCKS", "")
    if memtest_blocks:
        extra_args = ["--max_num_blocks", memtest_blocks]

    logger.info(
        "Running cuda_memtest: %d sub-tests, duration budget %ds, " "per-iter timeout %s",
        len(SUBTESTS),
        memtest_duration,
        f"{per_iter_timeout}s" if per_iter_timeout else "none",
    )

    start_time = time.time()
    deadline = time.monotonic() + memtest_duration
    passed = 0
    failed = 0
    results_log: list[str] = []
    all_stdout: list[str] = []
    all_stderr: list[str] = []

    dmesg_before = capture_dmesg()

    with gpu_monitor(cu_occupancy=False):
        for test_num in SUBTESTS:
            if time.monotonic() >= deadline:
                logger.info(
                    "Time budget exhausted after %d sub-tests",
                    test_num - 1,
                )
                break

            rc, stdout, stderr = _run_subtest(
                bin_path,
                test_num,
                extra_args=extra_args,
                timeout=per_iter_timeout,
            )
            all_stdout.append(stdout)
            all_stderr.append(stderr)

            # Log per-subtest output to individual files
            subtest_log = run_dir / f"subtest_{test_num}.log"
            subtest_log.write_text(
                f"=== enable_test {test_num} (exit_code={rc}) ===\n"
                f"--- stdout ---\n{stdout}\n"
                f"--- stderr ---\n{stderr}\n"
            )
            logger.info(
                "[cudamemtest] enable_test %d finished (rc=%d)",
                test_num,
                rc,
            )

            if rc == 124 or rc == -9:
                actual_timeout = per_iter_timeout or 3600
                if per_iter_timeout:
                    msg = (
                        f"[cudamemtest] FAIL: watchdog timeout — "
                        f"enable_test {test_num} did not complete within "
                        f"{actual_timeout}s. GPU is likely wedged; "
                        f"stopping further sub-tests."
                    )
                    logger.warning("%s", msg)
                    results_log.append(f"[{test_num}/10] {msg}")
                    failed += 1
                    break
                else:
                    msg = f"[{test_num}/10] FAIL (killed after " f"{actual_timeout}s): enable_test {test_num}"
                    logger.warning("[cudamemtest] %s", msg)
                    results_log.append(msg)
                    failed += 1
            elif rc != 0:
                msg = f"[{test_num}/10] FAIL (exit {rc}): " f"enable_test {test_num}"
                logger.warning("[cudamemtest] %s", msg)
                results_log.append(msg)
                failed += 1
            else:
                msg = f"[{test_num}/10] PASS: enable_test {test_num}"
                results_log.append(msg)
                passed += 1

    duration = time.time() - start_time
    ran = passed + failed

    logger.info(
        "[cudamemtest] Ran %d/10 sub-test(s), %d passed, %d failed " "(%.1fs)",
        ran,
        passed,
        failed,
        duration,
    )
    all_stdout.append(f"  [cudamemtest] Ran {ran}/10 sub-test(s), " f"first_fail_rc={'0' if failed == 0 else '1'}\n")

    dmesg_after = capture_dmesg()
    dmesg_new = dmesg_delta(dmesg_before, dmesg_after)

    if dmesg_before:
        (run_dir / "dmesg_pretest.log").write_text(dmesg_before)
    if dmesg_new:
        (run_dir / "dmesg.log").write_text(dmesg_new)

    combined_stdout = "\n".join(all_stdout)
    combined_stderr = "\n".join(all_stderr)
    exit_code = 1 if failed > 0 else 0

    validation_failed, validation_msg = validate_cudamemtest_result(
        combined_stdout,
        combined_stderr,
        exit_code,
        dmesg_new,
        subtests_ran=passed,
        subtests_failed=failed,
    )

    (run_dir / "cudamemtest_results.txt").write_text("\n".join(results_log) + "\n")
    (run_dir / "validation.txt").write_text(validation_msg + "\n")

    timeout_part = f"timeout {per_iter_timeout} " if per_iter_timeout else ""
    extra_str = (" " + " ".join(extra_args)) if extra_args else ""
    reproduce_cmd = (
        f"# cuda_memtest cycles through enable_test 1..10 "
        f"(up to {memtest_duration}s):\n"
        f"  for n in 1 2 3 4 5 6 7 8 9 10; do\n"
        f"    {timeout_part}{bin_path} --disable_all --enable_test $n "
        f"--num_passes {NUM_PASSES}{extra_str} || exit 1;\n"
        f"  done"
    )

    test_name = "gpu_cudamemtest_monitored"
    summary = {
        "test_name": test_name,
        "passed": not validation_failed,
        "exit_code": exit_code,
        "duration_sec": round(duration, 2),
        "subtests_total": len(SUBTESTS),
        "subtests_ran": ran,
        "subtests_passed": passed,
        "subtests_failed": failed,
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

    assert not validation_failed, f"cudamemtest validation failed:\n{validation_msg}\n" + "\n".join(
        line for line in results_log if "FAIL" in line
    )
