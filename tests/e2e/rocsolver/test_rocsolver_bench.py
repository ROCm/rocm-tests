# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""rocSOLVER benchmark stress test.

Runs rocsolver-bench (gesvd) in a timed loop to stress-test the GPU,
validating output for correctness markers and checking dmesg for
kernel-level faults.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import time

import pytest

from framework.executors.local_executor import run_cmd_get_stdout_stderr
from tests.common.gpu_monitored.validation import (
    capture_dmesg,
    dmesg_delta,
)

logger = logging.getLogger(__name__)

_BENCH_CMD_ARGS = [
    "-f",
    "gesvd",
    "--precision",
    "d",
    "--left_svect",
    "S",
    "--right_svect",
    "S",
    "-m",
    "250",
    "-n",
    "250",
]

_PASS_MARKERS = ("cpu_time_us", "gpu_time_us")

_FAIL_PATTERNS = (
    "Error",
    "error",
    "crash",
    "Core dump",
    "Fault",
    "Memory access fault",
    "abort",
    "No such file",
    "reboot",
    "hang",
    "hung",
    "interrupt",
    "panic",
    "stuck",
    "memleak",
    "Memory corruption",
    "out-of-bound",
    "Out of memory",
)


def _find_rocsolver_bench(rock_dir: str) -> Path | None:
    """Locate rocsolver-bench binary in the ROCm install."""
    candidate = Path(rock_dir) / "bin" / "rocsolver-bench"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return candidate
    return None


def _check_output_for_errors(stdout: str, stderr: str) -> tuple[bool, str]:
    """Check rocsolver-bench output for pass/fail indicators.

    Returns (passed, message).
    """
    combined = stdout + "\n" + stderr
    lines = combined.splitlines()

    # Check for failure patterns
    fail_lines = []
    for line in lines:
        for pat in _FAIL_PATTERNS:
            if pat in line:
                fail_lines.append(line.strip()[:150])
                break

    if fail_lines:
        msg = (
            f"FAIL: {len(fail_lines)} error line(s) detected\n"
            + "\n".join(f"  -> {fl}" for fl in fail_lines[:5])
        )
        return False, msg

    # Check for pass markers
    has_pass = all(marker in combined for marker in _PASS_MARKERS)
    if not has_pass:
        return False, "FAIL: pass markers (cpu_time_us, gpu_time_us) not found in output"

    return True, "PASS: rocsolver-bench completed with expected output markers"


@pytest.mark.runtime.medium
def test_rocsolver_bench(
    rock_dir,
    ld_path,
    request,
    framework_config,
):
    """Stress-test GPU with rocsolver-bench gesvd in a timed loop."""
    # Artifact directory
    test_name = re.sub(r"[^A-Za-z0-9._=-]+", "_", request.node.name).strip("._")
    run_dir = Path(framework_config.framework.artifact_dir) / "rocsolver" / test_name
    run_dir.mkdir(parents=True, exist_ok=True)
    bin_path = _find_rocsolver_bench(rock_dir)
    if bin_path is None:
        pytest.skip("rocsolver-bench binary not found (rocsolver-clients not installed)")

    rocm_lib = os.path.join(rock_dir, "lib")
    ld = ld_path["LD_LIBRARY_PATH"]
    if rocm_lib not in ld:
        ld = f"{rocm_lib}:{ld}"
    os.environ["LD_LIBRARY_PATH"] = ld

    duration_min = int(os.environ.get("ROCSOLVER_BENCH_DURATION_MIN", "1"))
    duration_sec = duration_min * 60

    logger.info(
        "Running rocsolver-bench gesvd stress test for %d minute(s)",
        duration_min,
    )

    dmesg_before = capture_dmesg()

    deadline = time.monotonic() + duration_sec
    iteration = 0
    all_passed = True
    all_messages: list[str] = []
    first_fail_output = ""
    console_log = run_dir / "console_output.log"

    with console_log.open("w") as log_fh:
        while time.monotonic() < deadline:
            iteration += 1
            rc, stdout, stderr = run_cmd_get_stdout_stderr(
                str(bin_path),
                *_BENCH_CMD_ARGS,
                timeout=300,
                quiet=True,
            )

            # Write full console output for this iteration
            log_fh.write(
                f"=== Iteration {iteration} (exit_code={rc}) ===\n"
                f"{stdout}\n"
            )
            if stderr.strip():
                log_fh.write(f"--- stderr ---\n{stderr}\n")
            log_fh.write("\n")
            log_fh.flush()

            if rc != 0:
                msg = f"Iteration {iteration}: non-zero exit code {rc}"
                logger.warning("[rocsolver-bench] %s", msg)
                all_messages.append(msg)
                if all_passed:
                    first_fail_output = stdout + "\n" + stderr
                all_passed = False
                continue

            passed, msg = _check_output_for_errors(stdout, stderr)
            if not passed:
                logger.warning("[rocsolver-bench] Iteration %d: %s", iteration, msg)
                all_messages.append(f"Iteration {iteration}: {msg}")
                if all_passed:
                    first_fail_output = stdout + "\n" + stderr
                all_passed = False
            else:
                all_messages.append(f"Iteration {iteration}: PASS")

    logger.info(
        "[rocsolver-bench] Completed %d iteration(s) in %d seconds",
        iteration, duration_sec,
    )

    # Check dmesg for kernel-level issues
    dmesg_after = capture_dmesg()
    dmesg_new = dmesg_delta(dmesg_before, dmesg_after)

    # Save dmesg artifacts
    if dmesg_before:
        (run_dir / "dmesg_pretest.log").write_text(dmesg_before)
    if dmesg_new:
        (run_dir / "dmesg.log").write_text(dmesg_new)

    dmesg_clean = True
    if dmesg_new:
        dmesg_clean = False
        logger.error(
            "[rocsolver-bench] dmesg has new entries during test — "
            "check dmesg.log artifact for details",
        )

    pass_count = sum(1 for m in all_messages if "PASS" in m)
    fail_count = iteration - pass_count

    # Save results log
    (run_dir / "rocsolver_bench_results.txt").write_text(
        "\n".join(all_messages) + "\n"
    )

    summary = (
        f"rocsolver-bench gesvd: {iteration} iteration(s), "
        f"{pass_count} passed, {fail_count} failed, "
        f"dmesg {'clean' if dmesg_clean else 'HAS CRITICAL EVENTS'}"
    )
    logger.info("[rocsolver-bench] %s", summary)

    assert all_passed, (
        f"rocsolver-bench failed:\n{summary}\n"
        + "\n".join(m for m in all_messages if "FAIL" in m)[:500]
    )
    assert dmesg_clean, (
        "dmesg has new entries during rocsolver-bench — "
        "check dmesg.log artifact for details"
    )
