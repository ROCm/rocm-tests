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
import re
import shlex
import time

import pytest

from framework.common.helpers import executor_log_file

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


def _fail_lines(text: str) -> list[str]:
    """Return the lines of *text* matching any failure pattern.

    One pattern list qualifies both the bench output and the dmesg window, as
    in the original test where a single ``fail_data`` set was passed to the
    output parser and to the dmesg validator.
    """
    matched = []
    for line in text.splitlines():
        for pat in _FAIL_PATTERNS:
            if pat in line:
                matched.append(line.strip()[:150])
                break
    return matched


def _check_output_for_errors(stdout: str, stderr: str) -> tuple[bool, str]:
    """Check rocsolver-bench output for pass/fail indicators.

    Returns (passed, message).
    """
    combined = stdout + "\n" + stderr

    fail_lines = _fail_lines(combined)
    if fail_lines:
        msg = f"FAIL: {len(fail_lines)} error line(s) detected\n" + "\n".join(f"  -> {fl}" for fl in fail_lines[:5])
        return False, msg

    has_pass = all(marker in combined for marker in _PASS_MARKERS)
    if not has_pass:
        return False, "FAIL: pass markers (cpu_time_us, gpu_time_us) not found in output"

    return True, "PASS: rocsolver-bench completed with expected output markers"


_DMESG_SOURCES = (
    "dmesg -T",
    "sudo -n dmesg -T",
    # -n bounds the read: the delta only ever spans one run, not the whole boot,
    # and the framework logs command output verbatim with no way to opt out.
    "journalctl -k --no-pager -n 200",
)

# auditd records the cwd and argv of every command run on the box, so these
# lines carry arbitrary text — including this test's own paths — into the window.
_AUDIT_RE = re.compile(r"\baudit(?:\[\d+\])?:")


def _capture_dmesg(executor) -> str | None:
    """Return the target host's kernel log, or ``None`` if no source is readable.

    ``kernel.dmesg_restrict=1`` hides the ring buffer from unprivileged readers,
    so retry under sudo as the original did (``runCmd("dmesg -T",
    privilege=True)``), then fall back to the journal, which stores the same
    kernel messages and is readable by ``adm``/``systemd-journal`` members. ``-n``
    stops a node without passwordless sudo from blocking on a password prompt.
    Only when every source fails does the caller downgrade the kernel check
    rather than fail a run over it.
    """
    for cmd in _DMESG_SOURCES:
        try:
            result = executor.run(cmd, timeout=60.0)
        except Exception as exc:
            logger.warning("[rocsolver-bench] '%s' failed: %s", cmd, exc)
            continue
        if result.ok and result.stdout.strip():
            return result.stdout
    return None


def _kernel_health_lines(lines: list[str]) -> list[str]:
    """Drop audit records, keeping only lines that say something about the kernel.

    Audit noise is not kernel health, and leaving it in lets an unrelated
    process — or this test's own sudo probe — trip a fail pattern via the
    command line auditd embeds. The artifact still records the full window.
    """
    return [line for line in lines if not _AUDIT_RE.search(line)]


def _dmesg_delta(before: str, after: str) -> list[str]:
    """Return the lines ``after`` gained relative to ``before``."""
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    if not before_lines:
        return after_lines
    anchor = before_lines[-1]
    for idx in range(len(after_lines) - 1, -1, -1):
        if after_lines[idx] == anchor:
            return after_lines[idx + 1 :]
    # The ring buffer wrapped past the anchor, so fall back to a line-set
    # difference instead of reporting the whole snapshot as new.
    seen = set(before_lines)
    return [line for line in after_lines if line not in seen]


@pytest.mark.runtime.medium
def test_rocsolver_bench(
    target_executor,
    rocsolver_bench_binary: str,
    ld_path: dict,
    request,
    framework_config,
):
    """Stress-test GPU with rocsolver-bench gesvd in a timed loop."""
    console_log = executor_log_file(
        framework_config.framework.artifact_dir,
        request.node.name,
        request.node.nodeid,
    )
    run_dir = console_log.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_stem = console_log.stem

    ld = ld_path["LD_LIBRARY_PATH"]
    bench_cmd = " ".join(
        [
            f"env LD_LIBRARY_PATH={shlex.quote(ld)}",
            shlex.quote(rocsolver_bench_binary),
            *_BENCH_CMD_ARGS,
        ]
    )

    duration_min = int(os.environ.get("ROCSOLVER_BENCH_DURATION_MIN", "1"))
    duration_sec = duration_min * 60

    logger.info(
        "Running rocsolver-bench gesvd stress test for %d minute(s)",
        duration_min,
    )

    dmesg_before = _capture_dmesg(target_executor)

    deadline = time.monotonic() + duration_sec
    iteration = 0
    all_passed = True
    all_messages: list[str] = []

    with console_log.open("w") as log_fh:
        while time.monotonic() < deadline:
            iteration += 1
            result = target_executor.run(bench_cmd, timeout=300.0)

            # Write full console output for this iteration
            log_fh.write(f"=== Iteration {iteration} (exit_code={result.exit_code}) ===\n{result.stdout}\n")
            if result.stderr.strip():
                log_fh.write(f"--- stderr ---\n{result.stderr}\n")
            log_fh.write("\n")
            log_fh.flush()

            if not result.ok:
                msg = f"Iteration {iteration}: non-zero exit code {result.exit_code}"
                logger.warning("[rocsolver-bench] %s", msg)
                all_messages.append(msg)
                all_passed = False
                continue

            passed, msg = _check_output_for_errors(result.stdout, result.stderr)
            if not passed:
                logger.warning("[rocsolver-bench] Iteration %d: %s", iteration, msg)
                all_messages.append(f"Iteration {iteration}: {msg}")
                all_passed = False
            else:
                all_messages.append(f"Iteration {iteration}: PASS")

    logger.info(
        "[rocsolver-bench] Completed %d iteration(s) in %d seconds",
        iteration,
        duration_sec,
    )

    # Check dmesg for kernel-level issues
    dmesg_after = _capture_dmesg(target_executor)
    dmesg_available = dmesg_before is not None and dmesg_after is not None
    dmesg_fail_lines: list[str] = []

    if dmesg_available:
        new_lines = _dmesg_delta(dmesg_before, dmesg_after)
        dmesg_fail_lines = _fail_lines("\n".join(_kernel_health_lines(new_lines)))
        (run_dir / f"{artifact_stem}_dmesg_pretest.log").write_text(dmesg_before)
        if new_lines:
            (run_dir / f"{artifact_stem}_dmesg.log").write_text("\n".join(new_lines) + "\n")
    else:
        logger.warning(
            "[rocsolver-bench] kernel ring buffer unreadable — skipping the dmesg check",
        )

    if dmesg_fail_lines:
        logger.error(
            "[rocsolver-bench] dmesg logs have errors, kindly verify the dmesg log artifact",
        )

    pass_count = sum(1 for m in all_messages if "PASS" in m)
    fail_count = iteration - pass_count

    # Save results log
    (run_dir / f"{artifact_stem}_results.txt").write_text("\n".join(all_messages) + "\n")

    if not dmesg_available:
        dmesg_state = "unavailable"
    elif dmesg_fail_lines:
        dmesg_state = "HAS ERRORS"
    else:
        dmesg_state = "clean"
    summary = (
        f"rocsolver-bench gesvd: {iteration} iteration(s), "
        f"{pass_count} passed, {fail_count} failed, "
        f"dmesg {dmesg_state}"
    )
    logger.info("[rocsolver-bench] %s", summary)

    assert all_passed, f"rocsolver-bench failed:\n{summary}\n" + "\n".join(m for m in all_messages if "FAIL" in m)[:500]
    assert not dmesg_fail_lines, "dmesg has error lines during rocsolver-bench:\n" + "\n".join(
        f"  -> {line}" for line in dmesg_fail_lines[:5]
    )
