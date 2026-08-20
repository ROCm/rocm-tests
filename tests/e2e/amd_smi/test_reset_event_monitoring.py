# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_reset_event_monitoring.py -- amd-smi event monitoring under GPU reset.

Validates that ``amd-smi event`` reports correct PRE_RESET/POST_RESET events with no
spurious NONE spam or duplicates, both under repeated resets and concurrent monitors.
"""

from __future__ import annotations

import logging
import os
import re
import time

import pytest

logger = logging.getLogger("rocm.test")

# Small, sensible defaults (env-overridable) — GPU reset is slow and destructive.
_RESET_ITERS = int(os.environ.get("ROCM_TEST_AMDSMI_RESET_ITERS", "5"))
_RESET_COOLDOWN = int(os.environ.get("ROCM_TEST_AMDSMI_RESET_COOLDOWN", "3"))
_NUM_MONITORS = int(os.environ.get("ROCM_TEST_AMDSMI_MONITORS", "3"))
_RESET_TARGET = os.environ.get("ROCM_TEST_AMDSMI_RESET_TARGET", "all")

# amd-smi event banner/status lines that are not real events.
_BANNER_PREFIXES = ("EVENT LISTENING", "Press q", "Escape Sequence")


def _monitor_cmd(env, fifo: str) -> str:
    """amd-smi event with stdin held open via an O_RDWR FIFO so it never sees EOF."""
    # ``exec`` makes amd-smi the direct child so stop()'s SIGTERM terminates it cleanly.
    return f"exec env ROCM_PATH={env.rocm_path} {env.amd_smi} event <> {fifo}"


def _reset_cmd(env) -> str:
    """Non-interactive privileged GPU reset (destructive)."""
    return f"sudo -n {env.amd_smi} reset -G -g {_RESET_TARGET}"


def _parse_event_log(text: str) -> dict[str, int]:
    """Return event-log metrics: NONE, PRE_RESET, POST_RESET, consecutive dups, line count."""
    lines = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or any(line.startswith(p) for p in _BANNER_PREFIXES):
            continue
        lines.append(line)

    # Consecutive-duplicate group count (mirrors `uniq -d`).
    dup = 0
    prev = None
    in_group = False
    for line in lines:
        if line == prev:
            if not in_group:
                dup += 1
                in_group = True
        else:
            in_group = False
        prev = line

    return {
        "lines": len(lines),
        "none": sum(1 for line in lines if "NONE" in line),
        "pre": sum(1 for line in lines if re.search(r"(GPU_)?PRE_RESET", line)),
        "post": sum(1 for line in lines if re.search(r"(GPU_)?POST_RESET", line)),
        "dup": dup,
        "reset": sum(1 for line in lines if "RESET" in line),
    }


def _count_crashes(executor, since: str) -> int:
    """Best-effort count of segfault/coredump entries in the journal since *since*."""
    grep = "grep -ci 'segfault\\|sigsegv\\|signal 11'"
    res = executor.run(f"{{ journalctl --since '{since}' 2>/dev/null | {grep}; }} || echo 0")
    match = re.search(r"\d+", res.stdout or "")
    return int(match.group()) if match else 0


def _assert_event_log(name: str, metrics: dict[str, int]) -> None:
    """Assert the shared per-log pass criteria on parsed event metrics."""
    assert metrics["lines"] > 0, f"{name}: event log is empty -- no events captured"
    assert metrics["none"] == 0, f"{name}: found {metrics['none']} NONE event(s) -- expected 0"
    assert metrics["pre"] >= 1, f"{name}: no PRE_RESET events captured"
    assert metrics["post"] >= 1, f"{name}: no POST_RESET events captured"
    assert metrics["dup"] == 0, f"{name}: found {metrics['dup']} consecutive duplicate event group(s)"


@pytest.mark.runtime.medium
def test_repeated_reset_event_stress(target_executor, random_events_env):
    """Run N GPU reset cycles under a background event monitor; validate the event log."""
    env = random_events_env
    ex = target_executor
    fifo = f"{env.scratch_dir}/evt_fifo"
    ex.run(f"rm -f {fifo} && mkfifo {fifo}")
    since = (ex.run("date '+%Y-%m-%d %H:%M:%S'").stdout or "").strip()

    monitor = ex.start_background(
        _monitor_cmd(env, fifo),
        log_path=os.path.join("output", "artifacts", "amd_smi", "repeated_reset_events.log"),
        console_label="amdsmi/event-monitor",
    )
    try:
        time.sleep(2)
        assert monitor.is_alive, "event monitor failed to start"

        for i in range(1, _RESET_ITERS + 1):
            logger.info("reset iteration %d/%d", i, _RESET_ITERS)
            ex.run(_reset_cmd(env))
            time.sleep(_RESET_COOLDOWN)
            assert monitor.is_alive, f"event monitor crashed during iteration {i}"

        time.sleep(2)  # let final events flush before stopping
    finally:
        result = monitor.stop(timeout=15.0)

    metrics = _parse_event_log(result.stdout)
    logger.info("repeated-reset event metrics: %s", metrics)
    _assert_event_log("repeated_reset", metrics)
    assert _count_crashes(ex, since) == 0, "segfault/coredump entries detected in journal during test"


@pytest.mark.runtime.fast
def test_concurrent_event_monitoring(target_executor, random_events_env):
    """Run N concurrent event monitors, trigger one reset, validate each log independently."""
    env = random_events_env
    ex = target_executor
    since = (ex.run("date '+%Y-%m-%d %H:%M:%S'").stdout or "").strip()

    monitors = []
    try:
        for m in range(_NUM_MONITORS):
            fifo = f"{env.scratch_dir}/evt_fifo_{m}"
            ex.run(f"rm -f {fifo} && mkfifo {fifo}")
            monitors.append(
                ex.start_background(
                    _monitor_cmd(env, fifo),
                    log_path=os.path.join("output", "artifacts", "amd_smi", f"concurrent_monitor_{m}.log"),
                    console_label=f"amdsmi/event-monitor-{m}",
                )
            )

        time.sleep(3)
        assert all(mon.is_alive for mon in monitors), "one or more monitors failed to start"

        ex.run(_reset_cmd(env))
        time.sleep(5)  # let reset events propagate to every monitor

        assert all(mon.is_alive for mon in monitors), "a monitor crashed after reset (possible race condition)"
    finally:
        results = [mon.stop(timeout=15.0) for mon in monitors]

    reset_counts = []
    for idx, result in enumerate(results):
        metrics = _parse_event_log(result.stdout)
        logger.info("concurrent monitor %d metrics: %s", idx, metrics)
        _assert_event_log(f"monitor_{idx}", metrics)
        reset_counts.append(metrics["reset"])

    if len(set(reset_counts)) > 1:
        logger.warning("monitors captured differing RESET counts (timing-dependent, non-fatal): %s", reset_counts)

    assert _count_crashes(ex, since) == 0, "segfault/coredump entries detected in journal during test"
