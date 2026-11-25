# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_multi_stream_queue_hang.py — multi-thread hipMemset*Async hang regression soak.

Ported from:   private source withheld
Upstream ref:  private source withheld; approval recorded
Ported on:     2026-11-18
OSS license:   MIT

Validates:
    1. Under GPU_MAX_HW_QUEUES=1 and HSA_USE_SVM=0, with elevated producer
       counts (default MSQH_THREADS/MSQH_STREAMS=64) on one AQL ring, continuous
       multi-stream hipMemset*Async rounds complete the soak without an AQL hang
       or KFD wedge (VERDICT=NOHANG). HANG and WEDGE are regressions.

Override producers via ROCM_TEST_MSQH_THREADS / ROCM_TEST_MSQH_STREAMS (1..127).

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hip_runtime/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.soak
    ci.weekly  (overrides profile ci.nightly; soak must not use ci.pr)
"""

from __future__ import annotations

import logging
import os
import shlex

import pytest

from tests.e2e.hip_runtime._multi_stream_queue_hang import (
    DEFAULT_MAX_SECONDS,
    detect_hang_cmd,
    preflight_cmd,
    round_loop_cmd,
)

logger = logging.getLogger(__name__)

_TEST_NAME = "multi_stream_queue_hang"
_MAX_SECONDS = int(os.environ.get("ROCM_TEST_MULTI_STREAM_QUEUE_HANG_MAX_SECONDS", str(DEFAULT_MAX_SECONDS)))


@pytest.mark.runtime.soak
@pytest.mark.ci.weekly
def test_multi_stream_queue_hang(
    target_executor,
    ld_path: dict,
    multi_stream_queue_hang_binary: str,
):
    """Soak under one HW queue; NOHANG is PASS, HANG/WEDGE are FAIL."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = multi_stream_queue_hang_binary

    pre = target_executor.run(preflight_cmd(binary, _TEST_NAME), timeout=60)
    assert pre.ok, (
        f"preflight failed (exit={pre.exit_code}):\n" f"stdout: {pre.stdout[:2000]}\nstderr: {pre.stderr[:500]}"
    )
    assert "PREFLIGHT_OK" in pre.stdout, f"preflight missing PREFLIGHT_OK:\n{pre.stdout[:1000]}"

    mk = target_executor.run("mktemp -d /tmp/multistream.XXXXXX", timeout=30)
    assert mk.ok, f"mktemp failed (exit={mk.exit_code}): stdout={mk.stdout[:500]} stderr={mk.stderr[:500]}"
    assert mk.stdout.strip(), f"mktemp returned empty stdout: stderr={mk.stderr[:500]}"
    logdir = mk.stdout.strip().splitlines()[-1].strip()
    runlog = f"{logdir}/runlog.txt"

    touch = target_executor.run(f"touch {shlex.quote(runlog)}", timeout=30)
    assert touch.ok, (
        f"touch runlog failed (exit={touch.exit_code}):\n" f"stdout: {touch.stdout[:500]}\nstderr: {touch.stderr[:500]}"
    )

    loop = target_executor.start_background(
        round_loop_cmd(ld, binary, runlog),
        log_path="output/artifacts/hip_runtime/multi_stream_queue_hang_loop.log",
        console_label="multi_stream_queue_hang/loop",
    )
    try:
        detect = target_executor.run(
            detect_hang_cmd(binary, runlog, _MAX_SECONDS),
            timeout=_MAX_SECONDS + 120,
        )
    finally:
        if loop.is_alive:
            loop.stop(timeout=30)
        target_executor.run(f"rm -rf {shlex.quote(logdir)}", timeout=30)

    verdict_line = ""
    for line in (detect.stdout or "").splitlines():
        if line.startswith("VERDICT="):
            verdict_line = line.strip()

    logger.info("detect exit=%s %s", detect.exit_code, verdict_line)
    assert detect.ok, (
        f"expected NOHANG (no AQL hang/wedge on current builds); got exit={detect.exit_code} "
        f"{verdict_line or 'VERDICT=<missing>'}:\n"
        f"stdout: {detect.stdout[:2000]}\nstderr: {detect.stderr[:500]}"
    )
    assert "VERDICT=NOHANG" in detect.stdout, f"expected VERDICT=NOHANG in detect output:\n{detect.stdout[:2000]}"
    assert "VERDICT=HANG" not in detect.stdout, f"unexpected AQL hang regression:\n{detect.stdout[:2000]}"
    assert "VERDICT=WEDGE" not in detect.stdout, f"unexpected KFD wedge regression:\n{detect.stdout[:2000]}"
