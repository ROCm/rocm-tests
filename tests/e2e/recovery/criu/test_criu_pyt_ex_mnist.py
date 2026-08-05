# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the upstream PyTorch MNIST example.

Launch main.py under the container's ambient ROCm PyTorch, ``criu dump`` then ``criu restore`` the
running training process, and verify it resumes and keeps making forward progress (does not hang).
"""

from __future__ import annotations

import logging
import re

import pytest
from tests.common.criu import steps as criu

from framework.reporting.allure_reporter import report_metric, step

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of this workload.
SUPPORTED_ARCHS = frozenset(
    {"gfx90a", "gfx908", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"}
)

# Training stdout is captured here; the forward-progress check measures its byte growth.
_TRAIN_MARKER = "Train Epoch"
_TRAIN_TIMEOUT = 600.0  # dataset download + first training batches
_PROGRESS_TIMEOUT = 300.0  # post-restore forward-progress window
_LAUNCH_TIMEOUT = 60.0


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    """Skip when ``--gpu-arch`` names a GFX target outside SUPPORTED_ARCHS."""
    if gpu_arch and gpu_arch not in SUPPORTED_ARCHS:
        pytest.skip(f"GPU arch {gpu_arch} not supported for CRIU MNIST: {sorted(SUPPORTED_ARCHS)}")


def _launch(executor, setup) -> str:
    """Launch MNIST main.py under the ambient ROCm python; return its PID."""
    with step("Launch PyTorch MNIST training"):
        # Capture the PID via $! -- `grep main.py` would also match the pytest argv.
        cmd = (
            f"cd {setup.workdir} && rm -f training.out dump.log restore.log *.img 2>/dev/null; "
            f"nohup {setup.python} main.py > training.out 2>&1 & pid=$!; disown 2>/dev/null || true; "
            "sleep 5; "
            'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 training.out; fi'
        )
        out = executor.run(cmd, timeout=_LAUNCH_TIMEOUT).stdout or ""
        pid = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")), "")
        assert pid, f"MNIST training did not start:\n{out[-1500:]}"
        logger.info("MNIST main.py running with PID %s", pid)
        return pid


def _wait_for_training(executor, setup) -> int:
    """Poll training.out for the first ``Train Epoch`` line; return its byte size then."""
    with step("Wait for MNIST training to start computing"):
        poll = (
            f"for i in $(seq 1 {int(_TRAIN_TIMEOUT // 2)}); do "
            f"if grep -q '{_TRAIN_MARKER}' {setup.workdir}/training.out 2>/dev/null; then "
            f"echo TRAIN_STARTED; echo BYTES=$(stat -c %s {setup.workdir}/training.out 2>/dev/null); break; fi; "
            "sleep 2; done"
        )
        out = executor.run(poll, timeout=_TRAIN_TIMEOUT + 60).stdout or ""
        assert "TRAIN_STARTED" in out, f"MNIST training never started within {_TRAIN_TIMEOUT:.0f}s:\n{out[-1500:]}"
        size = next((int(m.group(1)) for line in out.splitlines() if (m := re.match(r"BYTES=(\d+)", line.strip()))), 0)
        report_metric("MNIST_OUTPUT_BYTES_PRE_DUMP", float(size), "bytes")
        logger.info("MNIST training started; training.out is %d bytes pre-dump", size)
        return size


def _checkpoint(executor, criu_cmd: str, setup, pid: str, full_log: bool = False) -> None:
    """Checkpoint the training process with ``criu dump`` and assert it stopped."""
    with step("Checkpoint with criu dump"):
        dump = criu.criu_dump(executor, criu_cmd, setup.workdir, pid)
        log = criu.attach_criu_log(executor, setup.workdir, "dump.log", full=full_log)
        assert "OK" in dump.stdout, f"criu dump did not report OK:\n{dump.stdout[-1500:]}\n{log[-1500:]}"
        assert "PID_GONE" in dump.stdout, "training process still exists after criu dump."


def _restore(executor, criu_cmd: str, setup, full_log: bool = False) -> None:
    """Restore the checkpoint with ``criu restore`` and assert CRIU reported success."""
    with step("Restore with criu restore"):
        restore = criu.criu_restore(executor, criu_cmd, setup.workdir)
        log = criu.attach_criu_log(executor, setup.workdir, "restore.log", full=full_log)
        assert "RESTORE_OK" in restore.stdout, f"criu restore did not finish successfully:\n{log[-1500:]}"


def _wait_for_progress(executor, setup, baseline: int) -> int:
    """Poll until training.out grows beyond *baseline*; return the grown size (0 if it never does)."""
    with step("Verify training output grows after restore"):
        poll = (
            f"for i in $(seq 1 {int(_PROGRESS_TIMEOUT // 2)}); do "
            f"sz=$(stat -c %s {setup.workdir}/training.out 2>/dev/null || echo 0); "
            f'if [ "$sz" -gt {baseline} ]; then echo GREW=$sz; break; fi; '
            "sleep 2; done"
        )
        out = executor.run(poll, timeout=_PROGRESS_TIMEOUT + 60).stdout or ""
        return next((int(m.group(1)) for line in out.splitlines() if (m := re.match(r"GREW=(\d+)", line.strip()))), 0)


@pytest.mark.container(ipc="host", privileged=True)
@pytest.mark.runtime.medium
@pytest.mark.xdist_group("criu_serial")
def test_criu_pyt_ex_mnist(target_executor, pyt_mnist_setup, criu_runtime_target, gpu_arch, request):
    """Checkpoint/restore a live MNIST training process and confirm it resumes without hanging.

    Runs inside a privileged ROCm PyTorch container: launch -> wait for compute -> criu dump ->
    criu restore -> assert the PID is alive, still running main.py, and its output keeps growing.
    """
    _skip_if_unsupported_arch(gpu_arch)
    executor = target_executor
    setup = pyt_mnist_setup
    criu_cmd = criu_runtime_target
    full_log = request.config.getoption("capture") == "no"

    pid = _launch(executor, setup)
    try:
        pre_size = _wait_for_training(executor, setup)
        _checkpoint(executor, criu_cmd, setup, pid, full_log)
        _restore(executor, criu_cmd, setup, full_log)

        with step("Verify MNIST resumed and keeps making progress"):
            # criu restores into the original PID; confirm it is alive and still the workload.
            check = executor.run(
                f"if ps -p {pid} >/dev/null 2>&1 && "
                f'tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null | grep -q main.py; '
                "then echo RESUMED_OK; else echo RESUMED_NO; fi"
            )
            assert "RESUMED_OK" in check.stdout, f"MNIST did not resume under criu (PID {pid})."

            # A hung process stays alive but its output never grows -- require real forward progress.
            grown = _wait_for_progress(executor, setup, pre_size)
            report_metric("MNIST_OUTPUT_BYTES_POST_RESTORE", float(grown), "bytes")
            assert grown > pre_size, (
                f"MNIST restored but training.out never grew beyond {pre_size} bytes -- "
                "the process appears hung, not resumed."
            )
    finally:
        criu.kill_pid(executor, pid)
