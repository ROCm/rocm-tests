# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Reusable CRIU checkpoint/restore steps shared across the CRIU test suite.

Generic ``criu dump`` / ``criu restore`` command builders plus small helpers for attaching
CRIU logs and killing a tracked PID. Workload-specific setup lives in each test file.
"""

from __future__ import annotations

from framework.reporting.allure_reporter import attach_text

# criu dump can be slow for large VRAM; restore is backgrounded and its log polled.
DUMP_TIMEOUT = 300.0
RESTORE_TIMEOUT = 1200.0
_RESTORE_MARKER = "Restore finished successfully"


def criu_dump(executor, criu: str, workdir: str, pid: str, log: str = "dump.log", timeout: float = DUMP_TIMEOUT):
    """Checkpoint *pid*; the result carries an ``OK`` and a ``PID_GONE``/``PID_ALIVE`` sentinel."""
    cmd = (
        f"cd {workdir} && {criu} dump -t {pid} -j -vvv -o {log} --link-remap --file-lock && echo OK; "
        f"if ps -p {pid} >/dev/null 2>&1; then echo PID_ALIVE; else echo PID_GONE; fi"
    )
    return executor.run(cmd, timeout=timeout)


def criu_restore(executor, criu: str, workdir: str, log: str = "restore.log", timeout: float = RESTORE_TIMEOUT):
    """Restore in the background and poll *log* for the success marker; result carries ``RESTORE_OK``.

    ``criu restore`` (no ``-d``) waits on the restored process, so it is backgrounded and the
    log polled rather than blocking until the workload exits.
    """
    start = (
        f"cd {workdir} && rm -f {log} 2>/dev/null; "
        f"nohup {criu} restore -vvv -o {log} --shell-job --link-remap --file-lock "
        "> criu_restore.nohup 2>&1 & disown 2>/dev/null || true; echo STARTED"
    )
    executor.run(start, timeout=60)
    poll = (
        f"for i in $(seq 1 {int(timeout // 2)}); do "
        f"if sudo -n grep -q '{_RESTORE_MARKER}' {workdir}/{log} 2>/dev/null; then echo RESTORE_OK; break; fi; "
        "sleep 2; done"
    )
    return executor.run(poll, timeout=timeout + 60)


def attach_criu_log(executor, workdir: str, log: str) -> str:
    """Read a root-owned CRIU log and attach it to the Allure report; return its text."""
    text = executor.run(f"sudo -n cat {workdir}/{log} 2>/dev/null").stdout or ""
    attach_text(text, name=log)
    return text


def kill_pid(executor, pid: str | None) -> None:
    """Best-effort ``kill -9`` of an exact PID (never by name, which could match the runner)."""
    if pid:
        executor.run(f"sudo -n kill -9 {pid} 2>/dev/null; true")
