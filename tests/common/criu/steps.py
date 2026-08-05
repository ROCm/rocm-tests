# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generic CRIU checkpoint/restore command helpers.

Executor-transparent ``criu dump`` / ``criu restore`` builders plus small helpers for
attaching CRIU logs to Allure and killing a tracked PID. Workload-specific launch and
orchestration live with each workload's test files.
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
        # criu SIGKILLs the tree after dumping; wait briefly for init to reap it (teardown of a
        # heavyweight GPU process is not instant) before reporting, so it is not mis-flagged alive.
        f"for _ in $(seq 1 20); do ps -p {pid} >/dev/null 2>&1 || break; sleep 0.5; done; "
        f"if ps -p {pid} >/dev/null 2>&1; then echo PID_ALIVE; else echo PID_GONE; fi"
    )
    return executor.run(cmd, timeout=timeout)


def criu_restore(executor, criu: str, workdir: str, log: str = "restore.log", timeout: float = RESTORE_TIMEOUT):
    """Restore in the background and poll *log* for the success marker.

    ``criu restore`` (no ``-d``) waits on the restored process, so it is backgrounded and the
    log polled rather than blocking until the workload exits. The result always carries exactly
    one sentinel: ``RESTORE_OK`` when the marker appears, or ``RESTORE_FAIL`` when the restore
    process exits (or the timeout elapses) without it -- so callers get a definite outcome
    instead of empty output to assert on.
    """
    start = (
        f"cd {workdir} && rm -f {log} 2>/dev/null; "
        f"nohup {criu} restore -vvv -o {log} --shell-job --link-remap --file-lock "
        "> criu_restore.nohup 2>&1 & echo RPID=$!; disown 2>/dev/null || true"
    )
    started = executor.run(start, timeout=60)
    rpid = next(
        (ln.split("=", 1)[1].strip() for ln in (started.stdout or "").splitlines() if ln.strip().startswith("RPID=")),
        "",
    )
    # When the restore PID is known, break early once it exits (re-checking the marker to catch
    # the marker-then-exit race) instead of blocking for the full timeout on a dead process.
    liveness = (
        f"if [ ! -e /proc/{rpid} ]; then "
        f"sudo -n grep -q '{_RESTORE_MARKER}' {workdir}/{log} 2>/dev/null && ok=1; break; fi; "
        if rpid
        else ""
    )
    poll = (
        "ok=0; "
        f"for i in $(seq 1 {int(timeout // 2)}); do "
        f"if sudo -n grep -q '{_RESTORE_MARKER}' {workdir}/{log} 2>/dev/null; then ok=1; break; fi; "
        f"{liveness}"
        "sleep 2; done; "
        'if [ "$ok" = 1 ]; then echo RESTORE_OK; else echo RESTORE_FAIL; fi'
    )
    return executor.run(poll, timeout=timeout + 60)


def attach_criu_log(executor, workdir: str, log: str, full: bool = False) -> str:
    """Read a root-owned CRIU log, attach it to Allure, and return its text.

    Reads the whole file when *full* (e.g. under ``pytest -s``); otherwise only the tail, so the
    verbose CRIU log does not flood captured console output. The complete file always remains at
    ``workdir/<log>``.
    """
    reader = "cat" if full else "tail -c 20000"
    text = executor.run(f"sudo -n {reader} {workdir}/{log} 2>/dev/null").stdout or ""
    attach_text(text, name=log)
    return text


def kill_pid(executor, pid: str | None) -> None:
    """Best-effort ``kill -9`` of an exact PID (never by name, which could match the runner)."""
    if pid:
        executor.run(f"sudo -n kill -9 {pid} 2>/dev/null; true")
