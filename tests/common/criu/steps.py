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


# ---------------------------------------------------------------------------
# Assertion helpers (shared across all criu test suites)
# ---------------------------------------------------------------------------

_DUMP_CRASH_MARKERS: tuple[str, ...] = (
    "Segmentation fault",
    "Aborted (",
    "Core dump",
    "criu: Error",
)

_RESTORE_CRASH_MARKERS: tuple[str, ...] = (
    "Segmentation fault",
    "Aborted (",
    "Core dump",
    "criu: Error",
    "RESTORE_FAIL",
)


def assert_criu_dump_ok(result, workdir: str, label: str = "") -> None:
    """Assert the CRIU dump completed: result contains ``OK`` and ``PID_GONE``.

    Checks for crash markers in the combined output before the sentinel check,
    since a crash may suppress the sentinel without setting a non-zero exit code.

    Args:
        result:  Execution result from ``criu_dump()``.
        workdir: Checkpoint working directory (used in failure messages).
        label:   Short workload name for failure messages.
    """
    import pytest

    output = (result.stdout or "") + (result.stderr or "")
    prefix = f"{label}: " if label else ""

    for marker in _DUMP_CRASH_MARKERS:
        if marker in output:
            pytest.fail(
                f"{prefix}CRIU dump crash detected — '{marker}' in output.\n"
                f"workdir: {workdir}\noutput tail: {output[-2000:]}"
            )

    assert "OK" in output, (
        f"{prefix}CRIU dump did not emit 'OK' sentinel.\n" f"workdir: {workdir}\noutput: {output[-2000:]}"
    )
    assert "PID_GONE" in output, (
        f"{prefix}CRIU dump 'OK' but process still alive ('PID_ALIVE' or no 'PID_GONE').\n"
        f"workdir: {workdir}\noutput: {output[-2000:]}"
    )


def assert_criu_restore_ok(result, workdir: str, label: str = "") -> None:
    """Assert the CRIU restore completed: result contains ``RESTORE_OK``.

    Checks for crash markers and ``RESTORE_FAIL`` sentinel before asserting
    success, so the failure message names the specific cause.

    Args:
        result:  Execution result from ``criu_restore()``.
        workdir: Checkpoint working directory (used in failure messages).
        label:   Short workload name for failure messages.
    """
    import pytest

    output = (result.stdout or "") + (result.stderr or "")
    prefix = f"{label}: " if label else ""

    for marker in _RESTORE_CRASH_MARKERS:
        if marker in output and marker != "RESTORE_FAIL":
            pytest.fail(
                f"{prefix}CRIU restore crash detected — '{marker}' in output.\n"
                f"workdir: {workdir}\noutput tail: {output[-2000:]}"
            )

    assert "RESTORE_OK" in output, (
        f"{prefix}CRIU restore did not emit 'RESTORE_OK' — "
        f"'RESTORE_FAIL' or timeout reached.\n"
        f"workdir: {workdir}\noutput: {output[-2000:]}"
    )


def assert_workload_output_matches(
    before: str,
    after: str,
    label: str = "",
) -> None:
    """Assert pre-checkpoint and post-restore workload outputs match.

    Both strings are stripped before comparison to avoid trailing-newline
    mismatches.  Used by all criu test suites to validate correctness of the
    checkpoint-restore cycle.

    Args:
        before: Workload stdout captured before ``criu dump``.
        after:  Workload stdout captured after ``criu restore`` completes.
        label:  Short workload name for failure messages.
    """

    prefix = f"{label}: " if label else ""
    before_stripped = before.strip()
    after_stripped = after.strip()

    assert before_stripped == after_stripped, (
        f"{prefix}Workload output differs after restore.\n"
        f"before ({len(before_stripped)} chars):\n{before_stripped[:1000]}\n"
        f"after  ({len(after_stripped)} chars):\n{after_stripped[:1000]}"
    )


def launch_workload(
    executor,
    binary: str,
    workdir: str,
    ld_library_path: str,
    launch_sleep_secs: int = 3,
    timeout: float = 120.0,
    out_file: str = "workload.out",
) -> str:
    """Launch a workload binary in the background and return its PID.

    Runs the binary via ``nohup`` from *workdir* so CRIU can dump the process
    from that directory.  Polls briefly after launch to confirm the process is
    alive before returning.  Calls ``pytest.fail`` when the process does not
    start within *launch_sleep_secs* seconds.

    Args:
        executor:          Any executor with a ``.run()`` method.
        binary:            Absolute path to the workload binary.
        workdir:           Directory to run from (also used as CRIU image dir).
        ld_library_path:   Value for ``LD_LIBRARY_PATH``.
        launch_sleep_secs: Seconds to wait for the process to become alive.
        timeout:           Executor timeout for the launch command.
        out_file:          Filename (relative to *workdir*) to capture stdout/stderr.

    Returns:
        PID string of the running workload process.
    """
    import pytest

    cmd = (
        f"cd {workdir} && rm -f dump.log restore.log {out_file} *.img 2>/dev/null; "
        f"env LD_LIBRARY_PATH={ld_library_path} nohup {binary} > {out_file} 2>&1 & pid=$!; "
        "disown 2>/dev/null || true; "
        f"sleep {launch_sleep_secs}; "
        'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 '
        f"{out_file}; fi"
    )
    out = executor.run(cmd, timeout=timeout).stdout or ""
    pid = next(
        (ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")),
        "",
    )
    if not pid:
        pytest.fail(f"Workload binary {binary!r} did not start:\n{out[-1500:]}")
    return pid
