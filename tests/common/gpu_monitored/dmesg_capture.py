# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared dmesg capture helpers for orchestrator and pretest health probe."""

from __future__ import annotations

from datetime import datetime, timedelta
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.cpu_executor import CpuExecutor

from framework.executors.local_executor import run_cmd_get_stdout_stderr
from tests.common.gpu_monitored.privilege import run_priv

DMESG_SNAPSHOT_UNAVAILABLE = "# [dmesg-capture] snapshot unavailable\n"

# Marks a snapshot reconstructed from a persisted log rather than read from the
# ring buffer. Recorded in the run summary so a triager can tell a
# ring-buffer-verified result from a syslog-verified one: rsyslog may rate-limit
# or filter, so the fallback is a weaker guarantee than dmesg itself.
DMESG_SOURCE_PREFIX = "# [dmesg-capture] source: "
DMESG_SOURCE_RING_BUFFER = "kernel ring buffer"

# Persisted kernel logs, consulted only when the ring buffer itself is
# unreadable (``kernel.dmesg_restrict=1`` without CAP_SYSLOG). Membership in
# ``adm``/``systemd-journal`` is enough to read these, so a hardened host can
# still have its kernel health validated instead of failing the dmesg layer.
_KERN_LOG_CANDIDATES = ("/var/log/kern.log", "/var/log/messages", "/var/log/syslog")

# Only the tail is read: the consumers look at a lookback window (minutes) and
# at the delta against a pretest snapshot, so older entries cannot matter and a
# multi-GB syslog should never be pulled into memory.
_KERN_LOG_TAIL_BYTES = 8 * 1024 * 1024

# History retained after conversion. The snapshot only has to cover the pretest
# lookback probe (30 min by default) and give the delta an anchor, so keeping
# days of syslog would bloat every artifact for no gain. The line floor is well
# above the orchestrator's DMESG_ANCHOR_MAX_DEPTH (256) and guarantees a
# non-empty snapshot on hosts too quiet to log anything in the window.
_KERN_LOG_WINDOW_MIN = 240
_KERN_LOG_MIN_LINES = 512

# "Aug 23 03:01:30 host kernel: [270743.324371] msg" — traditional syslog,
# which omits the year. The day is space-padded for single digits.
_SYSLOG_BSD_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+\S+\s+kernel:\s*(?P<msg>.*)$"
)

# "2026-08-26T22:14:18.123456+05:30 host kernel: msg" — RFC 3339, emitted by
# newer rsyslog/journald defaults.
_SYSLOG_ISO_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:[.,]\d+)?"
    r"(?:Z|[+-]\d{2}:?\d{2})?\s+\S+\s+kernel:\s*(?P<msg>.*)$"
)

# Matches validation._parse_dmesg_ts, which only understands "dmesg -T".
_DMESG_TS_FMT = "%a %b %d %H:%M:%S %Y"


def dmesg_outcome(rc: int, output: str) -> tuple[bool, str]:
    """Return ``(available, output)`` for a dmesg exit code and its stdout.

    Some kernels emit a permission warning to stderr and exit non-zero while
    still printing a complete ring buffer on stdout. Treat empty stdout (or
    stdout that is only the permission diagnostic) as capture failure.
    """
    output = output or ""
    if rc == 0:
        return True, output
    diagnostic_only = bool(
        re.search(
            r"(?:read kernel buffer failed|Operation not permitted|" r"Permission denied)",
            output,
            re.IGNORECASE,
        )
    )
    return bool(output and not diagnostic_only), (output if output and not diagnostic_only else "")


def dmesg_result(res) -> tuple[bool, str]:
    """Adapt an executor / ``CompletedProcess`` result for :func:`dmesg_outcome`."""
    return dmesg_outcome(getattr(res, "returncode", getattr(res, "exit_code", 1)), res.stdout)


def _syslog_ts_to_dmesg(line: str, now: datetime) -> tuple[datetime, str] | None:
    """Rewrite one syslog kernel line into ``dmesg -T`` form, or ``None``.

    Returns the parsed timestamp alongside the rewritten line so callers can
    trim by age. Non-kernel facilities and unparseable stamps are dropped so
    the result carries the same content as the ring buffer would.
    """
    iso = _SYSLOG_ISO_RE.match(line)
    if iso:
        try:
            ts = datetime.strptime(iso.group("ts").replace(" ", "T"), "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
        return ts, f"[{ts.strftime(_DMESG_TS_FMT)}] {iso.group('msg')}"

    bsd = _SYSLOG_BSD_RE.match(line)
    if not bsd:
        return None
    stamp = f"{bsd.group('mon')} {int(bsd.group('day'))} {bsd.group('time')}"
    # Traditional syslog omits the year, so try the current one and fall back a
    # year when that lands in the future (the log crossed a New Year boundary).
    for year in (now.year, now.year - 1):
        try:
            ts = datetime.strptime(f"{stamp} {year}", "%b %d %H:%M:%S %Y")
        except ValueError:
            continue  # e.g. Feb 29 against a non-leap year
        if ts <= now + timedelta(days=1):
            return ts, f"[{ts.strftime(_DMESG_TS_FMT)}] {bsd.group('msg')}"
    return None


def _kern_log_to_dmesg_format(text: str, now: datetime | None = None) -> str:
    """Convert persisted syslog text into recent ``dmesg -T``-formatted lines."""
    reference = now or datetime.now()
    dated = [entry for line in text.splitlines() if (entry := _syslog_ts_to_dmesg(line, reference))]
    cutoff = reference - timedelta(minutes=_KERN_LOG_WINDOW_MIN)
    recent = [rendered for ts, rendered in dated if ts >= cutoff]
    if len(recent) < _KERN_LOG_MIN_LINES:
        recent = [rendered for _ts, rendered in dated[-_KERN_LOG_MIN_LINES:]]
    return "\n".join(recent)


def _read_log_tail(path: str, cpu_executor: CpuExecutor | None) -> str | None:
    """Return the tail of ``path``, or ``None`` when it cannot be read."""
    argv = ["tail", "-c", str(_KERN_LOG_TAIL_BYTES), path]
    try:
        if cpu_executor is not None:
            res = cpu_executor.run(" ".join(argv))
            return (res.stdout or "") if res.ok else None
        # quiet: the tail can be megabytes and must not reach the console.
        rc, stdout, _stderr = run_cmd_get_stdout_stderr(*argv, timeout=30, quiet=True)
        return stdout if rc == 0 else None
    except Exception:
        return None


def capture_kern_log_text(cpu_executor: CpuExecutor | None = None) -> tuple[bool, str]:
    """Reconstruct a dmesg-equivalent snapshot from persisted kernel logs.

    Read through ``cpu_executor`` when one is given so a remote run reports the
    kernel log of the host owning the GPUs rather than the local machine's.
    """
    for path in _KERN_LOG_CANDIDATES:
        raw = _read_log_tail(path, cpu_executor)
        if raw is None:
            continue
        converted = _kern_log_to_dmesg_format(raw)
        if converted:
            header = f"{DMESG_SOURCE_PREFIX}{path} (kernel ring buffer unreadable)"
            return True, f"{header}\n{converted}"
    return False, ""


def dmesg_source_from_snapshot(text: str) -> str:
    """Name where a captured snapshot came from, for the run summary."""
    if not text or text == DMESG_SNAPSHOT_UNAVAILABLE:
        return "unavailable"
    first = text.split("\n", 1)[0]
    if first.startswith(DMESG_SOURCE_PREFIX):
        return first[len(DMESG_SOURCE_PREFIX) :].split(" (", 1)[0]
    return DMESG_SOURCE_RING_BUFFER


def capture_dmesg_text(cpu_executor: CpuExecutor | None = None) -> tuple[bool, str]:
    """Read the kernel ring buffer, trying framework, local, then privileged paths.

    Falls back to the persisted kernel log when every ring-buffer read is
    refused, which is the normal case under ``kernel.dmesg_restrict=1`` without
    passwordless sudo.
    """
    if cpu_executor is not None:
        try:
            res = cpu_executor.run("dmesg -T")
            available, output = dmesg_result(res)
            if available:
                return available, output
        except Exception:
            pass
    try:
        # quiet: a full ring buffer must not be streamed to the console.
        rc, stdout, _stderr = run_cmd_get_stdout_stderr("dmesg", "-T", timeout=15, quiet=True)
        available, output = dmesg_outcome(rc, stdout)
        if available:
            return available, output
    except Exception:
        pass
    try:
        proc = run_priv(
            ["dmesg", "-T"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        available, output = dmesg_result(proc)
        if available:
            return available, output
    except Exception:
        pass
    return capture_kern_log_text(cpu_executor)
