# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""5-layer validation framework for GPU workloads (RVS, hipBLASLt, cudamemtest).

Layers:
    1. Crash/kernel indicators in stdout (segfault, core dump, device reset)
    2. Test-specific pass/fail parsing (RVS markers or shape counts)
    3. dmesg delta — GPU resets, panics, faults that appeared during test
    4. Silent death — empty output with non-zero exit code
    5. Exit-code consistency
"""

from __future__ import annotations

from datetime import datetime, timedelta
import logging
import re
import subprocess

from framework.executors.local_executor import run_cmd_get_stdout_stderr

logger = logging.getLogger(__name__)

_CRASH_PATTERNS = re.compile(
    r"(segfault|segmentation fault|core dump|SIGBUS|SIGSEGV|" r"Kernel panic|device reset|GPU hang|unrecoverable)",
    re.IGNORECASE,
)

_DMESG_CRITICAL = re.compile(
    r"(Kernel panic|Oops|GPU reset|amdgpu.*ring.*timeout|"
    r"amdgpu.*job timedout|RAS.*error|IOMMU.*fault|"
    r"thermal throttle|Hardware Error)",
    re.IGNORECASE,
)

DMESG_CATEGORY_RULES: list[tuple[str, re.Pattern]] = [
    ("gpu_reset", re.compile(r"GPU reset|amdgpu.*ring.*timeout|amdgpu.*job timedout", re.IGNORECASE)),
    ("kernel_panic", re.compile(r"Kernel panic|Oops", re.IGNORECASE)),
    ("ras_error", re.compile(r"RAS.*error", re.IGNORECASE)),
    ("iommu_fault", re.compile(r"IOMMU.*fault", re.IGNORECASE)),
    ("thermal", re.compile(r"thermal throttle", re.IGNORECASE)),
    ("hw_error", re.compile(r"Hardware Error", re.IGNORECASE)),
]

# cudamemtest-specific patterns
_MEMTEST_ERROR = re.compile(
     r"\[ERROR\]\s|" r"^\s*ERROR:\s+\S|" r"\bMemory access fault\b|" r"\bHIP error:",
     re.IGNORECASE | re.MULTILINE,
)
_MEMTEST_WATCHDOG = re.compile(
    r"\[cudamemtest\]\s*FAIL:\s*watchdog timeout",
    re.IGNORECASE,
)
_MEMTEST_RAN = re.compile(
    r"\[cudamemtest\]\s+Ran\s+(\d+)/10\s+sub-test",
    re.IGNORECASE,
)


def _filter_dmesg_recent(dmesg_output: str, cutoff: datetime) -> str:
    """Filter dmesg lines to only those with timestamps after cutoff.

    Parses timestamps in the format emitted by ``dmesg -T``:
    ``[Mon Aug 10 11:30:00 2026]`` at the start of each line.
    Lines without a parseable timestamp are included (conservative).
    """
    ts_re = re.compile(r"^\[([A-Za-z]+ [A-Za-z]+ +\d+ \d+:\d+:\d+ \d+)\]")
    recent_lines: list[str] = []
    for line in dmesg_output.splitlines():
        m = ts_re.match(line)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%a %b %d %H:%M:%S %Y")
                if ts < cutoff:
                    continue
            except ValueError:
                pass
        recent_lines.append(line)
    return "\n".join(recent_lines)


def categorize_dmesg_critical(dmesg_text: str) -> dict[str, int]:
    """Count dmesg lines matching each category in DMESG_CATEGORY_RULES."""
    counts = {cat: 0 for cat, _ in DMESG_CATEGORY_RULES}
    for line in dmesg_text.splitlines():
        for cat, pat in DMESG_CATEGORY_RULES:
            if pat.search(line):
                counts[cat] += 1
                break
    return counts


def pretest_health_probe(lookback_min: int = 30) -> tuple[bool, dict]:
    """Probe pre-existing dmesg for critical events in the last N minutes.

    Returns ``(clean, summary)``:

    * ``clean`` — ``True`` iff no DMESG_CRITICAL matches were found
      in the lookback window. Probe failures (dmesg unavailable,
      timeout, non-zero return code) return ``True`` so the probe
      never blocks a run on its own bug.
    * ``summary`` — dict suitable for ``json.dump`` into
      ``pretest_health.json``: probe status, lookback window, total
      critical count, per-category counts, and up to 5 sample
      matching lines for triage.
    """
    summary: dict = {
        "probe": "skipped",
        "lookback_min": lookback_min,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "critical_total": 0,
        "by_category": {cat: 0 for cat, _ in DMESG_CATEGORY_RULES},
        "samples": [],
        "reason": None,
    }
    try:
        res = subprocess.run(
            ["dmesg", "-T"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError) as e:
        summary["reason"] = f"{type(e).__name__}: {e}"
        return True, summary

    if not res.stdout:
        summary["reason"] = f"dmesg returncode={res.returncode} (empty stdout)"
        return True, summary

    cutoff = datetime.now() - timedelta(minutes=lookback_min)
    recent = _filter_dmesg_recent(res.stdout, cutoff)
    counts = categorize_dmesg_critical(recent)
    total = sum(counts.values())
    summary["probe"] = "ok"
    summary["critical_total"] = total
    summary["by_category"] = counts

    samples: list[str] = []
    for cat, pat in DMESG_CATEGORY_RULES:
        if counts[cat] == 0:
            continue
        for line in recent.splitlines():
            if pat.search(line):
                samples.append(line.strip())
                break
        if len(samples) >= 5:
            break
    summary["samples"] = samples

    return total == 0, summary


def capture_dmesg(cmake_executor=None) -> str | None:
    """Capture current dmesg output. Falls back through multiple methods."""
    methods = [
        ["dmesg"],
        ["sudo", "-n", "dmesg"],
        ["journalctl", "-k", "--no-pager", "-n", "500"],
    ]

    for cmd in methods:
        try:
            rc, stdout, _stderr = run_cmd_get_stdout_stderr(*cmd, timeout=10, quiet=True)
            if rc == 0 and stdout.strip():
                return stdout
        except Exception:
            continue

    logger.debug("dmesg capture unavailable (all methods failed)")
    return None


def dmesg_delta(before: str | None, after: str | None) -> str | None:
    """Return new dmesg lines that appeared between before and after snapshots."""
    if before is None or after is None:
        return None
    before_lines = set(before.splitlines())
    after_lines = after.splitlines()
    new_lines = [line for line in after_lines if line not in before_lines]
    return "\n".join(new_lines) if new_lines else ""


def validate_rvs_result(stdout: str, stderr: str, exit_code: int, dmesg_new: str | None) -> tuple[bool, str]:
    """5-layer validation of RVS test output.

    Returns:
        (failed: bool, message: str) — failed=True means test should be marked FAIL
    """
    # Layer 2: RVS-specific pass/fail parsing
    true_count = len(re.findall(r"\bpass:\s*TRUE\b", stdout))
    false_count = len(re.findall(r"\bpass:\s*FALSE\b", stdout))
    abort_count = len(re.findall(r"\bABORT\b", stdout))

    layer2_failed = False
    if abort_count > 0:
        layer2_msg = f"Layer 2 FAIL: {abort_count} ABORT(s) detected"
        layer2_failed = True
    elif false_count > 0:
        layer2_msg = f"Layer 2 FAIL: {false_count} test(s) reported pass: FALSE " f"({true_count} passed)"
        layer2_failed = True
    elif true_count > 0:
        layer2_msg = f"Layer 2 PASS: {true_count} test(s) reported pass: TRUE"
    else:
        layer2_msg = "Layer 2 SKIP: no RVS pass/fail markers found in output"

    return _validate_common_layers(stdout, stderr, exit_code, dmesg_new, layer2_msg, layer2_failed)


def validate_hipblaslt_result(
    stdout: str,
    stderr: str,
    exit_code: int,
    dmesg_new: str | None,
    shapes_passed: int = 0,
    shapes_failed: int = 0,
) -> tuple[bool, str]:
    """5-layer validation of hipBLASLt bench output.

    Layer 2 checks shape pass/fail counts rather than RVS pass: TRUE/FALSE markers.

    Returns:
        (failed: bool, message: str) — failed=True means test should be marked FAIL
    """
    # Layer 2: hipBLASLt shape pass/fail
    total = shapes_passed + shapes_failed
    layer2_failed = False
    if shapes_failed > 0:
        layer2_msg = f"Layer 2 FAIL: {shapes_failed}/{total} shapes failed " f"({shapes_passed} passed)"
        layer2_failed = True
    elif shapes_passed > 0:
        layer2_msg = f"Layer 2 PASS: {shapes_passed}/{total} shapes passed"
    else:
        layer2_msg = "Layer 2 WARN: no shapes executed"

    return _validate_common_layers(stdout, stderr, exit_code, dmesg_new, layer2_msg, layer2_failed)


def validate_cudamemtest_result(
    stdout: str,
    stderr: str,
    exit_code: int,
    dmesg_new: str | None,
    subtests_ran: int = 0,
    subtests_failed: int = 0,
) -> tuple[bool, str]:
    """5-layer validation of cudamemtest output.

    Layer 2 checks for:
    - Watchdog timeout sentinels
    - Memory/runtime error patterns ([ERROR], Memory access fault, HIP error)
    - Sub-test pass/fail counts

    Returns:
        (failed: bool, message: str) — failed=True means test should be marked FAIL
    """
    combined = stdout + "\n" + stderr

    # Layer 2: cudamemtest-specific pass/fail
    layer2_failed = False

    if _MEMTEST_WATCHDOG.search(combined):
        layer2_msg = "Layer 2 FAIL: watchdog timeout (sub-test did not complete)"
        layer2_failed = True
    else:
        err_count = len(_MEMTEST_ERROR.findall(combined))
        if err_count > 0:
            layer2_msg = f"Layer 2 FAIL: {err_count} memory/runtime error(s) detected"
            layer2_failed = True
        elif subtests_failed > 0:
            layer2_msg = (
                f"Layer 2 FAIL: {subtests_failed}/{subtests_ran + subtests_failed} "
                f"sub-test(s) failed ({subtests_ran} passed)"
            )
            layer2_failed = True
        elif subtests_ran > 0:
            layer2_msg = f"Layer 2 PASS: {subtests_ran}/10 sub-test(s) completed, " f"0 memory errors"
        else:
            layer2_msg = "Layer 2 WARN: no sub-tests executed"

    return _validate_common_layers(stdout, stderr, exit_code, dmesg_new, layer2_msg, layer2_failed)


def _validate_common_layers(
    stdout: str,
    stderr: str,
    exit_code: int,
    dmesg_new: str | None,
    layer2_msg: str,
    layer2_failed: bool,
) -> tuple[bool, str]:
    """Shared implementation for layers 1, 3, 4, 5 plus caller-provided Layer 2."""
    messages: list[str] = []
    failed = False

    # Layer 1: Crash indicators
    crash_hits = [line for line in stdout.splitlines() if _CRASH_PATTERNS.search(line)]
    stderr_crashes = [line for line in stderr.splitlines() if _CRASH_PATTERNS.search(line)]
    crash_hits.extend(stderr_crashes)
    if crash_hits:
        messages.append(f"Layer 1 FAIL: {len(crash_hits)} crash indicator(s) in output")
        for h in crash_hits[:3]:
            messages.append(f"  \u2192 {h.strip()[:120]}")
        failed = True
    else:
        messages.append("Layer 1 PASS: no crash indicators")

    # Layer 2: caller-provided
    messages.append(layer2_msg)
    if layer2_failed:
        failed = True

    # Layer 3: dmesg critical events during test
    if dmesg_new is None:
        messages.append("Layer 3 SKIP: dmesg capture unavailable")
    elif not dmesg_new:
        messages.append("Layer 3 PASS: no new kernel messages during test")
    else:
        critical = [line for line in dmesg_new.splitlines() if _DMESG_CRITICAL.search(line)]
        if critical:
            messages.append(f"Layer 3 FAIL: {len(critical)} critical kernel event(s) during test")
            for c in critical[:3]:
                messages.append(f"  \u2192 {c.strip()[:120]}")
            failed = True
        else:
            messages.append(f"Layer 3 PASS: {len(dmesg_new.splitlines())} new kernel messages, " f"none critical")

    # Layer 4: Silent death detection
    has_output = bool(stdout.strip()) or bool(stderr.strip())
    if not has_output and exit_code != 0:
        messages.append(f"Layer 4 FAIL: silent death \u2014 no output but exit code {exit_code}")
        failed = True
    elif not has_output and exit_code == 0:
        messages.append("Layer 4 WARN: no output but exit code 0 (possible no-op)")
    else:
        messages.append("Layer 4 PASS: output present")

    # Layer 5: Exit code consistency
    if exit_code != 0 and not failed:
        messages.append(f"Layer 5 FAIL: non-zero exit code ({exit_code}) despite other layers passing")
        failed = True
    elif exit_code != 0 and failed:
        messages.append(f"Layer 5 INFO: exit code {exit_code} (consistent with failures above)")
    else:
        messages.append(f"Layer 5 PASS: exit code {exit_code}")

    messages.append("--- 5/5 layers evaluated ---")
    return failed, "\n".join(messages)


def validate_transferbench_result(
    stdout: str,
    stderr: str,
    exit_code: int,
    dmesg_new: str | None,
    timed_out: bool = False,
) -> tuple[bool, str]:
    """5-layer validation of TransferBench rsweep output.

    Layer 2 checks for:
    - Watchdog timeout sentinel
    - [ERROR] lines from TransferBench
    - Transfer data lines (Transfer N | X.XX GB/s)
    - Aggregate (CPU) summary line

    Returns:
        (failed: bool, message: str) — failed=True means test should be marked FAIL
    """
    combined = stdout + "\n" + stderr

    # Layer 2: TransferBench-specific pass/fail
    layer2_failed = False

    if timed_out or "[transferbench] FAIL: watchdog timeout" in combined:
        layer2_msg = "Layer 2 FAIL: watchdog timeout (rsweep did not complete)"
        layer2_failed = True
    else:
        errors = len(re.findall(r"\[ERROR\]\s*\w+", combined))
        if errors > 0:
            layer2_msg = f"Layer 2 FAIL: {errors} TransferBench error(s)"
            layer2_failed = True
        else:
            # Count transfer data lines (ASCII | or Unicode │)
            transfers = len(
                re.findall(
                    r"Transfer\s+\d+\s*[│|]\s*[\d.]+\s*GB/s",
                    stdout,
                )
            )
            has_aggregate = bool(
                re.search(
                    r"Aggregate\s*\(CPU\)\s*[│|]\s*[\d.]+\s*GB/s",
                    stdout,
                )
            )

            if transfers == 0:
                layer2_msg = "Layer 2 FAIL: no transfer results in output"
                layer2_failed = True
            else:
                agg_str = ", aggregate OK" if has_aggregate else ""
                layer2_msg = f"Layer 2 PASS: {transfers} transfer(s){agg_str}"

    return _validate_common_layers(stdout, stderr, exit_code, dmesg_new, layer2_msg, layer2_failed)
