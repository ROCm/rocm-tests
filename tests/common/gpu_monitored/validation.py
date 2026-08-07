# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""5-layer validation framework for RVS and GPU workloads.

Layers:
  1. Crash/kernel indicators in stdout (segfault, core dump, device reset)
  2. RVS-specific pass/fail parsing (pass: TRUE/FALSE, ABORT)
  3. dmesg delta — GPU resets, panics, faults that appeared during test
  4. Silent death — empty output with non-zero exit code
  5. Exit-code consistency
"""

import logging
import re
from typing import List, Optional, Tuple

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


def capture_dmesg(cmake_executor=None) -> Optional[str]:
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


def dmesg_delta(before: Optional[str], after: Optional[str]) -> Optional[str]:
    """Return new dmesg lines that appeared between before and after snapshots."""
    if before is None or after is None:
        return None
    before_lines = set(before.splitlines())
    after_lines = after.splitlines()
    new_lines = [line for line in after_lines if line not in before_lines]
    return "\n".join(new_lines) if new_lines else ""


def validate_rvs_result(stdout: str, stderr: str, exit_code: int, dmesg_new: Optional[str]) -> Tuple[bool, str]:
    """5-layer validation of RVS test output.

    Returns:
        (failed: bool, message: str) — failed=True means test should be marked FAIL
    """
    messages: List[str] = []
    failed = False
    layers_evaluated = 0

    # Layer 1: Crash indicators
    crash_hits = [line for line in stdout.splitlines() if _CRASH_PATTERNS.search(line)]
    stderr_crashes = [line for line in stderr.splitlines() if _CRASH_PATTERNS.search(line)]
    crash_hits.extend(stderr_crashes)
    if crash_hits:
        messages.append(f"Layer 1 FAIL: {len(crash_hits)} crash indicator(s) in output")
        for h in crash_hits[:3]:
            messages.append(f"  → {h.strip()[:120]}")
        failed = True
    else:
        messages.append("Layer 1 PASS: no crash indicators")
    layers_evaluated += 1

    # Layer 2: RVS-specific pass/fail parsing
    true_count = len(re.findall(r"\bpass:\s*TRUE\b", stdout))
    false_count = len(re.findall(r"\bpass:\s*FALSE\b", stdout))
    abort_count = len(re.findall(r"\bABORT\b", stdout))

    if abort_count > 0:
        messages.append(f"Layer 2 FAIL: {abort_count} ABORT(s) detected")
        failed = True
    elif false_count > 0:
        messages.append(f"Layer 2 FAIL: {false_count} test(s) reported pass: FALSE " f"({true_count} passed)")
        failed = True
    elif true_count > 0:
        messages.append(f"Layer 2 PASS: {true_count} test(s) reported pass: TRUE")
    else:
        messages.append("Layer 2 SKIP: no RVS pass/fail markers found in output")
    layers_evaluated += 1

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
                messages.append(f"  → {c.strip()[:120]}")
            failed = True
        else:
            messages.append(f"Layer 3 PASS: {len(dmesg_new.splitlines())} new kernel messages, " f"none critical")
    layers_evaluated += 1

    # Layer 4: Silent death detection
    has_output = bool(stdout.strip()) or bool(stderr.strip())
    if not has_output and exit_code != 0:
        messages.append(f"Layer 4 FAIL: silent death — no output but exit code {exit_code}")
        failed = True
    elif not has_output and exit_code == 0:
        messages.append("Layer 4 WARN: no output but exit code 0 (possible no-op)")
    else:
        messages.append("Layer 4 PASS: output present")
    layers_evaluated += 1

    # Layer 5: Exit code consistency
    if exit_code != 0 and not failed:
        messages.append(f"Layer 5 FAIL: non-zero exit code ({exit_code}) despite other layers passing")
        failed = True
    elif exit_code != 0 and failed:
        messages.append(f"Layer 5 INFO: exit code {exit_code} (consistent with failures above)")
    else:
        messages.append(f"Layer 5 PASS: exit code {exit_code}")
    layers_evaluated += 1

    messages.append(f"--- {layers_evaluated}/5 layers evaluated ---")
    return failed, "\n".join(messages)


