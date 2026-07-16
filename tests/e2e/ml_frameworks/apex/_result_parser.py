# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parser for the verbose ``unittest`` output emitted by the Apex L0 suite.

The L0 runner drives Python's ``unittest`` in verbose mode, so each sub-test
prints a header ``<name> ... `` followed by its outcome::

    test_adam (test_optim.TestFusedAdam.test_adam) [torch.float16] ... ok
    test_mlp (test_mlp.TestMLP.test_mlp) ... skipped 'requires >1 GPU'
    test_norm (test_ln.TestLayerNorm.test_norm) ... FAIL
    test_dist (test_dist.TestPP.test_dist) ... ERROR

The outcome does **not** always land on the same line as the header: when a
sub-test writes to stdout/stderr while running (a warning, a compile log), the
``ok`` / ``skipped`` / ``FAIL`` / ``ERROR`` token is pushed onto a later line.
The parser therefore tracks a "pending" sub-test between its header and its
outcome, exactly as the terminal reader sees it.

A sub-test whose header appears but whose outcome never does -- because the
process aborted mid-test (e.g. a GPU memory-access fault / core dump) -- is
counted as *errored* (its name recorded in ``unresolved_names``). Without this,
a hard crash would leave zero parsed failures and masquerade as a clean run.

Failures and errors are additionally summarised at the end of each module in
``ERROR:``/``FAIL:`` header blocks and a trailing ``FAILED (failures=.., ...)`` /
``OK`` line; these are used as diagnostics and as a fallback when no per-line
outcomes were parsed at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# A sub-test header line: "<name> (<module>.<Class>.<method>)[ [params]] ... [rest]".
# ``rest`` may hold the outcome (same-line case) or be empty/log text (the outcome
# then arrives on a later line). The parenthesised id must contain a dot so plain
# log lines that happen to contain " ... " are not mistaken for test headers.
_START_LINE = re.compile(
    r"^(?P<name>[^\s(]+\s+\([^)]*\.[^)]*\)(?:\s+\[[^\]]+\])*)\s+\.\.\.\s*(?P<rest>.*)$"
)

# The "ERROR:"/"FAIL:" summary header preceding each traceback. Requires the
# unittest test-id form so ordinary "ERROR: ..." log lines are not captured.
_HEADER_LINE = re.compile(r"^(?P<kind>ERROR|FAIL):\s+(?P<name>[^\s(]+\s+\([^)]*\.[^)]*\).*)$")

# Trailing "Ran N tests in ..." / "FAILED (...)" summary lines (one per module).
_RAN_LINE = re.compile(r"^Ran (\d+) tests? in")
_FAILED_LINE = re.compile(r"^FAILED\s+\((?P<body>.*)\)\s*$")
_COUNT_IN_BODY = re.compile(r"(failures|errors)=(\d+)")

# Outcome buckets. ``None`` marks a header seen with no outcome yet (pending).
_PASS, _SKIP, _FAIL, _ERROR, _XFAIL = "pass", "skip", "fail", "error", "xfail"


def _classify_rest(rest: str) -> str | None:
    """Classify the text after ``... `` on a header line, or None if pending."""
    low = rest.strip().lower()
    if not low:
        return None
    if low.startswith("expected failure"):
        return _XFAIL
    if low.startswith("unexpected success"):
        return _FAIL
    if low.startswith("ok"):
        return _PASS
    if low.startswith("skipped"):
        return _SKIP
    if low.startswith("error"):
        return _ERROR
    if low.startswith("fail"):
        return _FAIL
    return None


def _classify_continuation(line: str) -> str | None:
    """Classify a later line as the outcome of a pending sub-test, or None.

    Kept deliberately strict (bare tokens or a "... <token>" tail) so interleaved
    warning/log text does not accidentally resolve a pending sub-test.
    """
    stripped = line.strip()
    if stripped == "ok" or stripped.endswith("... ok") or "... ok" in stripped:
        return _PASS
    if stripped.startswith("skipped") or "... skipped" in stripped:
        return _SKIP
    if stripped == "FAIL" or stripped.endswith("... FAIL"):
        return _FAIL
    if stripped == "ERROR" or stripped.endswith("... ERROR"):
        return _ERROR
    if stripped.startswith("expected failure"):
        return _XFAIL
    if stripped.startswith("unexpected success"):
        return _FAIL
    return None


@dataclass
class ApexRunSummary:
    """Aggregated outcome of an Apex L0 unittest run.

    Attributes:
        passed:             Count of sub-tests reporting ``ok``.
        skipped:            Count of skipped sub-tests.
        failed:             Count of failed sub-tests (includes unexpected
                            successes, which unittest treats as failures).
        errored:            Count of errored sub-tests, including sub-tests whose
                            outcome never appeared (see ``unresolved_names``).
        expected_failures:  Count of ``expected failure`` results (not a failure).
        failed_names:       Identifiers of failing sub-tests, for diagnostics.
        errored_names:      Identifiers of erroring sub-tests, for diagnostics.
        unresolved_names:   Sub-tests whose header was seen but whose outcome was
                            never printed -- the signature of a mid-test crash.
        ran_total:          Sum of the per-module ``Ran N tests`` counters.
    """

    passed: int = 0
    skipped: int = 0
    failed: int = 0
    errored: int = 0
    expected_failures: int = 0
    failed_names: list[str] = field(default_factory=list)
    errored_names: list[str] = field(default_factory=list)
    unresolved_names: list[str] = field(default_factory=list)
    ran_total: int = 0

    @property
    def total(self) -> int:
        """Total number of sub-tests observed (all outcomes, including pending)."""
        return self.passed + self.skipped + self.failed + self.errored + self.expected_failures

    @property
    def is_clean(self) -> bool:
        """True when at least one test ran and none failed, errored, or crashed."""
        ran = self.total > 0 or self.ran_total > 0
        return ran and self.failed == 0 and self.errored == 0


def parse_unittest_output(text: str) -> ApexRunSummary:
    """Parse verbose ``unittest`` output into an :class:`ApexRunSummary`.

    Args:
        text: Combined stdout/stderr captured from the L0 runner.

    Returns:
        An :class:`ApexRunSummary` with per-outcome counts and the identifiers of
        the failing, erroring, and unresolved (crashed) sub-tests.
    """
    # name -> outcome bucket, or None while the sub-test's outcome is still
    # pending. A dict keyed by name also deduplicates when the runner's output is
    # captured more than once (e.g. streamed and re-logged).
    parsed: dict[str, str | None] = {}
    header_names: dict[str, list[str]] = {"FAIL": [], "ERROR": []}
    summary_failures = 0
    summary_errors = 0
    ran_total = 0
    pending: str | None = None  # most recently started, not-yet-resolved sub-test

    for raw in text.splitlines():
        stripped = raw.strip()

        header = _HEADER_LINE.match(stripped)
        if header:
            header_names[header.group("kind")].append(header.group("name").strip())
            pending = None
            continue

        ran = _RAN_LINE.match(stripped)
        if ran:
            ran_total += int(ran.group(1))
            pending = None  # module boundary: a trailing bare "OK" must not resolve
            continue

        failed = _FAILED_LINE.match(stripped)
        if failed:
            for kind, count in _COUNT_IN_BODY.findall(failed.group("body")):
                if kind == "failures":
                    summary_failures += int(count)
                else:
                    summary_errors += int(count)
            pending = None
            continue

        start = _START_LINE.match(stripped)
        if start:
            name = start.group("name").strip()
            outcome = _classify_rest(start.group("rest"))
            parsed[name] = outcome
            pending = name if outcome is None else None
            continue

        if pending is not None:
            outcome = _classify_continuation(stripped)
            if outcome is not None:
                parsed[pending] = outcome
                pending = None
            continue

    summary = ApexRunSummary(ran_total=ran_total)
    for name, outcome in parsed.items():
        if outcome == _PASS:
            summary.passed += 1
        elif outcome == _SKIP:
            summary.skipped += 1
        elif outcome == _XFAIL:
            summary.expected_failures += 1
        elif outcome == _FAIL:
            summary.failed += 1
            summary.failed_names.append(name)
        elif outcome == _ERROR:
            summary.errored += 1
            summary.errored_names.append(name)
        else:  # None: header seen, outcome never printed -> mid-test crash/abort
            summary.errored += 1
            summary.unresolved_names.append(name)
            summary.errored_names.append(name)

    # Fallback: a non-verbose run prints no per-line outcomes, only the per-module
    # "FAILED (...)" counters and the ERROR:/FAIL: header blocks.
    if summary.failed == 0 and summary.errored == 0:
        summary.failed = summary_failures
        summary.errored = summary_errors
    if not summary.failed_names:
        summary.failed_names = header_names["FAIL"]
    if not summary.errored_names:
        summary.errored_names = header_names["ERROR"]

    return summary
