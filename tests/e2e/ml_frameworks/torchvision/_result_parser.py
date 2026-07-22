# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parser for the verbose ``pytest`` output emitted by the TorchVision UT suites.

Unlike Apex (which drives Python's ``unittest``), the TorchVision functional and
transforms tensor suites are run with ``pytest -v``, so each selected case prints
its node id followed by an outcome and a progress percentage::

    test/test_functional_tensor.py::TestName::test_case[cuda] PASSED    [ 12%]
    test/test_transforms_tensor.py::TestX::test_y[cuda] FAILED          [ 34%]
    test/test_functional_tensor.py::TestZ::test_w[cuda] SKIPPED (reason) [ 56%]
    test/test_functional_tensor.py::TestA::test_b[cuda] XFAIL           [ 78%]
    test/test_functional_tensor.py::TestC::test_d[cuda] XPASS           [ 90%]
    test/test_functional_tensor.py::TestE::test_f[cuda] ERROR           [ 99%]

The outcome does **not** always land on the same line as the node id: when a case
streams to stdout/stderr while running (a warning, live log output), pytest emits
the node id first and appends the ``PASSED`` / ``FAILED`` / ... token (with its
``[ NN%]`` marker) onto a later line. The parser therefore tracks a "pending"
case between its node id and its outcome, exactly as the terminal reader sees it.

A case whose node id appears but whose outcome never does -- because the process
aborted mid-test (e.g. a GPU memory-access fault / core dump) -- is counted as
*errored* (its name recorded in ``unresolved_names``). Without this, a hard crash
would leave zero parsed failures and masquerade as a clean run.

pytest also prints a "short test summary info" block (``FAILED <nodeid> - ...``,
``ERROR <nodeid>``) and a trailing summary line
(``===== 3 failed, 120 passed, 5 skipped in 42.10s =====``). These are used as
diagnostics and as a fallback / cross-check when no per-line outcomes were parsed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

# A per-case node id line: "<file>.py::<...>[ <outcome> [ NN%]]". The node id has
# no whitespace and must contain "::" (and a ".py") so ordinary log lines are not
# mistaken for case lines. ``rest`` holds the outcome (same-line case) or is empty
# (the outcome then arrives on a later line).
_NODEID_LINE = re.compile(r"^(?P<name>[^\s]+\.py::[^\s]+)\s*(?P<rest>.*)$")

# A later line carrying the outcome of a pending case: an outcome word followed by
# the "[ NN%]" progress marker. Kept strict so interleaved warning/log text does
# not accidentally resolve a pending case.
_CONT_OUTCOME = re.compile(r"^(?P<word>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b.*\[\s*\d+%\]\s*$")

# The "short test summary info" lines pytest prints near the end; used as a source
# of failing/erroring case ids when the verbose per-line output was not captured.
_SUMMARY_NAME_LINE = re.compile(r"^(?P<kind>FAILED|ERROR)\s+(?P<name>[^\s]+\.py::[^\s]+)")

# The trailing "===== N failed, M passed, K skipped in T s =====" summary line.
_SUMMARY_LINE = re.compile(r"^=+.*\bin\s+[\d.]+s.*=+$")
# Individual "<count> <outcome>" pairs inside the trailing summary line.
_SUMMARY_COUNT = re.compile(r"(\d+)\s+(passed|failed|error|errors|skipped|xfailed|xpassed|deselected|warnings?)")

# Outcome buckets. ``None`` marks a node id seen with no outcome yet (pending).
_PASS, _SKIP, _FAIL, _ERROR, _XFAIL = "pass", "skip", "fail", "error", "xfail"

# Maps a pytest outcome word to its bucket. XPASS (unexpectedly passed) is treated
# as a failure -- mirroring Apex's handling of unittest "unexpected success" -- so
# a strict-xfail regression cannot masquerade as clean.
_WORD_TO_BUCKET = {
    "PASSED": _PASS,
    "SKIPPED": _SKIP,
    "FAILED": _FAIL,
    "ERROR": _ERROR,
    "XFAIL": _XFAIL,
    "XPASS": _FAIL,
}


def _classify_rest(rest: str) -> str | None:
    """Classify the text after the node id on a case line, or None if pending."""
    token = rest.strip()
    if not token:
        return None
    word = token.split()[0].upper()
    return _WORD_TO_BUCKET.get(word)


def _classify_continuation(line: str) -> str | None:
    """Classify a later line as the outcome of a pending case, or None."""
    match = _CONT_OUTCOME.match(line.strip())
    if not match:
        return None
    return _WORD_TO_BUCKET.get(match.group("word").upper())


@dataclass
class TorchVisionRunSummary:
    """Aggregated outcome of a TorchVision pytest UT run.

    Mirrors ``ApexRunSummary`` in shape and ``is_clean`` / ``total`` / ``ran_total``
    semantics so ``test_torchvision.py`` can assert exactly like ``test_apex.py``.

    Attributes:
        passed:             Count of cases reporting ``PASSED``.
        skipped:            Count of ``SKIPPED`` cases.
        failed:             Count of ``FAILED`` cases (includes ``XPASS``, which is
                            treated as a failure -- an unexpected pass).
        errored:            Count of ``ERROR`` cases, including cases whose outcome
                            never appeared (see ``unresolved_names``).
        expected_failures:  Count of ``XFAIL`` results (not a failure).
        failed_names:       Node ids of failing cases, for diagnostics.
        errored_names:      Node ids of erroring cases, for diagnostics.
        unresolved_names:   Cases whose node id was seen but whose outcome was never
                            printed -- the signature of a mid-test crash.
        ran_total:          Total case count from the trailing pytest summary line
                            (cross-check independent of the per-line parse).
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
        """Total number of cases observed (all outcomes, including pending)."""
        return self.passed + self.skipped + self.failed + self.errored + self.expected_failures

    @property
    def is_clean(self) -> bool:
        """True when at least one case ran and none failed, errored, or crashed."""
        ran = self.total > 0 or self.ran_total > 0
        return ran and self.failed == 0 and self.errored == 0


def _parse_summary_counts(stripped: str) -> tuple[int, int, int]:
    """Parse a trailing pytest summary line into ``(failed, errored, ran)`` counts."""
    failed = errored = ran = 0
    for count, word in _SUMMARY_COUNT.findall(stripped):
        n = int(count)
        if word == "failed":
            failed += n
            ran += n
        elif word in ("error", "errors"):
            errored += n
            ran += n
        elif word in ("passed", "skipped", "xfailed", "xpassed"):
            ran += n
    return failed, errored, ran


def _tally(parsed: dict[str, str | None], ran_total: int) -> TorchVisionRunSummary:
    """Fold the per-case ``name -> outcome`` map into a :class:`TorchVisionRunSummary`."""
    summary = TorchVisionRunSummary(ran_total=ran_total)
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
        else:  # None: node id seen, outcome never printed -> mid-test crash/abort
            summary.errored += 1
            summary.unresolved_names.append(name)
            summary.errored_names.append(name)
    return summary


def parse_pytest_output(text: str) -> TorchVisionRunSummary:
    """Parse verbose ``pytest`` output into a :class:`TorchVisionRunSummary`.

    Args:
        text: Combined stdout/stderr captured from the two UT suites.

    Returns:
        A :class:`TorchVisionRunSummary` with per-outcome counts and the node ids
        of the failing, erroring, and unresolved (crashed) cases.
    """
    # name -> outcome bucket, or None while the case's outcome is still pending. A
    # dict keyed by node id also deduplicates when output is captured more than
    # once (e.g. streamed and re-logged).
    parsed: dict[str, str | None] = {}
    summary_names: dict[str, list[str]] = {"FAILED": [], "ERROR": []}
    summary_failed = summary_errored = summary_ran = 0
    pending: str | None = None  # most recently started, not-yet-resolved case

    for raw in text.splitlines():
        stripped = raw.strip()

        # Trailing "===== N failed, M passed ... in T s =====" summary line.
        if _SUMMARY_LINE.match(stripped):
            failed, errored, ran = _parse_summary_counts(stripped)
            summary_failed += failed
            summary_errored += errored
            summary_ran += ran
            pending = None
            continue

        # "short test summary info" name lines: FAILED/ERROR <nodeid> [ - reason].
        name_line = _SUMMARY_NAME_LINE.match(stripped)
        if name_line:
            summary_names[name_line.group("kind")].append(name_line.group("name").strip())
            pending = None
            continue

        # A per-case node id line (possibly with the outcome on the same line).
        node = _NODEID_LINE.match(stripped)
        if node:
            name = node.group("name").strip()
            outcome = _classify_rest(node.group("rest"))
            parsed[name] = outcome
            pending = name if outcome is None else None
            continue

        # A pending case awaiting its outcome on a later line.
        if pending is not None:
            outcome = _classify_continuation(stripped)
            if outcome is not None:
                parsed[pending] = outcome
                pending = None

    summary = _tally(parsed, summary_ran)

    # Fallback: a non-verbose run prints no per-line outcomes, only the trailing
    # summary line and the short-summary FAILED/ERROR name lines.
    if summary.failed == 0 and summary.errored == 0:
        summary.failed = summary_failed
        summary.errored = summary_errored
    if not summary.failed_names:
        summary.failed_names = summary_names["FAILED"]
    if not summary.errored_names:
        summary.errored_names = summary_names["ERROR"]

    return summary
