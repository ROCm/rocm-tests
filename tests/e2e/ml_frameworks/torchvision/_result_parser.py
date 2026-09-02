# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parser for the JUnit XML report emitted by a TorchVision UT run.

Each suite is run with ``pytest --junitxml=<path>``; the XML is a machine-readable
record of every case (``tests`` / ``failures`` / ``errors`` / ``skipped`` counts on
each ``<testsuite>``, and a ``<failure>`` / ``<error>`` / ``<skipped>`` child on the
cases that were not clean). Parsing this structured report -- rather than scraping
verbose terminal output -- means the pass/fail decision does not depend on pytest's
line formatting or on outcome tokens that stream onto later lines.

A run whose report is missing or unparseable -- because the process aborted before
pytest wrote it (e.g. a GPU memory-access fault / core dump) -- yields an empty
summary (``total == 0``); the caller treats that as "no results produced" and fails
the test, so a hard crash can never masquerade as a clean run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree


@dataclass
class TorchVisionRunSummary:
    """Aggregated outcome of a TorchVision pytest UT run.

    Attributes:
        passed:        Count of cases that passed.
        skipped:       Count of skipped cases (includes ``xfail``).
        failed:        Count of failing cases.
        errored:       Count of erroring cases (setup/teardown/collection errors).
        failed_names:  Node ids of failing cases, for diagnostics.
        errored_names: Node ids of erroring cases, for diagnostics.
    """

    passed: int = 0
    skipped: int = 0
    failed: int = 0
    errored: int = 0
    failed_names: list[str] = field(default_factory=list)
    errored_names: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of cases recorded in the report."""
        return self.passed + self.skipped + self.failed + self.errored

    @property
    def is_clean(self) -> bool:
        """True when at least one case ran and none failed or errored."""
        return self.total > 0 and self.failed == 0 and self.errored == 0


def _case_id(case: ElementTree.Element) -> str:
    """Return a readable node id for a ``<testcase>`` element."""
    classname = case.get("classname", "")
    name = case.get("name", "")
    return f"{classname}::{name}" if classname else name


def parse_junit_xml(xml_text: str) -> TorchVisionRunSummary:
    """Parse a pytest JUnit XML report into a :class:`TorchVisionRunSummary`.

    Args:
        xml_text: The ``--junitxml`` report content (may be empty when the run
            crashed before pytest wrote the report).

    Returns:
        A :class:`TorchVisionRunSummary` with per-outcome counts and the node ids
        of failing and erroring cases. An empty/unparseable report yields a zeroed
        summary (``total == 0``).
    """
    summary = TorchVisionRunSummary()
    text = xml_text.strip()
    if not text:
        return summary

    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return summary

    # The root is ``<testsuites>`` (wrapping one or more ``<testsuite>``) or, for a
    # single suite, ``<testsuite>`` directly.
    suites = root.iter("testsuite")
    for suite in suites:
        for case in suite.findall("testcase"):
            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")
            if error is not None:
                summary.errored += 1
                summary.errored_names.append(_case_id(case))
            elif failure is not None:
                summary.failed += 1
                summary.failed_names.append(_case_id(case))
            elif skipped is not None:
                summary.skipped += 1
            else:
                summary.passed += 1

    return summary
