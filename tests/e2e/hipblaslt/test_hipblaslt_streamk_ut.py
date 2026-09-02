# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the hipBLASLt unit tests with Tensile StreamK solution selection forced.

Ported from ROCmTest ``hipBlasltUT_With_StreamK``, which sets
``TENSILE_SOLUTION_SELECTION_METHOD=2`` and runs the preinstalled ``hipblaslt-test``
gtest binary over ``*quick*`` or ``*nightly*``, re-running ordinary GEMM correctness
while Tensile selects solutions through the StreamK path. Tests must have actually run
and passed -- reporting no failures is not enough, since a filter that matches nothing
also exits 0 and prints ``[  PASSED  ] 0 tests.``
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import re

import pytest

logger = logging.getLogger(__name__)

#: ROCmTest's ``HIPBLASLTStreamK.execute`` env override.
_STREAMK_ENV = "TENSILE_SOLUTION_SELECTION_METHOD=2"

#: ROCmTest maps its coverage levels onto these gtest filters via ``HIPBLASLT``'s
#: ``hipblaslt_sanity_incl_cases`` / ``hipblaslt_nightly_incl_cases``.
_COVERAGE_FILTERS = {
    "quick": "*quick*",
    "nightly": "*nightly*",
}

#: Runtime weight is per parameter, not per function: on MI210 ``quick`` is ~8.1k
#: tests in ~1.8 min while ``nightly`` is ~11.8k in ~9.3 min.
_COVERAGE_PARAMS = (
    pytest.param("quick", marks=pytest.mark.runtime.fast),
    pytest.param("nightly", marks=pytest.mark.runtime.medium),
)

#: Seconds allowed per coverage run, with headroom for slower parts or a busier host.
_TIMEOUT = float(os.environ.get("ROCM_TEST_HIPBLASLT_STREAMK_TIMEOUT", "5400"))

_ANNOUNCE_RE = re.compile(r"\[=+\]\s*Running\s+(\d+)\s+tests?\s+from\s+(\d+)\s+test\s+(?:suite|case)s?\.")
_RAN_RE = re.compile(r"\[=+\]\s*(\d+)\s+tests?\s+from\s+(\d+)\s+test\s+(?:suite|case)s?\s+ran\.")
_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s*(\d+)\s+tests?\.")
_FAILED_TOTAL_RE = re.compile(r"\[\s*FAILED\s*\]\s*(\d+)\s+tests?,\s*listed below:")
_SKIPPED_TOTAL_RE = re.compile(r"\[\s*SKIPPED\s*\]\s*(\d+)\s+tests?,\s*listed below:")
# Named failures appear once inline and once in the trailing list; the count line
# ("[  FAILED  ] 2 tests, listed below:") is filtered out by the isdigit check.
_FAILED_NAME_RE = re.compile(r"^\[\s*FAILED\s*\]\s+(\S+)", re.MULTILINE)
# Emitted exactly once per test case and never repeated in the summary, which makes
# it a safe independent cross-check on the completed count.
_RUN_MARKER_RE = re.compile(r"^\[\s*RUN\s*\]", re.MULTILINE)

# Upstream sometimes reports a resource shortfall as a gtest skip. ROCmTest's parser
# reclassifies those as failures; keeping the same list preserves that behaviour
# instead of silently accepting a masked failure.
_SKIP_MASKED_FAILURES: tuple[re.Pattern, ...] = (re.compile(r"Host\s+memory\s+usage\s+limit\s+exceed", re.I),)

# Kept narrow so ordinary gtest failure text cannot trip it.
_CRASH_PATTERNS: tuple[str, ...] = ("core dumped", "Segmentation fault", "terminate called", "HIP error")


@dataclass(frozen=True)
class _GTestReport:
    """Counts extracted from a Google Test run."""

    announced: int
    ran: int
    passed: int
    failed: int
    skipped: int
    failed_names: tuple[str, ...]
    run_markers: int


def _parse_report(output: str) -> _GTestReport | None:
    """Parse the gtest summary from *output*.

    Returns ``None`` when the run never announced a test count, meaning the binary
    failed before starting the suite (missing library, bad filter syntax, unreadable
    test data) rather than running cleanly.
    """
    announce = _ANNOUNCE_RE.search(output)
    if announce is None:
        return None

    ran_match = _RAN_RE.search(output)
    passed_match = _PASSED_RE.search(output)
    failed_match = _FAILED_TOTAL_RE.search(output)
    skipped_match = _SKIPPED_TOTAL_RE.search(output)

    names: list[str] = []
    for token in _FAILED_NAME_RE.findall(output):
        if not token.isdigit() and token not in names:
            names.append(token.rstrip(","))

    return _GTestReport(
        announced=int(announce.group(1)),
        ran=int(ran_match.group(1)) if ran_match else 0,
        passed=int(passed_match.group(1)) if passed_match else 0,
        failed=int(failed_match.group(1)) if failed_match else 0,
        skipped=int(skipped_match.group(1)) if skipped_match else 0,
        failed_names=tuple(names),
        run_markers=len(_RUN_MARKER_RE.findall(output)),
    )


def _find_crashes(output: str) -> list[str]:
    """Return crash evidence found in *output*."""
    return [pattern for pattern in _CRASH_PATTERNS if pattern.lower() in output.lower()]


def _completion_problems(report: _GTestReport) -> list[str]:
    """Return reasons the run did not complete a coherent set of tests."""
    problems: list[str] = []
    if report.announced == 0:
        problems.append("gtest announced 0 tests — the filter matched nothing, so nothing was validated")
        return problems
    if report.ran == 0:
        problems.append(
            f"gtest announced {report.announced} tests but never printed a completion line — the run was truncated"
        )
        return problems
    if report.ran != report.announced:
        problems.append(f"gtest ran {report.ran} of the {report.announced} tests it announced")
    if report.run_markers != report.ran:
        problems.append(f"gtest emitted {report.run_markers} [ RUN ] markers for {report.ran} completed tests")
    return problems


def _describe_problems(report: _GTestReport, output: str, exit_code: int) -> list[str]:
    """Return every reason the gtest run should not be accepted as a pass."""
    problems = _completion_problems(report)

    if report.failed or report.failed_names:
        shown = ", ".join(report.failed_names[:10]) or "(names not listed)"
        suffix = ", ..." if len(report.failed_names) > 10 else ""
        problems.append(f"{report.failed or len(report.failed_names)} test(s) failed: {shown}{suffix}")

    # Deliberately not asserting passed + skipped + failed == ran: googletest has
    # varied across versions on whether a skipped test also counts as successful, so
    # that identity would risk failing a healthy run. The checks below hold under
    # either convention.
    if report.ran and report.skipped >= report.ran:
        problems.append(f"all {report.ran} tests were skipped — nothing was validated")

    if report.ran and report.passed == 0:
        problems.append(f"0 of {report.ran} tests passed ({report.skipped} skipped)")

    for pattern in _SKIP_MASKED_FAILURES:
        if pattern.search(output):
            problems.append(f"a skip masks a real failure: output matches {pattern.pattern!r}")

    if exit_code != 0 and not problems:
        problems.append(f"gtest reported no failures but exited {exit_code}")

    return problems


@pytest.mark.gpu_count(1)
@pytest.mark.parametrize("coverage", _COVERAGE_PARAMS)
def test_hipblaslt_streamk_ut(  # pylint: disable=unused-argument
    target_executor,
    ld_path: dict,
    hipblaslt_test_binary: str,
    require_tensile_solution_selection,  # requested for its skip side effect
    coverage: str,
) -> None:
    """Run the hipBLASLt gtest suite with StreamK solution selection forced."""
    gtest_filter = os.environ.get("ROCM_TEST_HIPBLASLT_STREAMK_FILTER") or _COVERAGE_FILTERS[coverage]
    ld = ld_path["LD_LIBRARY_PATH"]

    logger.info("hipBLASLt StreamK UT (%s): filter=%s", coverage, gtest_filter)
    # HIPBLASLT_TENSILE_LIBPATH is deliberately not exported: the preinstalled client
    # resolves its own kernel directory for the device architecture, whereas the
    # tensile_lib_path fixture points at the kernel-less base directory unless
    # --gpu-arch is given, which would fail every test in hipModuleLoad.
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {_STREAMK_ENV} {hipblaslt_test_binary} --gtest_filter={gtest_filter}",
        timeout=_TIMEOUT,
    )
    output = (result.stdout or "") + (result.stderr or "")

    crashes = _find_crashes(output)
    assert (
        not crashes
    ), f"hipblaslt-test ({coverage}) crashed ({', '.join(crashes)}), exit={result.exit_code}:\n{output[-4000:]}"

    report = _parse_report(output)
    assert report is not None, (
        f"hipblaslt-test ({coverage}) never announced a test count (exit={result.exit_code}); "
        f"it failed before starting the suite:\n{output[-4000:]}"
    )

    problems = _describe_problems(report, output, result.exit_code)
    assert not problems, f"hipBLASLt StreamK UT ({coverage}, filter={gtest_filter}) failed:\n  " + "\n  ".join(problems)

    logger.info(
        "hipBLASLt StreamK UT (%s): %d/%d tests passed (%d skipped) with %s",
        coverage,
        report.passed,
        report.ran,
        report.skipped,
        _STREAMK_ENV,
    )
