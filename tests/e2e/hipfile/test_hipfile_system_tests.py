# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hipfile_system_tests.py -- hipFile (AMD Infinity Storage) system tests.

Ported from:   hipFile_System_Tests/system_test (AMD hipFile system test suite)
Upstream ref:  private source withheld; approval recorded
Ported on:     2026-11-26
OSS license:   MIT (test source); hipFile built from ROCm/rocm-systems (MIT)

Validates:
    1. Full suite (read, write, perf, size sweep, alignment, concurrent) passes 6/6
       in AIS fastpath mode and in POSIX fallback mode.
    2. Alignment: aligned I/O is accepted in both modes; unaligned offset, buffer and
       size are rejected on fastpath (ret=-5030) and accepted on fallback.
    3. Concurrent I/O reports 0 failed operations.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hipfile/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.fast

Requires ROCM_TEST_HIPFILE_NVME_PATH to name an NVMe ext4 (data=ordered) mount.
"""

from __future__ import annotations

import re
import shlex

import pytest

_FASTPATH = ("HIPFILE_ALLOW_COMPAT_MODE=false", "fastpath-strict")
_FALLBACK = ("HIPFILE_FORCE_COMPAT_MODE=true", "fallback")

_TESTS_RUN_RE = re.compile(r"^Tests run:\s+(\d+)\s*$", re.MULTILINE)
_TESTS_PASSED_RE = re.compile(r"^Tests passed:\s+(\d+)\s*$", re.MULTILINE)
_TESTS_FAILED_RE = re.compile(r"^Tests failed:\s+(\d+)\s*$", re.MULTILINE)
_ALIGN_OK_RE = re.compile(r"^\s*5/5 alignment checks passed\s*$", re.MULTILINE)
_CONCURRENT_RE = re.compile(r"Completed: (\d+)/(\d+), failed: (\d+)")


def _summary_int(pattern: re.Pattern, text: str, label: str) -> int:
    match = pattern.search(text)
    assert match is not None, f"hipFile output has no '{label}' line:\n{text[-2000:]}"
    return int(match.group(1))


def _case_line(case: str, text: str) -> str:
    """Return the alignment-case result line starting with the case letter."""
    match = re.search(rf"^.*\b{case} (?:aligned|unaligned).*$", text, re.MULTILINE)
    assert match is not None, f"alignment case {case} missing from output:\n{text[-2000:]}"
    return match.group(0)


@pytest.mark.runtime.fast
@pytest.mark.parametrize(
    ("mode", "selection", "expected_tests"),
    [
        pytest.param(_FASTPATH, "all", 6, id="fastpath-all"),
        pytest.param(_FALLBACK, "all", 6, id="fallback-all"),
        pytest.param(_FASTPATH, "align", 1, id="fastpath-align"),
        pytest.param(_FALLBACK, "align", 1, id="fallback-align"),
    ],
)
def test_hipfile_system_tests(target_executor, hipfile_env, mode, selection, expected_tests):
    """Run the hipFile system suite (or the alignment test) in AIS fastpath or POSIX fallback mode."""
    mode_env, mode_banner = mode
    fastpath = mode_banner == "fastpath-strict"
    ld = shlex.quote(hipfile_env.ld_library_path)
    binary = shlex.quote(hipfile_env.binary)
    nvme = shlex.quote(hipfile_env.nvme_path)
    result = target_executor.run(
        f"env {mode_env} LD_LIBRARY_PATH={ld} {binary} --nvme-path {nvme} {selection}",
        timeout=900.0,
    )
    out = result.stdout
    assert result.ok, (
        f"hipFile_system_test {selection} failed (exit={result.exit_code}):\n"
        f"stdout: {out[-3000:]}\nstderr: {result.stderr[:500]}"
    )

    assert _summary_int(_TESTS_RUN_RE, out, "Tests run") == expected_tests, out[-2000:]
    assert _summary_int(_TESTS_PASSED_RE, out, "Tests passed") == expected_tests, out[-2000:]
    assert _summary_int(_TESTS_FAILED_RE, out, "Tests failed") == 0, out[-2000:]

    # Alignment: the suite reports the mode it ran in and checks A, sweep, B, C, D.
    assert f"Mode: {mode_banner}" in out, f"alignment test did not run in {mode_banner} mode:\n{out[-2000:]}"
    assert _ALIGN_OK_RE.search(out), f"alignment checks did not all pass:\n{out[-2000:]}"
    line_a = _case_line("A", out)
    assert "accepted" in line_a, f"case A not accepted: {line_a}"
    assert "rejected" not in line_a, f"case A unexpectedly rejected: {line_a}"
    for case in ("B", "C", "D"):
        line = _case_line(case, out)
        if fastpath:
            assert "rejected" in line, f"case {case} not rejected on fastpath: {line}"
            assert "ret=-5030" in line, f"case {case} rejected with unexpected code: {line}"
        else:
            assert "accepted" in line, f"case {case} not accepted on fallback: {line}"
            assert "rejected" not in line, f"case {case} unexpectedly rejected on fallback: {line}"

    if selection == "all":
        concurrent = _CONCURRENT_RE.findall(out)
        assert len(concurrent) == 2, f"expected two concurrent sub-tests, found {len(concurrent)}:\n{out[-2000:]}"
        for done, total, failed in concurrent:
            assert failed == "0", f"concurrent I/O reported {failed} failed operations"
            assert done == total, f"concurrent I/O completed {done}/{total}"
