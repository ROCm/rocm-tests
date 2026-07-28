# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocblas.py -- rocBLAS GTest client (``rocblas-test``) coverage-tier suite.

Validates:
    Runs the prebuilt ``rocblas-test`` GTest binary shipped with ROCm/TheRock,
    selecting cases per CI gate via ``--gtest_filter``
    The rocBLAS gtest binary self-validates: it exits non-zero on any failing case
    and prints ``[  PASSED  ] N tests.`` on success.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/rocm_libs/:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers per function override the injected ci.* dimension where needed
(ci.pr / ci.weekly) and always declare runtime.*.

"""

import pytest

# Negative filter applied to every tier: known-bug cases are quarantined upstream.
_EXCLUDE_KNOWN_BUGS = "*known_bugs*"
# Stress cases are excluded from the fast/nightly gates and run only in ci.weekly.
_EXCLUDE_STRESS = "*stress*"

# gtest_filter syntax: "<positive1:positive2>-<negative1:negative2>"
_FILTER_PR = f"*quick*-{_EXCLUDE_STRESS}:{_EXCLUDE_KNOWN_BUGS}"
_FILTER_NIGHTLY = f"*quick*:*pre_checkin*-{_EXCLUDE_STRESS}:{_EXCLUDE_KNOWN_BUGS}"
_FILTER_WEEKLY = f"*stress*-{_EXCLUDE_KNOWN_BUGS}"
_FILTER_HMM = f"*HMM*-{_EXCLUDE_KNOWN_BUGS}"

# GPU memory fault patterns written to stderr on invalid-address access; detected
# before the generic result.ok check for a more actionable diagnostic.
_GPU_FAULT_PATTERNS = [
    "Memory Fault Error",
    "GPU core dump",
]


def _assert_gtest_passed(result, label: str) -> None:
    """Assert the rocBLAS gtest run passed, with GPU-fault-aware diagnostics."""
    for pat in _GPU_FAULT_PATTERNS:
        assert pat not in result.stderr, (
            f"GPU memory fault in {label} (pattern: {pat!r}).\n" f"faulting stderr:\n{result.stderr[:1000]}"
        )
    assert result.ok, (
        f"{label} failed (exit={result.exit_code}):\n" f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    assert "FAILED" not in result.stdout, f"{label}: GTest reported test failures:\n{result.stdout[:3000]}"
    assert "PASSED" in result.stdout, f"{label}: GTest pass token not found in stdout:\n{result.stdout[:3000]}"


@pytest.mark.ci.pr
@pytest.mark.runtime.fast
@pytest.mark.usefixtures("rocblas_library_guard")
def test_rocblas_pr_quick(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """PR gate: fast rocBLAS ``*quick*`` smoke subset (stress/known-bugs excluded)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_PR}",
        timeout=600.0,  # runtime.fast cap: < 5 min
    )
    _assert_gtest_passed(result, "rocblas_pr_quick")


@pytest.mark.runtime.medium
@pytest.mark.usefixtures("rocblas_library_guard")
def test_rocblas_nightly(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """Nightly gate: rocBLAS ``*quick*`` + ``*pre_checkin*`` (stress/known-bugs excluded).

    ci.nightly is auto-injected by the rocm_libs CATEGORY_PROFILE.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_NIGHTLY}",
        timeout=1800.0,  # runtime.medium cap: 30 min
    )
    _assert_gtest_passed(result, "rocblas_nightly")


@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
@pytest.mark.usefixtures("rocblas_library_guard")
def test_rocblas_weekly_stress(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """Weekly gate: rocBLAS stress-oriented ``*stress*`` cases (known-bugs excluded).

    Overrides the injected ci.nightly with ci.weekly; these cases can run for a long
    time and are scheduled only in the weekly gate (runtime.soak).
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_WEEKLY}",
        timeout=14400.0,  # runtime.soak cap: 4 hours
    )
    _assert_gtest_passed(result, "rocblas_weekly_stress")


@pytest.mark.runtime.medium
@pytest.mark.usefixtures("rocblas_library_guard")
def test_rocblas_hmm(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """rocBLAS HMM (managed-memory) cases: ``*HMM*`` filter (known-bugs excluded)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_HMM}",
        timeout=1800.0,  # runtime.medium cap: 30 min
    )
    _assert_gtest_passed(result, "rocblas_hmm")
