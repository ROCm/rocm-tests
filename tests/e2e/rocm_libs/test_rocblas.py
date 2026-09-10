# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocblas.py -- rocBLAS GTest client (``rocblas-test``) coverage-tier suite.

Validates:
    Runs the prebuilt ``rocblas-test`` GTest binary shipped with ROCm/TheRock,
    selecting cases per CI gate via ``--gtest_filter``.
    The rocBLAS gtest binary self-validates: it exits non-zero on any failing case
    and prints ``[  PASSED  ] N tests.`` on success.

Markers:
    None are declared on the test functions. They are applied from
    ``tests/e2e/rocm_libs/conftest.py``:
      - hw.gpu / layer.math_lib / e2e.stack / os.linux and the default ci.nightly
        come from the rocm_libs CATEGORY_PROFILE (``framework/markers/taxonomy.py``).
      - The per-test ci.* overrides (ci.pr / ci.weekly), runtime.* durations, and
        the ``rocblas_library_guard`` fixture are attached by the conftest
        ``pytest_collection_modifyitems`` hook, keyed on the test function name.

Coverage tiers mirror ``executeRocBlas.py``'s ``Coverage_Map`` (gtest filter form
``include1:include2-exclude1:exclude2``):
      - PR / nightly -> source SANITY == NIGHTLY: ``*quick*`` + ``*pre_checkin*``,
        excluding ``*stress*`` and ``*known_bugs*``.
      - weekly       -> source stress-mode WEEKLY: ``*stress*`` only, excluding
        ``*known_bugs*`` (the rocBLAS_stress coverage).
      - HMM          -> ``*HMM*`` managed-memory cases, excluding ``*known_bugs*``.
"""

# Negative filters. known_bugs cases are quarantined upstream in every tier;
# stress cases are excluded from the PR/nightly gates and run only in weekly.
_EXCLUDE_KNOWN_BUGS = "*known_bugs*"
_EXCLUDE_STRESS = "*stress*"

# gtest_filter syntax: "<positive1:positive2>-<negative1:negative2>".
# PR and nightly are identical, matching source SANITY == NIGHTLY (non-stress mode).
_FILTER_PR = f"*quick*:*pre_checkin*-{_EXCLUDE_STRESS}:{_EXCLUDE_KNOWN_BUGS}"
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


def test_rocblas_pr_quick(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """PR gate: rocBLAS ``*quick*`` + ``*pre_checkin*`` (stress/known-bugs excluded).

    Matches source SANITY (identical to NIGHTLY). ci.pr / runtime.medium are applied
    by the conftest hook.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_PR}",
        timeout=1800.0,  # runtime.medium cap: 30 min
    )
    _assert_gtest_passed(result, "rocblas_pr_quick")


def test_rocblas_nightly(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """Nightly gate: rocBLAS ``*quick*`` + ``*pre_checkin*`` (stress/known-bugs excluded).

    ci.nightly comes from the rocm_libs CATEGORY_PROFILE; runtime.medium is applied
    by the conftest hook.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_NIGHTLY}",
        timeout=1800.0,  # runtime.medium cap: 30 min
    )
    _assert_gtest_passed(result, "rocblas_nightly")


def test_rocblas_weekly_stress(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """Weekly gate: rocBLAS stress-oriented ``*stress*`` cases (known-bugs excluded).

    Mirrors source stress-mode WEEKLY. ci.weekly / runtime.soak are applied by the
    conftest hook; these cases can run for a long time and are scheduled only weekly.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_WEEKLY}",
        timeout=14400.0,  # runtime.soak cap: 4 hours
    )
    _assert_gtest_passed(result, "rocblas_weekly_stress")


def test_rocblas_hmm(
    target_executor,
    ld_path: dict,
    rocblas_test_binary: str,
):
    """rocBLAS HMM (managed-memory) cases: ``*HMM*`` filter (known-bugs excluded).

    runtime.medium is applied by the conftest hook.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {rocblas_test_binary} --gtest_filter={_FILTER_HMM}",
        timeout=1800.0,  # runtime.medium cap: 30 min
    )
    _assert_gtest_passed(result, "rocblas_hmm")
