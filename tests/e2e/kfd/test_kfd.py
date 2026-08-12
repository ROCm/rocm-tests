# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_kfd.py -- KFD (Kernel Fusion Driver) validation via the libhsakmt kfdtest suite.

KFD is the AMD GPU kernel driver / thunk layer (libhsakmt) at the base of the
ROCm stack; ``kfdtest`` is its upstream GTest suite. This module builds kfdtest
from the ROCm/rocm-systems monorepo (see conftest.py) and exercises it in three
ways, mirroring the original test's coverage:

    1. test_kfd_smoke            -- fast gate: kfdtest binary loads libhsakmt and
                                    enumerates its KFD test suites (--gtest_list_tests).
    2. test_kfd_full_suite       -- the whole kfdtest suite (the original test's
                                    default ``./run_kfdtest.sh`` behavior).
    3. test_kfd_hmm_svm          -- the HMM/SVM path: KFDSVMRangeTest / KFDSVMEvictTest
                                    (the original test's ``"HMM"`` case selection).
    4. test_kfd_multi_gpu_parallel -- parallel multi-GPU mode via HSA_TEST_GPUS_NUM
                                    (the original test's ``--hsa_test_gpus_num`` option).

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/kfd/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux
Per-function overrides below adjust hw.* / ci.* / runtime.* where needed.

"""

from __future__ import annotations

import re

import pytest

# HMM (heterogeneous memory management) maps to KFD's shared-virtual-memory suites,
# exactly as the original test selected them for its "HMM" case.
_HMM_SVM_FILTER = "KFDSVMRangeTest.*:KFDSVMEvictTest.*"


def _passed_count(stdout: str) -> int:
    """Return the number of tests gtest reported as passed (0 if none/absent)."""
    match = re.search(r"\[  PASSED  \] (\d+) test", stdout)
    return int(match.group(1)) if match else 0


def _assert_gtest_passed(result, label: str) -> None:
    """Assert a kfdtest gtest run exited clean, ran >0 tests, and reported none failed."""
    assert result.ok, (
        f"{label} failed (exit={result.exit_code}):\n" f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "FAILED" not in result.stdout, f"{label} reported gtest failures:\n{result.stdout[:2000]}"
    # Guard against a --gtest_filter matching zero tests (gtest exits 0 in that case).
    assert (
        _passed_count(result.stdout) > 0
    ), f"{label} ran no tests — check the gtest filter and KFD availability:\n{result.stdout[:2000]}"


@pytest.mark.runtime.fast
def test_kfd_smoke(
    target_executor,
    ld_path: dict,
    kfdtest_binary: str,
):
    """Fast gate: the kfdtest binary loads and enumerates its KFD test suites.

    Uses ``--gtest_list_tests`` so the gate is quick and proves the binary links
    against libhsakmt and can be launched on the node before the heavier runs.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {kfdtest_binary} --gtest_list_tests",
        timeout=120.0,
    )
    assert result.ok, (
        f"kfdtest --gtest_list_tests failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "KFD" in result.stdout, f"kfdtest listed no KFD test suites:\n{result.stdout[:2000]}"


@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
def test_kfd_full_suite(
    target_executor,
    ld_path: dict,
    kfdtest_binary: str,
):
    """Run the full kfdtest suite (the original ``./run_kfdtest.sh`` default).

    The complete KFD suite exercises memory, queue, event, topology and eviction
    paths and can run for tens of minutes, so it is weekly/soak rather than nightly.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {kfdtest_binary}",
        timeout=3600.0,
    )
    _assert_gtest_passed(result, "kfdtest full suite")


@pytest.mark.runtime.medium
def test_kfd_hmm_svm(
    target_executor,
    ld_path: dict,
    kfdtest_binary: str,
):
    """Run the HMM/SVM KFD path (KFDSVMRangeTest / KFDSVMEvictTest).

    Mirrors the original test's ``"HMM"`` case, which appended these shared-virtual-
    memory gtest filters for heterogeneous-memory-management coverage.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {kfdtest_binary} --gtest_filter={_HMM_SVM_FILTER}",
        timeout=1800.0,
    )
    _assert_gtest_passed(result, "kfdtest HMM/SVM suite")


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_kfd_multi_gpu_parallel(
    target_executor,
    ld_path: dict,
    requested_gpu_count: int,
    kfdtest_binary: str,
):
    """Run kfdtest in parallel multi-GPU mode via HSA_TEST_GPUS_NUM.

    Mirrors the original test's ``--hsa_test_gpus_num`` option, which set
    ``HSA_TEST_GPUS_NUM=<n>`` to fan kfdtest out across ``n`` GPUs concurrently.
    The count is bound to the GPUs the framework acquired (``requested_gpu_count``);
    GPU visibility is injected by ``target_executor``.
    """
    if requested_gpu_count < 2:
        pytest.skip("kfdtest parallel multi-GPU mode requires at least 2 acquired GPUs")
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env HSA_TEST_GPUS_NUM={requested_gpu_count} LD_LIBRARY_PATH={ld} {kfdtest_binary}",
        timeout=3600.0,
    )
    _assert_gtest_passed(result, f"kfdtest parallel ({requested_gpu_count} GPUs)")
