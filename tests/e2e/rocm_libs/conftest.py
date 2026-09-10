# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/rocm_libs/.

Each binary has its own cmake_build_dir call with ``target=`` so that running
a single test file compiles only the binary that test needs.

Build output layout::

    output/test-binaries/rocm_libs/small_sliding_contact/small_sliding_contact
    output/test-binaries/rocm_libs/jacobian_svd_multistream/jacobian_svd_multistream
    output/test-binaries/rocm_libs/equilibration_batch_kalman/equilibration_batch_kalman
    output/test-binaries/rocm_libs/async_mixed_precision_workflow/async_mixed_precision_workflow
    output/test-binaries/rocm_libs/sparse_csrrf_analysis_reuse/sparse_csrrf_analysis_reuse
    output/test-binaries/rocm_libs/hip_mempool_probe/hip_mempool_probe
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from tests.e2e.rocm_libs._workload import HIP_MEM_POOL_ENV

logger = logging.getLogger(__name__)

_CORE_SRC = "tests/e2e/rocm_libs/src"


# Per-test ci.* / runtime.* markers and library guard for the rocBLAS coverage
# suite (test_rocblas.py), kept here instead of on the test functions so the
# coverage-tier policy lives in one place. The rocm_libs CATEGORY_PROFILE injects
# hw.gpu / layer.math_lib / ci.nightly / e2e.stack / os.linux; this hook adds the
# per-test ci.* overrides, runtime.* durations, and the rocblas_library_guard
# fixture. Keyed on the test function name.
_ROCBLAS_TEST_MARKERS: dict[str, tuple[str, ...]] = {
    "test_rocblas_pr_quick": ("ci.pr", "runtime.medium"),
    "test_rocblas_nightly": ("runtime.medium",),
    "test_rocblas_weekly_stress": ("ci.weekly", "runtime.soak"),
    "test_rocblas_hmm": ("runtime.medium",),
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):  # pylint: disable=unused-argument
    """Attach the rocBLAS suite's per-test markers and library-guard fixture.

    Runs before ``markers_plugin`` (``tryfirst``) so that a ci.* override applied
    here is already present when profile injection checks covered dimensions —
    ``markers_plugin`` only fills dimensions not already covered, so the profile's
    ci.nightly is suppressed on the PR and weekly tests.

    Args:
        config: Active pytest config (unused; required by the hook spec).
        items:  Collected test items, modified in place.
    """
    for item in items:
        name = getattr(item, "originalname", None) or item.name
        markers = _ROCBLAS_TEST_MARKERS.get(name)
        if markers is None:
            continue
        for marker_str in markers:
            item.add_marker(getattr(pytest.mark, marker_str))
        item.add_marker(pytest.mark.usefixtures("rocblas_library_guard"))


def check_rocblas_library(rock_dir: str, remote: bool = False, cmake_executor=None) -> None:
    """Fail with an actionable message if ``librocblas.so`` is absent from the ROCm install.

    Args:
        rock_dir:       Path to the ROCm/TheRock install root.
        remote:         When ``True``, delegate the filesystem check to ``cmake_executor`` via SSH.
        cmake_executor: Session-scoped ``SshExecutor``; required when ``remote=True``.
    """
    fail_msg = (
        f"rocBLAS library not found under {rock_dir}/lib — "
        "ensure the rocblas artifact was downloaded and extracted correctly."
    )
    if remote:
        if cmake_executor is not None:
            result = cmake_executor.run(f"ls {rock_dir}/lib/librocblas.so* 2>/dev/null")
            if not result.ok or not result.stdout.strip():
                pytest.fail(fail_msg)
        return
    lib_dir = pathlib.Path(rock_dir) / "lib"
    if not list(lib_dir.glob("librocblas.so*")):
        pytest.fail(fail_msg)


@pytest.fixture(scope="session")
def rocblas_library_guard(rock_dir: str, cmake_executor) -> None:
    """Session-scoped guard: fail early if rocBLAS is absent from the ROCm install.

    Tests declare this fixture to avoid threading ``rock_dir`` and ``cmake_executor``
    through their own signatures.
    """
    check_rocblas_library(rock_dir, remote=cmake_executor is not None, cmake_executor=cmake_executor)


@pytest.fixture(scope="session")
def rocblas_test_binary(rock_dir: str, cmake_executor) -> str:
    """Locate the prebuilt ``rocblas-test`` GTest client shipped with ROCm/TheRock.

    rocBLAS ships a prebuilt ``rocblas-test`` binary (with its ``rocblas_gtest.data``
    / ``rocblas_gtest.yaml`` data files alongside it) under ``<rock_dir>/bin``.  The
    binary locates its data files relative to its own path, so it runs correctly from
    any working directory.

    We run the *installed* binary directly rather than rebuilding rocBLAS from source
    (the legacy ``git clone`` + ``./install.sh -cd`` path): the framework consumes a
    prebuilt TheRock/ROCm artifact, so the client is already present when the rocBLAS
    test package was installed.  When the test client is not part of the install the
    whole rocBLAS suite is skipped (not failed) — the client is an optional artifact.

    Works both locally and against a remote fleet node (via ``cmake_executor``).
    """
    binary = os.path.join(rock_dir, "bin", "rocblas-test")
    if cmake_executor is not None:
        result = cmake_executor.run(f"test -x {binary} && echo FOUND")
        if "FOUND" not in (result.stdout or ""):
            pytest.skip(f"rocblas-test client not found at {binary} on the remote node")
    elif not os.path.isfile(binary):
        pytest.skip(
            f"rocblas-test client not found at {binary} — install the rocBLAS test "
            "package (rocblas-test) into the ROCm/TheRock artifact to enable this suite."
        )
    return binary


_COMMON_BUILD_KWARGS = {
    "src": _CORE_SRC,
    "compiler_mode": "optional_cxx_hip",
    "sync_dirs": [_CORE_SRC],
}


@pytest.fixture(scope="session")
def small_sliding_contact_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the small sliding-contact sparse solve workload."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/small_sliding_contact",
        gpu_arch=gpu_arch,
        label="rocm_libs/small_sliding_contact",
        artifact="small_sliding_contact",
        target="small_sliding_contact",
    )
    return built_binary(os.path.join(build_dir, "small_sliding_contact"), "small_sliding_contact")


@pytest.fixture(scope="session")
def jacobian_svd_multistream_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the multi-stream Jacobian/SVD workload binary."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/jacobian_svd_multistream",
        gpu_arch=gpu_arch,
        label="rocm_libs/jacobian_svd_multistream",
        artifact="jacobian_svd_multistream",
        target="jacobian_svd_multistream",
    )
    return built_binary(os.path.join(build_dir, "jacobian_svd_multistream"), "jacobian_svd_multistream")


@pytest.fixture(scope="session")
def equilibration_batch_kalman_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the batched equilibration/Kalman workload binary."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/equilibration_batch_kalman",
        gpu_arch=gpu_arch,
        label="rocm_libs/equilibration_batch_kalman",
        artifact="equilibration_batch_kalman",
        target="equilibration_batch_kalman",
    )
    return built_binary(os.path.join(build_dir, "equilibration_batch_kalman"), "equilibration_batch_kalman")


@pytest.fixture(scope="session")
def async_mixed_precision_workflow_binary(
    gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary
) -> str:
    """Compile and return the async mixed-precision ROCm libraries workflow binary."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/async_mixed_precision_workflow",
        gpu_arch=gpu_arch,
        label="rocm_libs/async_mixed_precision_workflow",
        artifact="async_mixed_precision_workflow",
        target="async_mixed_precision_workflow",
    )
    return built_binary(os.path.join(build_dir, "async_mixed_precision_workflow"), "async_mixed_precision_workflow")


@pytest.fixture(scope="session")
def sparse_csrrf_analysis_reuse_binary(
    gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary
) -> str:
    """Compile and return the sparse CSR refactorization analysis-reuse workload."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/sparse_csrrf_analysis_reuse",
        gpu_arch=gpu_arch,
        label="rocm_libs/sparse_csrrf_analysis_reuse",
        artifact="sparse_csrrf_analysis_reuse",
        target="sparse_csrrf_analysis_reuse",
    )
    return built_binary(os.path.join(build_dir, "sparse_csrrf_analysis_reuse"), "sparse_csrrf_analysis_reuse")


@pytest.fixture(scope="session")
def hip_mempool_probe_binary(gpu_arch: str | None, cmake_build_dir, require_gpu_arch_for, built_binary) -> str:
    """Compile and return the HIP stream-ordered memory pool capability probe."""
    require_gpu_arch_for("rocm_libs")
    build_dir = cmake_build_dir(
        **_COMMON_BUILD_KWARGS,
        subdir="rocm_libs/hip_mempool_probe",
        gpu_arch=gpu_arch,
        label="rocm_libs/hip_mempool_probe",
        artifact="hip_mempool_probe",
        target="hip_mempool_probe",
    )
    return built_binary(os.path.join(build_dir, "hip_mempool_probe"), "hip_mempool_probe")


@pytest.fixture(scope="session")
def _hip_mempool_env_cache() -> dict[str, str]:
    """Session cache: host identity -> extra env prefix for the solver run command.

    Probing once per host avoids re-running the capability probe for every test
    that lands on the same node in a fleet run.
    """
    return {}


@pytest.fixture
def hip_mempool_env(  # pylint: disable=redefined-outer-name
    target_executor, ld_path: dict, hip_mempool_probe_binary: str, _hip_mempool_env_cache: dict
) -> str:
    """Return the env-var prefix needed for the HIP stream-ordered memory pool.

    The probe runs on the node selected by ``target_executor``.  If VM-backed
    async pools are unavailable, the fixture returns the legacy
    ``DEBUG_HIP_MEM_POOL_VMHEAP=0`` prefix; otherwise it returns ``""``.  The
    decision is cached per host and does not change workload sizing or pass/fail
    criteria.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    first = next(iter(target_executor))
    host_key = getattr(getattr(first, "node_spec", None), "label", None) or type(first).__name__

    if host_key not in _hip_mempool_env_cache:
        probe = target_executor.run(f"env LD_LIBRARY_PATH={ld} {hip_mempool_probe_binary}")
        stdout = probe.stdout or ""
        if "VMM_POOL=1" in stdout:
            decision = ""
            logger.info("HIP mem-pool probe on %s: VM-backed async pool works; no workaround.", host_key)
        elif "VMM_POOL=0" in stdout:
            decision = HIP_MEM_POOL_ENV
            logger.warning(
                "HIP mem-pool probe on %s: async pool allocation failed (%s); applying %s.",
                host_key,
                stdout.strip(),
                HIP_MEM_POOL_ENV,
            )
        else:
            decision = ""
            logger.warning(
                "HIP mem-pool probe on %s inconclusive (%r); not applying workaround.", host_key, stdout.strip()
            )
        _hip_mempool_env_cache[host_key] = decision

    return _hip_mempool_env_cache[host_key]


# requested_gpu_count is provided by the shared suite-level conftest (tests/conftest.py).
