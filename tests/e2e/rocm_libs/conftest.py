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


_COMMON_BUILD_KWARGS = dict(
    src=_CORE_SRC,
    compiler_mode="optional_cxx_hip",
    sync_dirs=[_CORE_SRC],
)


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
def hip_mempool_env(target_executor, ld_path: dict, hip_mempool_probe_binary: str, _hip_mempool_env_cache: dict) -> str:
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
