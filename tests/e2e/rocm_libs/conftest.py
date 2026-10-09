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
    output/test-binaries/rocm_libs/hipblas_samples/build/clients/staging/
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess

import pytest

from framework.builder.binary_builder import external_build_lock, resolve_parallel_jobs
from tests.e2e.rocm_libs._workload import HIP_MEM_POOL_ENV

logger = logging.getLogger(__name__)

_CORE_SRC = "tests/e2e/rocm_libs/src"

# ---------------------------------------------------------------------------
# hipBLAS samples — sparse clone of ROCm/rocm-libraries
# ---------------------------------------------------------------------------
_HIPBLAS_LIBRARIES_REPO = "https://github.com/ROCm/rocm-libraries.git"
_HIPBLAS_SPARSE_PROJECT = "projects/hipblas"
# Sentinel binary: its presence means all samples were built successfully
_HIPBLAS_SAMPLE_SENTINEL = "hipblas-example-sgemm"
# Default branch — override with HIPBLAS_LIBRARIES_REF env var when needed
_HIPBLAS_LIBRARIES_REF = os.environ.get("HIPBLAS_LIBRARIES_REF", "develop")


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


# ---------------------------------------------------------------------------
# hipBLAS samples bin-dir fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def hipblas_samples_bin_dir(
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
) -> str:
    """Return the directory that contains the pre-built hipBLAS sample binaries.

    Resolution order:

    1. ``rock_dir/bin`` — binaries shipped with the ``hipblas-samples`` package.
    2. Build from source using a sparse clone of ``ROCm/rocm-libraries`` via
       ``external_build.clone_repo``.  The build is cached for the session
       (idempotent clone + sentinel check skip rebuild when already built).

    All dependencies are session-scoped — no function-scoped fixtures are
    requested here to preserve the session-scope guarantee.

    Returns:
        Absolute path to the directory containing the sample binaries.
    """
    rocm_bin = os.path.join(rock_dir, "bin")
    sentinel_in_pkg = os.path.join(rocm_bin, _HIPBLAS_SAMPLE_SENTINEL)

    # --- 1. Pre-installed package -------------------------------------------
    # Use cmake_executor for remote nodes; fall back to local os.path.isfile.
    # Both paths are session-safe — no function-scoped fixture is needed here.
    if cmake_executor is not None:
        probe = cmake_executor.run(f"test -f {sentinel_in_pkg}", timeout=15.0)
        found_in_pkg = probe.ok
    else:
        found_in_pkg = os.path.isfile(sentinel_in_pkg)

    if found_in_pkg:
        logger.info("hipBLAS samples found pre-installed at %s", rocm_bin)
        return rocm_bin

    # --- 2. Build from source -----------------------------------------------
    build_timeout = float(framework_config.therock.build_timeout_secs)

    # Sparse clone: only the projects/hipblas subtree is fetched
    dest = pathlib.Path(compiler_build_dir) / "rocm_libs" / "hipblas_samples" / "rocm-libraries"
    repo_dir = external_build.clone_repo(
        _HIPBLAS_LIBRARIES_REPO,
        dest,
        ref=_HIPBLAS_LIBRARIES_REF,
        sparse_subtree=_HIPBLAS_SPARSE_PROJECT,
        timeout=build_timeout,
    )
    external_build.assert_license_present(repo_dir)

    # repo_dir = <clone_root>/projects/hipblas  (sparse subtree path returned by clone_repo)
    # repo_root = <clone_root>/                 (the actual rocm-libraries checkout root)
    # build_dir = <clone_root>/build            (matches: cmake -S projects/hipblas -B build)
    # staging   = <clone_root>/build/clients/staging
    repo_root = repo_dir.parent
    build_dir = str(repo_root / "build")
    staging_dir = str(repo_root / "build" / "clients" / "staging")
    sentinel_in_build = os.path.join(staging_dir, _HIPBLAS_SAMPLE_SENTINEL)

    # Serialise the configure+build across xdist workers. Each worker runs this
    # session-scoped fixture, so without the lock all N race to cmake --build
    # into the same build_dir; a worker then tries to exec a sample binary while
    # another's linker is still writing it, which fails with ETXTBSY ("Text file
    # busy"). The lock lets one worker build while the rest wait, then re-check
    # the sentinel inside the lock and skip straight to running the binaries.
    with external_build_lock(build_dir):
        already_built = (
            cmake_executor.run(f"test -f {sentinel_in_build}", timeout=15.0).ok
            if cmake_executor is not None
            else os.path.isfile(sentinel_in_build)
        )

        if not already_built:
            jobs = resolve_parallel_jobs(remote_executor=cmake_executor)
            cmake_src = str(repo_dir)  # projects/hipblas — cmake -S source

            configure_cmd = (
                f"cmake -S {cmake_src} -B {build_dir}"
                f" -DCMAKE_PREFIX_PATH={rock_dir}"
                f" -DBUILD_CLIENTS_SAMPLES=ON"
                f" -DBUILD_CLIENTS_TESTS=OFF"
                f" -DBUILD_CLIENTS_BENCHMARKS=OFF"
            )
            logger.info("hipBLAS samples: configuring — %s", configure_cmd)
            if cmake_executor is not None:
                cfg = cmake_executor.run(configure_cmd, timeout=build_timeout)
                cfg_stdout, cfg_stderr = cfg.stdout, cfg.stderr
            else:
                proc = subprocess.run(configure_cmd, shell=True, text=True, capture_output=True)
                cfg_stdout, cfg_stderr = proc.stdout, proc.stderr
                logger.info("hipBLAS samples cmake configure stdout:\n%s", cfg_stdout[-3000:])
                cfg = type(
                    "R",
                    (),
                    {
                        "ok": proc.returncode == 0,
                        "exit_code": proc.returncode,
                        "stdout": cfg_stdout,
                        "stderr": cfg_stderr,
                    },
                )()
            if not cfg.ok:
                pytest.fail(
                    f"hipBLAS samples cmake configure failed (exit={cfg.exit_code}):\n"
                    f"stdout: {cfg_stdout[:2000]}\nstderr: {cfg_stderr[:1000]}"
                )
            logger.info("hipBLAS samples: cmake configure done")

            build_cmd = f"cmake --build {build_dir} -j {jobs}"
            logger.info("hipBLAS samples: building with %d jobs — %s", jobs, build_cmd)
            if cmake_executor is not None:
                bld = cmake_executor.run(build_cmd, timeout=build_timeout)
                bld_stdout, bld_stderr = bld.stdout, bld.stderr
            else:
                proc = subprocess.run(build_cmd, shell=True, text=True, capture_output=True)
                bld_stdout, bld_stderr = proc.stdout, proc.stderr
                logger.info("hipBLAS samples cmake build stdout:\n%s", bld_stdout[-3000:])
                bld = type(
                    "R",
                    (),
                    {
                        "ok": proc.returncode == 0,
                        "exit_code": proc.returncode,
                        "stdout": bld_stdout,
                        "stderr": bld_stderr,
                    },
                )()
            if not bld.ok:
                pytest.fail(
                    f"hipBLAS samples cmake build failed (exit={bld.exit_code}):\n"
                    f"stdout: {bld_stdout[:2000]}\nstderr: {bld_stderr[:1000]}"
                )
            logger.info("hipBLAS samples: build complete")

    # Verify sentinel is present
    verify_ok = (
        cmake_executor.run(f"test -f {sentinel_in_build}", timeout=15.0).ok
        if cmake_executor is not None
        else os.path.isfile(sentinel_in_build)
    )
    if not verify_ok:
        pytest.fail(f"hipBLAS samples build completed but sentinel binary not found: {sentinel_in_build}")

    logger.info("hipBLAS samples built at %s", staging_dir)
    return staging_dir
