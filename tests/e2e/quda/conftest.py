# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone/build fixtures for tests/e2e/quda/.

QUDA (https://github.com/lattice/quda) is a large third-party lattice-QCD
library.  It cannot be built with the single-file ``compile_binary``/``hipcc``
path — it is a full CMake project that downloads and builds USQCD/QMP/Eigen and
links against an MPI runtime.  This module therefore uses the framework's
remote-transparent external-build primitives:

    * ``external_build.clone_repo``            -- idempotent git clone (local/remote)
    * ``external_build.assert_license_present`` -- provenance guard
    * ``external_build.detect_mpi_runtime`` /
      ``external_build.provision_openmpi_runtime`` -- MPI discovery / bootstrap
    * a bespoke ``cmake configure/build/install`` runner (subprocess locally,
      ``cmake_executor`` on the remote build node) with the MPI + ROCm build
      environment injected via an ``env VAR=... cmd`` prefix rather than by
      mutating ``os.environ``.

The QUDA ctest suite is then executed by ``target_executor`` on the GPU node in
``tests/e2e/quda/test_quda.py``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re

import pytest

from framework.builder.binary_builder import find_rocm_clangpp, resolve_parallel_jobs
from framework.executors.background_process import _blocking_stream_run
from tests.e2e.quda._workload import CONFIG_LABEL, GRID, NUM_GPUS, NUM_PROCS

logger = logging.getLogger(__name__)

# Upstream QUDA lattice-QCD library.
_QUDA_URL = "https://github.com/lattice/quda"
# Track upstream ``develop`` by default; pin a known
# good branch/tag/commit with QUDA_REF when reproducibility is required.
_DEFAULT_QUDA_REF = "develop"
_QUDA_REF = os.environ.get("QUDA_REF", _DEFAULT_QUDA_REF)

# Sentinel written by a successful cmake configure — used for build idempotency.
_BUILD_SENTINEL = "CTestTestfile.cmake"


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _path_exists(path: str, cmake_executor) -> bool:
    """Return True when *path* is a file, transparently for local/remote nodes."""
    if cmake_executor is not None:
        return cmake_executor.run(f"test -f {path}", timeout=30.0).ok
    return os.path.isfile(path)


def _resolve_compilers(rock_dir: str, cmake_executor) -> tuple[str, str]:
    """Return ``(cxx, cc)`` ROCm compiler paths for the QUDA CMake build.

    Locally we probe the install with ``find_rocm_clangpp`` (TheRock and standard
    ROCm layouts).  On a remote build node the ROCm tree lives on the far host, so
    we fall back to the TheRock flattened layout ``<rock_dir>/lib/llvm/bin/clang++``
    — the same default ``cmake_build_dir`` uses.  The C compiler is the sibling
    ``clang``/``amdclang`` next to ``clang++``.
    """
    clangpp = find_rocm_clangpp(rock_dir) if cmake_executor is None else None
    cxx = str(clangpp) if clangpp is not None else f"{rock_dir}/lib/llvm/bin/clang++"
    # "clang++" -> "clang" and "amdclang++" -> "amdclang" (path preserved).
    cc = cxx.replace("clang++", "clang")
    return cxx, cc


def _run_build_step(cmd: str, *, cmake_executor, timeout: float, label: str, log_path: str) -> None:
    """Run a single cmake build step, streaming output live and to *log_path*.

    QUDA's configure/build/install each run for many minutes, so a buffered
    ``subprocess.run(capture_output=True)`` (which shows nothing until it exits)
    is unusable here.  Remote steps stream through the SSH executor's own
    ``stream=True`` path; local steps use the framework's shared streaming Popen
    runner (``_blocking_stream_run``) which forwards stdout+stderr to the console
    in real time *and* appends them to *log_path*.

    Live console output still requires ``pytest -s`` (pytest captures fd output by
    default); the *log_path* file is written either way — ``tail -f`` it to watch
    a run started without ``-s``.
    """
    logger.info("QUDA %s -> streaming to %s", label, log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"QUDA {label} failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return
    result = _blocking_stream_run(
        command=cmd,
        env=os.environ.copy(),
        cwd=None,
        timeout=timeout,
        stream_stdout=True,
        stream_stderr=True,
        log_path=log_path,
    )
    if not result.ok:
        raise RuntimeError(
            f"QUDA {label} failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


@pytest.fixture(scope="session")
def quda_mpi_runtime(external_build):
    """Return an ``MpiRuntime`` (launcher + build/run env) for the QUDA suite.

    Discovery is read-only first (system ``mpirun`` / known OpenMPI prefixes);
    when no MPI is present we provision a private OpenMPI under the framework
    build dir so the test never depends on global package state or on any
    site-specific, hardcoded OpenMPI install path.
    """
    runtime = external_build.detect_mpi_runtime()
    if runtime is not None:
        logger.info("QUDA: using discovered MPI runtime at %s", runtime.launcher)
        return runtime
    version = os.environ.get("ROCM_TEST_QUDA_OPENMPI_VERSION") or os.environ.get("OPENMPI_VERSION") or "4.1.4"
    logger.info("QUDA: no MPI runtime found; provisioning OpenMPI %s", version)
    return external_build.provision_openmpi_runtime(version=version)


@pytest.fixture(scope="session")
def quda_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    quda_mpi_runtime,
    require_gpu_arch_for,
) -> str:
    """Clone, configure, build, and install QUDA; return the ctest build dir.

    ``QUDA_GPU_ARCH`` comes from the framework ``--gpu-arch`` flag, ROCm paths
    from ``rock_dir``, and the MPI toolchain is injected into the build
    environment for ``FindMPI`` / QMP / USQCD.
    """
    require_gpu_arch_for("quda")  # QUDA_GPU_ARCH must be explicit — no auto-detect here.
    if not GRID:
        raise RuntimeError(
            f"QUDA: no default grid for QUDA_NUM_GPUS={NUM_GPUS}; set QUDA_TEST_GRID_SIZE "
            "to an 'X Y Z T' lattice decomposition matching the rank count."
        )
    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)

    # --- clone -------------------------------------------------------------
    ref_label = _safe_ref_name(_QUDA_REF)
    dest = pathlib.Path(compiler_build_dir) / "quda" / f"quda-{ref_label}"
    quda_dir = external_build.clone_repo(url=_QUDA_URL, dest=dest, ref=_QUDA_REF, timeout=build_timeout)
    external_build.assert_license_present(quda_dir)

    # Namespace the build/install trees by GPU arch AND rank-count config. Two
    # mismatches must each force a distinct build:
    #   * QUDA_GPU_ARCH bakes a single-arch code object in, so a gfx942 build
    #     cannot run on a gfx950 GPU ("device kernel image is invalid").
    #   * QUDA_TEST_NUM_PROCS / grid are baked into the CTest launch at configure
    #     time, so a 1-rank build and a 2-rank build are distinct artifacts.
    # Arch+config-specific build dirs keep the idempotency check below correct:
    # switching --gpu-arch or QUDA_NUM_GPUS triggers a fresh build instead of
    # silently reusing a mismatched one. The clone/source tree stays shared.
    arch_label = gpu_arch or "auto"
    build_dir = f"{quda_dir}/quda_build-{arch_label}-{CONFIG_LABEL}"
    install_dir = f"{quda_dir}/quda_install-{arch_label}-{CONFIG_LABEL}"

    # --- MPI + ROCm build environment (never via os.environ) ----------------
    mpi_home = quda_mpi_runtime.env.get("MPI_HOME", "")
    mpi_bin = os.path.dirname(quda_mpi_runtime.launcher)
    mpi_lib = quda_mpi_runtime.env.get("LD_LIBRARY_PATH", "")
    cxx, cc = _resolve_compilers(rock_dir, cmake_executor)
    llvm_bin = os.path.dirname(cxx)
    # QUDA bakes QUDA_TEST_NUM_PROCS / QUDA_TEST_GRID_SIZE into the CTest test
    # definitions (the "mpirun -np N ... --gridsize" launch) at *configure* time,
    # so these must be present when cmake runs — not just at ctest runtime; we set
    # them across all build steps.  These MUST match the values test_quda.py uses at
    # ctest time (both come from _workload).  QUDA_TEST_GRID_SIZE keeps its embedded
    # spaces via single quotes.  (QUDA_ENABLE_TUNING is a runtime var — see
    # test_quda.py; harmless here.)
    quda_env = (
        f"QUDA_ENABLE_TUNING=0 QUDA_TEST_NUM_PROCS={NUM_PROCS} QUDA_ENABLE_P2P=0 "
        f"QUDA_TEST_GRID_SIZE='{GRID}'"
    )
    env_prefix = (
        f"MPI_HOME={mpi_home} "
        f"ROCM_PATH={rocm_path} "
        f"PATH={mpi_bin}:{rocm_path}/bin:{llvm_bin}:$PATH "
        f"LD_LIBRARY_PATH={mpi_lib}:{rocm_path}/lib:$LD_LIBRARY_PATH "
        f"{quda_env}"
    )

    # --- idempotency: skip rebuild when a configured build tree exists -------
    if _path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
        logger.info("QUDA: existing build tree at %s — skipping configure/build/install", build_dir)
        return build_dir

    jobs = resolve_parallel_jobs(remote_executor=cmake_executor)

    # Per-step build logs (streamed live + persisted) — tail -f these to watch a
    # long build, especially when pytest is run without -s.
    log_dir = os.path.join(framework_config.framework.artifact_dir, "quda")
    os.makedirs(log_dir, exist_ok=True)

    configure_flags = (
        "-DQUDA_TARGET_TYPE=HIP "
        f"-DQUDA_GPU_ARCH={gpu_arch} "
        f"-DROCM_PATH={rocm_path} "
        f"-DCMAKE_PREFIX_PATH={rocm_path} "
        "-DQUDA_DIRAC_CLOVER=ON "
        "-DQUDA_DIRAC_CLOVER_HASENBUSCH=OFF "
        "-DQUDA_DIRAC_DOMAIN_WALL=ON "
        "-DQUDA_DIRAC_NDEG_TWISTED_MASS=ON "
        "-DQUDA_DIRAC_STAGGERED=ON "
        "-DQUDA_DIRAC_TWISTED_MASS=ON "
        "-DQUDA_DIRAC_TWISTED_CLOVER=ON "
        "-DQUDA_DIRAC_WILSON=ON "
        "-DQUDA_FAST_COMPILE_REDUCE=ON "
        "-DQUDA_FAST_COMPILE_DSLASH=ON "
        "-DQUDA_CLOVER_DYNAMIC=ON "
        "-DQUDA_QDPJIT=OFF "
        "-DQUDA_INTERFACE_QDPJIT=OFF "
        "-DQUDA_INTERFACE_MILC=ON "
        "-DQUDA_INTERFACE_CPS=OFF "
        "-DQUDA_INTERFACE_QDP=ON "
        "-DQUDA_INTERFACE_TIFR=OFF "
        "-DQUDA_QMP=ON "
        "-DQUDA_DOWNLOAD_USQCD=ON "
        "-DQUDA_OPENMP=OFF "
        "-DQUDA_MULTIGRID=ON "
        "-DQUDA_DOWNLOAD_EIGEN=ON "
        "-DQUDA_PRECISION=8 "
        f"-DCMAKE_INSTALL_PREFIX={install_dir} "
        "-DCMAKE_BUILD_TYPE=RELEASE "
        f"-DCMAKE_CXX_COMPILER={cxx} "
        f"-DCMAKE_C_COMPILER={cc} "
        f"-DCMAKE_HIP_COMPILER={cxx} "
        "-DBUILD_SHARED_LIBS=ON "
        "-DQUDA_BUILD_SHAREDLIB=ON "
        "-DQUDA_BUILD_ALL_TESTS=ON "
        "-DQUDA_CTEST_DISABLE_BENCHMARKS=ON "
        "-DCMAKE_C_STANDARD=99"
    )

    configure_cmd = f"env {env_prefix} cmake {quda_dir} -B {build_dir} {configure_flags}"
    build_cmd = f"env {env_prefix} cmake --build {build_dir} --parallel {jobs}"
    install_cmd = f"env {env_prefix} cmake --install {build_dir}"

    logger.info("QUDA: configuring (arch=%s, mpi_home=%s)", gpu_arch, mpi_home or "<unset>")
    _run_build_step(
        configure_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="cmake configure",
        log_path=os.path.join(log_dir, "build-configure.log"),
    )
    logger.info("QUDA: building with %s parallel jobs", jobs)
    _run_build_step(
        build_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="cmake build",
        log_path=os.path.join(log_dir, "build-compile.log"),
    )
    logger.info("QUDA: installing to %s", install_dir)
    _run_build_step(
        install_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="cmake install",
        log_path=os.path.join(log_dir, "build-install.log"),
    )

    return build_dir
