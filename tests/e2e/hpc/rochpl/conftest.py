# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone/build fixtures for tests/e2e/hpc/rochpl/.

rocHPL (https://github.com/ROCm/rocHPL) is AMD's GPU-accelerated High-Performance
Linpack benchmark.  It is a full CMake project driven by an ``install.sh`` wrapper
that links against an MPI runtime and rocBLAS -- it cannot be built with the
single-file ``compile_binary``/``hipcc`` path.  This module therefore uses the
framework's remote-transparent external-build primitives:

    * ``external_build.clone_repo``             -- idempotent git clone (local/remote)
    * ``external_build.assert_license_present``  -- provenance guard
    * ``external_build.detect_mpi_runtime`` /
      ``external_build.provision_openmpi_runtime`` -- MPI discovery / bootstrap
    * a bespoke ``./install.sh`` runner (streaming subprocess locally, or the
      ``cmake_executor`` on the remote build node) with the MPI + ROCm build
      environment injected via an ``env VAR=... cmd`` prefix rather than by
      mutating ``os.environ``.

The resulting ``mpirun_rochpl`` launcher is then executed by ``target_executor``
on the GPU node in ``tests/e2e/hpc/rochpl/test_rochpl.py``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re

import pytest

from framework.builder.binary_builder import find_rocm_clangpp
from framework.executors.background_process import _blocking_stream_run
from tests.e2e.hpc.rochpl._workload import CONFIG_LABEL, NUM_GPUS, P, Q

logger = logging.getLogger(__name__)

# Upstream AMD ROCm High-Performance Linpack benchmark.
_ROCHPL_URL = "https://github.com/ROCm/rocHPL"
# Track "main" by default; pin a known-good branch/tag/commit with ROCHPL_REF
# when reproducibility is required.
_DEFAULT_ROCHPL_REF = "main"
_ROCHPL_REF = os.environ.get("ROCHPL_REF", _DEFAULT_ROCHPL_REF)

# install.sh drops the launcher wrapper here on success -- used for idempotency.
_BUILD_SENTINEL = "mpirun_rochpl"


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _path_exists(path: str, cmake_executor) -> bool:
    """Return True when *path* is a file, transparently for local/remote nodes."""
    if cmake_executor is not None:
        return cmake_executor.run(f"test -f {path}", timeout=30.0).ok
    return os.path.isfile(path)


def _resolve_llvm_bin(rock_dir: str, cmake_executor) -> str:
    """Return the ROCm LLVM ``bin`` directory to prepend to PATH for the build."""
    clangpp = find_rocm_clangpp(rock_dir) if cmake_executor is None else None
    cxx = str(clangpp) if clangpp is not None else f"{rock_dir}/lib/llvm/bin/clang++"
    return os.path.dirname(cxx)


def _run_build_step(cmd: str, *, cmake_executor, timeout: float, label: str, log_path: str) -> None:
    """Run a single build step, streaming output live and to *log_path*.

    rocHPL's ``install.sh`` (cmake configure + compile + MPI wrapper generation)
    runs for several minutes, so a buffered ``subprocess.run(capture_output=True)``
    (which shows nothing until it exits) is unusable here.  Remote steps stream
    through the SSH executor's own ``stream=True`` path; local steps use the
    framework's shared streaming Popen runner (``_blocking_stream_run``) which
    forwards stdout+stderr to the console in real time *and* appends them to
    *log_path*.

    Live console output still requires ``pytest -s`` (pytest captures fd output by
    default); the *log_path* file is written either way -- ``tail -f`` it to watch
    a run started without ``-s``.
    """
    logger.info("rocHPL %s -> streaming to %s", label, log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"rocHPL {label} failed on remote (exit={result.exit_code}):\n"
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
            f"rocHPL {label} failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


@pytest.fixture(scope="session")
def rochpl_mpi_runtime(external_build):
    """Return an ``MpiRuntime`` (launcher + build/run env) for the rocHPL suite.

    Discovery is read-only first (system ``mpirun`` / known OpenMPI prefixes);
    when no MPI is present we provision a private OpenMPI under the framework
    build dir so the test never depends on global package state or on any
    site-specific, hardcoded OpenMPI install path.  ``mpirun_rochpl`` links
    against and re-invokes this same launcher at run time.
    """
    runtime = external_build.detect_mpi_runtime()
    if runtime is not None:
        logger.info("rocHPL: using discovered MPI runtime at %s", runtime.launcher)
        return runtime
    version = os.environ.get("ROCM_TEST_ROCHPL_OPENMPI_VERSION") or os.environ.get("OPENMPI_VERSION") or "4.1.4"
    logger.info("rocHPL: no MPI runtime found; provisioning OpenMPI %s", version)
    return external_build.provision_openmpi_runtime(version=version)


@pytest.fixture(scope="session")
def rochpl_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    rochpl_mpi_runtime,
) -> str:
    """Clone, build (``install.sh``), and return the rocHPL ``build`` directory.

    ROCm paths come from ``rock_dir`` and the MPI toolchain is injected into the
    build environment so rocHPL's cmake ``FindMPI`` and the generated
    ``mpirun_rochpl`` wrapper resolve the same launcher used at run time.

    The clone/build tree is namespaced by GPU arch and process-grid config so
    switching ``--gpu-arch`` or ``ROCHPL_NUM_GPUS`` forces a clean rebuild instead
    of silently reusing a mismatched artifact (rocHPL builds in-tree under
    ``build/`` and cannot relocate that directory).
    """
    if P * Q != NUM_GPUS:
        raise RuntimeError(
            f"rocHPL: process grid P*Q ({P}*{Q}={P * Q}) must equal "
            f"ROCHPL_NUM_GPUS ({NUM_GPUS}); adjust ROCHPL_P/ROCHPL_Q/ROCHPL_NUM_GPUS."
        )
    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)

    # --- clone (arch + config namespaced; install.sh builds in-tree) ---------
    ref_label = _safe_ref_name(_ROCHPL_REF)
    arch_label = gpu_arch or "auto"
    dest = pathlib.Path(compiler_build_dir) / "rochpl" / f"rochpl-{ref_label}-{arch_label}-{CONFIG_LABEL}"
    rochpl_dir = external_build.clone_repo(url=_ROCHPL_URL, dest=dest, ref=_ROCHPL_REF, timeout=build_timeout)
    external_build.assert_license_present(rochpl_dir)

    build_dir = f"{rochpl_dir}/build"

    # --- MPI + ROCm build environment (never via os.environ) -----------------
    mpi_home = rochpl_mpi_runtime.env.get("MPI_HOME", "")
    mpi_bin = os.path.dirname(rochpl_mpi_runtime.launcher)
    mpi_lib = rochpl_mpi_runtime.env.get("LD_LIBRARY_PATH", "")
    llvm_bin = _resolve_llvm_bin(rock_dir, cmake_executor)
    env_prefix = (
        f"ROCM_PATH={rocm_path} "
        f"PATH={mpi_bin}:{rocm_path}/bin:{llvm_bin}:$PATH "
        f"LD_LIBRARY_PATH={mpi_lib}:{rocm_path}/lib:$LD_LIBRARY_PATH"
    )

    # --- idempotency: skip rebuild when the launcher already exists -----------
    if _path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
        logger.info("rocHPL: existing build at %s -- skipping install.sh", build_dir)
        return build_dir

    log_dir = os.path.join(framework_config.framework.artifact_dir, "rochpl")
    os.makedirs(log_dir, exist_ok=True)

    # install.sh runs cmake configure + build under the hood. --with-rocm and
    # --with-mpi point it at this session's ROCm and MPI toolchains. Parallelism
    # is not user-configurable: install.sh hardcodes `make -j$(nproc)` internally
    # (it rejects any -j/--jobs flag via getopt). Runs as the invoking user in a
    # user-writable clone (no sudo -- the original build wrapper's sudo/chown/
    # password steps are dropped).
    mpi_arg = f" --with-mpi={mpi_home}" if mpi_home else ""
    install_cmd = f"cd {rochpl_dir} && env {env_prefix} ./install.sh --with-rocm={rocm_path}{mpi_arg}"

    logger.info("rocHPL: building via install.sh (arch=%s, mpi_home=%s)", arch_label, mpi_home or "<unset>")
    _run_build_step(
        install_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="install.sh",
        log_path=os.path.join(log_dir, "build-install.log"),
    )

    if not _path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
        raise RuntimeError(
            f"rocHPL: install.sh completed but {build_dir}/{_BUILD_SENTINEL} is missing -- "
            "the build layout may have changed upstream."
        )
    return build_dir
