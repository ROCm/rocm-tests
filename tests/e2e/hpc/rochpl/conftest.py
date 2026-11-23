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

import pytest

from tests.common.hpc.build import path_exists, run_build_step, safe_ref_name

logger = logging.getLogger(__name__)

# Upstream AMD ROCm High-Performance Linpack benchmark.
_ROCHPL_URL = "https://github.com/ROCm/rocHPL"
# Track "main" by default; pin a known-good branch/tag/commit with ROCHPL_REF
# when reproducibility is required.
_DEFAULT_ROCHPL_REF = "main"
_ROCHPL_REF = os.environ.get("ROCHPL_REF", _DEFAULT_ROCHPL_REF)

# install.sh drops the launcher wrapper here on success -- used for idempotency.
_BUILD_SENTINEL = "mpirun_rochpl"

# OpenMPI version to provision when no MPI runtime is discovered on the node.
# Kept in lockstep with the OpenMPI that rocHPL's own ``install.sh`` bootstraps
# (``git clone --branch v5.0.7 ... ompi.git``) so the provisioned launcher matches
# the toolchain rocHPL is validated against. Override with
# ``ROCM_TEST_ROCHPL_OPENMPI_VERSION`` / ``OPENMPI_VERSION`` if a node needs a
# different release.
_OPENMPI_VERSION = "5.0.7"


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
    version = (
        os.environ.get("ROCM_TEST_ROCHPL_OPENMPI_VERSION") or os.environ.get("OPENMPI_VERSION") or _OPENMPI_VERSION
    )
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

    The clone/build tree is namespaced by GPU arch (and ref) only -- P/Q/N/NB are
    runtime arguments to ``mpirun_rochpl``, so one build serves every variant of
    the parametrized run matrix; switching ``--gpu-arch`` forces a clean rebuild
    (rocHPL builds in-tree under ``build/`` and cannot relocate that directory).

    Because the test is parametrized over GPU counts, several xdist workers may
    request this session-scoped build concurrently for the same arch. A
    cross-process ``filelock`` serializes them: the first worker builds; the rest
    block, then reuse the launcher via the sentinel check.
    """
    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)

    ref_label = safe_ref_name(_ROCHPL_REF)
    arch_label = gpu_arch or "auto"
    dest = pathlib.Path(compiler_build_dir) / "rochpl" / f"rochpl-{ref_label}-{arch_label}"

    # Serialize concurrent xdist workers building the same arch tree. The lock is
    # local (coordinates the workers that launch the build); remote builds still
    # run once because the sentinel check inside the lock short-circuits reuse.
    lock_path = str(dest.parent / f".build-{ref_label}-{arch_label}.lock")
    os.makedirs(dest.parent, exist_ok=True)
    from filelock import FileLock  # pylint: disable=import-outside-toplevel

    with FileLock(lock_path, timeout=build_timeout):
        # --- clone (arch namespaced; install.sh builds in-tree) --------------
        rochpl_dir = external_build.clone_repo(url=_ROCHPL_URL, dest=dest, ref=_ROCHPL_REF, timeout=build_timeout)
        external_build.assert_license_present(rochpl_dir)

        build_dir = f"{rochpl_dir}/build"

        # --- MPI + ROCm build environment (never via os.environ) -------------
        mpi_home = rochpl_mpi_runtime.env.get("MPI_HOME", "")
        mpi_bin = os.path.dirname(rochpl_mpi_runtime.launcher)
        mpi_lib = rochpl_mpi_runtime.env.get("LD_LIBRARY_PATH", "")
        # rocHPL's install.sh sets ``PATH=$PATH:$ROCM_PATH/bin`` and its CMake finds
        # the HIP compiler through ROCm's HIP cmake package via ``ROCM_PATH`` (the
        # script accepts no compiler flag). So ``ROCM_PATH`` + ``$ROCM_PATH/bin`` on
        # PATH is all the build needs -- no separate LLVM ``bin`` entry is required
        # (this also matches the run-time env in test_rochpl.py).
        env_prefix = (
            f"ROCM_PATH={rocm_path} "
            f"PATH={mpi_bin}:{rocm_path}/bin:$PATH "
            f"LD_LIBRARY_PATH={mpi_lib}:{rocm_path}/lib:$LD_LIBRARY_PATH"
        )

        # --- idempotency: skip rebuild when the launcher already exists -------
        if path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
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
        run_build_step(
            install_cmd,
            cmake_executor=cmake_executor,
            timeout=build_timeout,
            label="install.sh",
            log_path=os.path.join(log_dir, "build-install.log"),
        )

        if not path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
            raise RuntimeError(
                f"rocHPL: install.sh completed but {build_dir}/{_BUILD_SENTINEL} is missing -- "
                "the build layout may have changed upstream."
            )
        return build_dir
