# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Clone and HIP-build QMCPACK for the functional deterministic suite.

QMCPACK is a third-party CMake project, so this uses the framework's
remote-transparent external-build primitives:

    * ``external_build.clone_repo``             -- idempotent git clone
    * ``external_build.assert_license_present`` -- provenance guard
    * a cmake configure/build runner with the ROCm environment injected via
      an ``env VAR=... cmd`` prefix rather than by mutating ``os.environ``.

``test_qmcpack_functional`` then runs ``ctest -R deterministic`` on each of
the four OpenMP+HIP configurations. This is the functional suite. It does
not collect or compare QMCPACK performance scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import pathlib
import re
import shlex

import pytest

from framework.builder.binary_builder import resolve_parallel_jobs
from framework.executors.background_process import _blocking_stream_run
from tests.e2e.hpc.qmcpack._workload import BUILD_TIMEOUT, CONFIGS, QMCPACK_REF, QMCPACK_URL

logger = logging.getLogger(__name__)

_BUILD_SENTINEL = "CTestTestfile.cmake"

_BUILD_SCRIPT = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH="${ROCK}/lib:${ROCK}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${ROCK}/bin:${ROCK}/llvm/bin:${ROCK}/lib/llvm/bin:${PATH}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    git cmake g++ make ca-certificates python3 \
    libxml2-dev libhdf5-dev zlib1g-dev libopenblas-dev liblapack-dev \
    libboost-dev libboost-all-dev libomp-dev libfftw3-dev
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y \
    git cmake gcc-c++ make ca-certificates python3 \
    boost-devel lapack-devel openblas-devel hdf5-devel fftw3-devel \
    libxml2-devel zlib-devel
elif command -v zypper >/dev/null 2>&1; then
  zypper --non-interactive install \
    git cmake gcc-c++ make ca-certificates python3 \
    boost-devel lapack-devel openblas-devel hdf5-devel fftw3-devel \
    libxml2-devel zlib-devel
fi

# Same candidate order as find_rocm_clangpp: TheRock flattened layout, then
# the standard ROCm llvm/bin tree, then bin/. Prefer amdclang++ when that
# name is present; QMCPACK's OpenMP+HIP configure uses it as the C++ compiler.
cc=""
cxx=""
for bindir in "${ROCK}/lib/llvm/bin" "${ROCK}/llvm/bin" "${ROCK}/bin"; do
  if [ -x "${bindir}/amdclang++" ]; then
    cc="${bindir}/amdclang"
    cxx="${bindir}/amdclang++"
    break
  fi
  if [ -x "${bindir}/clang++" ]; then
    cc="${bindir}/clang"
    cxx="${bindir}/clang++"
    break
  fi
done
if [ -z "${cxx}" ]; then
  echo "ROCm clang not found under ${ROCK}" >&2
  exit 1
fi

# QMC_CONFIGS is "name:complex:mixed" entries separated by spaces, from CONFIGS.
IFS=' ' read -r -a specs <<< "${QMC_CONFIGS}"
for spec in "${specs[@]}"; do
  name="${spec%%:*}"
  rest="${spec#*:}"
  complex="${rest%%:*}"
  mixed="${rest#*:}"
  build="${SRC}/build_${name}-${ARCH_LABEL}"
  rm -rf "${build}"
  mkdir -p "${build}"
  # Nexus generation during cmake can fail when a user-site package prints
  # DeprecationWarning into the configure Python subprocess.
  # QMC_OFFLOAD_MEM_ASSOCIATED is left unset: current QMCPACK only honors it
  # for CUDA or SYCL, and forces it off for this HIP build.
  PYTHONWARNINGS="ignore::DeprecationWarning" \
    cmake -S "${SRC}" -B "${build}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DROCM_PATH="${ROCK}" \
      -DCMAKE_C_COMPILER="${cc}" \
      -DCMAKE_CXX_COMPILER="${cxx}" \
      -DQMC_MPI=OFF \
      "-DQMC_GPU=openmp;hip" \
      -DQMC_GPU_ARCHS="${GFX}" \
      -DQMC_COMPLEX="${complex}" \
      -DQMC_MIXED_PRECISION="${mixed}"
  cmake --build "${build}" -j"${JOBS}"
done
"""


@dataclass(frozen=True)
class QmcpackBuild:
    """Result of a session-scoped QMCPACK HIP build.

    Attributes:
        build_dirs: One CMake build directory per complex/precision configuration.
        source_dir: Cloned QMCPACK checkout.
    """

    build_dirs: tuple[str, ...]
    source_dir: str


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _path_exists(path: str, cmake_executor) -> bool:
    """Return True when *path* is a file, locally or on the build node."""
    if cmake_executor is not None:
        return bool(cmake_executor.run(f"test -f {shlex.quote(path)}", timeout=30.0).ok)
    return os.path.isfile(path)


def _run_build(cmd: str, *, cmake_executor, timeout: float, log_path: str) -> None:
    """Run the QMCPACK configure/build script, streaming output."""
    logger.info("QMCPACK build -> %s", log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"QMCPACK build failed on remote (exit={result.exit_code}):\n"
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
            f"QMCPACK build failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


@pytest.fixture(scope="session")
def qmcpack_functional_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    node_pool,
    require_gpu_arch_for,
) -> QmcpackBuild:
    """Clone QMCPACK ``develop`` and build the four HIP configurations once."""
    if not rock_dir:
        pytest.fail("rock_dir is empty — pass --rock-dir or set ROCK_DIR to the ROCm install path.")

    resolved_gpu_arch = gpu_arch
    if not resolved_gpu_arch and node_pool:
        try:
            resolved_gpu_arch = next(gpu for alloc in node_pool._allocators.values() for gpu in alloc._pool).arch
        except (StopIteration, AttributeError):
            resolved_gpu_arch = None
    if not resolved_gpu_arch:
        require_gpu_arch_for("qmcpack")
    resolved_gpu_arch = resolved_gpu_arch.split(":", 1)[0]

    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = max(float(framework_config.therock.build_timeout_secs), BUILD_TIMEOUT)

    ref_label = _safe_ref_name(QMCPACK_REF)
    dest = pathlib.Path(compiler_build_dir) / "qmcpack" / f"qmcpack-{ref_label}"
    qmc_dir = str(external_build.clone_repo(url=QMCPACK_URL, dest=dest, ref=QMCPACK_REF, timeout=build_timeout))
    external_build.assert_license_present(qmc_dir)

    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        qmc_dir = str(cmake_executor.workspace_path_for(qmc_dir))
    else:
        qmc_dir = os.path.abspath(qmc_dir)

    arch_label = _safe_ref_name(resolved_gpu_arch)
    build_dirs = tuple(f"{qmc_dir}/build_{name}-{arch_label}" for name, _complex, _mixed in CONFIGS)
    if all(_path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor) for build_dir in build_dirs):
        logger.info("QMCPACK: existing build trees under %s — skipping configure/build", qmc_dir)
        return QmcpackBuild(build_dirs=build_dirs, source_dir=qmc_dir)

    jobs = resolve_parallel_jobs(remote_executor=cmake_executor)
    log_dir = os.path.join(framework_config.framework.artifact_dir, "qmcpack")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "qmcpack-build.log")
    config_specs = " ".join(f"{name}:{complex_}:{mixed}" for name, complex_, mixed in CONFIGS)
    cmd = (
        f"env ROCK={shlex.quote(rocm_path)} GFX={shlex.quote(resolved_gpu_arch)} "
        f"ARCH_LABEL={shlex.quote(arch_label)} "
        f"SRC={shlex.quote(qmc_dir)} JOBS={jobs} "
        f"QMC_CONFIGS={shlex.quote(config_specs)} "
        f"bash -lc {shlex.quote(_BUILD_SCRIPT)}"
    )
    logger.info("QMCPACK: configuring HIP builds (gpu_arch=%s, rocm=%s)", resolved_gpu_arch, rocm_path)
    _run_build(cmd, cmake_executor=cmake_executor, timeout=build_timeout, log_path=log_path)
    return QmcpackBuild(build_dirs=build_dirs, source_dir=qmc_dir)
