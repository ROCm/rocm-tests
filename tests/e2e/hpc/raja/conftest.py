# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Clone and HIP-build LLNL/RAJA for the functional unit suite.

RAJA is a third-party CMake project, so this uses the framework's
remote-transparent external-build primitives:

    * ``external_build.clone_repo``             -- idempotent git clone
    * ``external_build.assert_license_present`` -- provenance guard
    * a cmake configure/build runner with the ROCm environment injected via
      an ``env VAR=... cmd`` prefix rather than by mutating ``os.environ``.

``test_raja_functional`` then runs ctest on the GPU node. This is the
functional suite. It does not build or score RAJAPerf.
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
from tests.e2e.hpc.raja._workload import RAJA_REF, RAJA_URL

logger = logging.getLogger(__name__)

_BUILD_SENTINEL = "CTestTestfile.cmake"

_BUILD_SCRIPT = r"""
set -eu
export DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH="${ROCK}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PATH="${ROCK}/bin:${ROCK}/llvm/bin:${PATH}"

if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    git cmake g++ make ca-certificates openmpi-bin libopenmpi-dev libomp-dev
fi
if [ -d /usr/lib/x86_64-linux-gnu/openmpi ]; then
  export MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi
elif [ -d /usr/lib64/openmpi ]; then
  export MPI_HOME=/usr/lib64/openmpi
fi

if [ -x "${ROCK}/llvm/bin/amdclang++" ]; then
  cc="${ROCK}/llvm/bin/amdclang"
  cxx="${ROCK}/llvm/bin/amdclang++"
  hip_path="${ROCK}/llvm/bin"
else
  cc="${ROCK}/bin/amdclang"
  cxx="${ROCK}/bin/amdclang++"
  hip_path="${ROCK}/bin"
fi

git -C "${SRC}" submodule update --init --recursive

hip_root="${ROCK}"
if [ -d "${ROCK}/share/hip" ]; then
  hip_root="${ROCK}/share/hip"
fi
rm -rf "${BUILD}"
mkdir -p "${BUILD}"
cmake -S "${SRC}" -B "${BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_HIP_ARCHITECTURES="${GFX}" \
  -DROCM_ROOT_DIR="${ROCK}" \
  -DROCPRIM_DIR="${ROCK}" \
  -DHIP_ROOT_DIR="${hip_root}" \
  -DHIP_PATH="${hip_path}" \
  -DCMAKE_C_COMPILER="${cc}" \
  -DCMAKE_CXX_COMPILER="${cxx}" \
  -DGPU_TARGETS="${GFX}" \
  -DAMDGPU_TARGETS="${GFX}" \
  -DCMAKE_CXX_FLAGS="-munsafe-fp-atomics" \
  -DBLT_CXX_STD=c++20 \
  -DENABLE_MPI=ON \
  -DENABLE_OPENMP=ON \
  -DENABLE_HIP=ON \
  -DENABLE_CUDA=OFF \
  -DRAJA_COMPILER=RAJA_COMPILER_CLANG \
  -DCMAKE_CXX_FLAGS_RELEASE=-O2 \
  -DRAJA_HIPCC_FLAGS="-fPIC -O2"
cmake --build "${BUILD}" -j"${JOBS}"
"""


@dataclass(frozen=True)
class RajaBuild:
    """Result of a session-scoped RAJA HIP build.

    Attributes:
        build_dir: CMake build directory. The ctest suite runs here.
        source_dir: Cloned RAJA checkout.
    """

    build_dir: str
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
    """Run the RAJA configure/build script, streaming output."""
    logger.info("RAJA build -> %s", log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"RAJA build failed on remote (exit={result.exit_code}):\n"
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
            f"RAJA build failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


@pytest.fixture(scope="session")
def raja_functional_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    node_pool,
    require_gpu_arch_for,
) -> RajaBuild:
    """Clone RAJA ``develop`` and build the HIP unit-test configuration once."""
    if not rock_dir:
        pytest.fail("rock_dir is empty — pass --rock-dir or set ROCK_DIR to the ROCm install path.")

    resolved_gpu_arch = gpu_arch
    if not resolved_gpu_arch and node_pool:
        try:
            resolved_gpu_arch = next(gpu for alloc in node_pool._allocators.values() for gpu in alloc._pool).arch
        except (StopIteration, AttributeError):
            resolved_gpu_arch = None
    if not resolved_gpu_arch:
        require_gpu_arch_for("raja")

    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)

    ref_label = _safe_ref_name(RAJA_REF)
    dest = pathlib.Path(compiler_build_dir) / "raja" / f"raja-{ref_label}"
    raja_dir = str(external_build.clone_repo(url=RAJA_URL, dest=dest, ref=RAJA_REF, timeout=build_timeout))
    external_build.assert_license_present(raja_dir)

    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        raja_dir = str(cmake_executor.workspace_path_for(raja_dir))
    else:
        raja_dir = os.path.abspath(raja_dir)

    build_dir = f"{raja_dir}/raja_fn_build-{resolved_gpu_arch}"
    if _path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
        logger.info("RAJA: existing build tree at %s — skipping configure/build", build_dir)
        return RajaBuild(build_dir=build_dir, source_dir=raja_dir)

    jobs = resolve_parallel_jobs(remote_executor=cmake_executor)
    log_dir = os.path.join(framework_config.framework.artifact_dir, "raja")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "raja-build.log")
    cmd = (
        f"env ROCK={shlex.quote(rocm_path)} GFX={shlex.quote(resolved_gpu_arch)} "
        f"SRC={shlex.quote(raja_dir)} BUILD={shlex.quote(build_dir)} JOBS={jobs} "
        f"bash -lc {shlex.quote(_BUILD_SCRIPT)}"
    )
    logger.info("RAJA: configuring HIP build (gpu_arch=%s, rocm=%s)", resolved_gpu_arch, rocm_path)
    _run_build(cmd, cmake_executor=cmake_executor, timeout=build_timeout, log_path=log_path)
    return RajaBuild(build_dir=build_dir, source_dir=raja_dir)
