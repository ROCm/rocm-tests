# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build GROMACS with HIP once per session and return the ctest directory.

UCX comes from the shared session fixture in ``tests/e2e/hpc/ucx/conftest.py``
(configure-release). Open MPI 4.1.4 comes from
``external_build.provision_openmpi_runtime``, which is the framework's
CPU-only prefix. This module does not clone or configure either of those
again. ctest keeps the isolated TCP BTL, so the run does not need a second
UCX-enabled Open MPI. The GROMACS tree is taken under
``external_build.build_lock`` so xdist workers do not share the checkout.
"""

from __future__ import annotations

import os
import posixpath
import shlex
import subprocess  # nosec B404 - local bash invocation when there is no remote executor

import pytest

from tests.e2e.hpc.gromacs._workload import GROMACS_REF, GROMACS_URL, GromacsBuild

# Re-export so pytest registers the shared session fixture for this directory.
# A nested conftest cannot declare pytest_plugins.
from tests.e2e.hpc.ucx.conftest import ucx_build as ucx_build

_OPENMPI_VERSION = "4.1.4"
_PREFIX = "/tmp/hpc-gromacs"  # nosec B108 - shared build dir, serialized by build_lock

_BUILD = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="${MPI_HOME}/bin:${ROCK}/bin:${ROCK}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCK}/lib:${UCX_PREFIX}/lib:${MPI_HOME}/lib:${MPI_HOME}/lib64:${LD_LIBRARY_PATH:-}"
src="${GROMACS_SRC}"
build="${src}/build-mpi"
sentinel="${build}/tests/CTestTestfile.cmake"
if [ -x "${build}/bin/gmx" ] && [ -f "${sentinel}" ]; then
  echo "reusing GROMACS build at ${build}"
  exit 0
fi
apt-get update -qq
apt-get install -y --no-install-recommends \
  git cmake g++ make ca-certificates python3 openssh-client
gfx=$("${ROCK}/bin/rocminfo" | awk '/Name:[[:space:]]+gfx/{print $2; exit}')
gfx=${gfx%%:*}
test -n "$gfx"
echo "GPU arch: $gfx"
mkdir -p "${build}" "${GROMACS_INSTALL}"
cmake -S "${src}" -B "${build}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DGMX_MPI=ON \
  -DMPI_C_COMPILER="${MPI_HOME}/bin/mpicc" \
  -DMPI_CXX_COMPILER="${MPI_HOME}/bin/mpicxx" \
  -DGMX_HIP_TARGET_ARCH="${gfx}" \
  -DGMX_GPU=HIP \
  -DCMAKE_PREFIX_PATH="${MPI_HOME};${ROCK}" \
  -DREGRESSIONTEST_DOWNLOAD=OFF \
  -DCMAKE_INSTALL_PREFIX="${GROMACS_INSTALL}" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build "${build}" -j"$(nproc)"
cmake --install "${build}"
cmake --build "${build}" --target tests -j"$(nproc)"
"""


def _bash(cmake_executor, script: str, env: dict[str, str], timeout: float):
    """Run a bash script on the cmake node, or locally when there is no remote."""
    if cmake_executor is not None:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        command = f"env {prefix} bash -lc {shlex.quote(script)}" if prefix else f"bash -lc {shlex.quote(script)}"
        result = cmake_executor.run(command, timeout=timeout, stream=True)
        if not result.ok:
            text = (result.stdout or "") + "\n" + (result.stderr or "")
            raise RuntimeError(f"command failed (exit={result.exit_code}):\n{text[-4000:]}")
        return result
    # Arguments are fixed framework scripts, not a shell string.
    proc = subprocess.run(  # nosec B603 B607
        ["bash", "-lc", script],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        raise RuntimeError(f"command failed (exit={proc.returncode}):\n{text[-4000:]}")
    return proc


def _os_id(cmake_executor) -> str:
    """Return the build node's OS id from /etc/os-release."""
    result = _bash(cmake_executor, '. /etc/os-release; printf %s "$ID"', {}, 60)
    return (result.stdout or "").strip()


@pytest.fixture(scope="session")
def gromacs_build(
    rock_dir: str,
    ucx_build: str,
    external_build,
    framework_config,
    cmake_executor,
) -> GromacsBuild:
    """Clone and build GROMACS HIP once for every test in the session.

    Requesting ``ucx_build`` builds UCX through the shared hpc/ucx fixture.
    Open MPI is the framework's private 4.1.4 prefix. A later session reuses
    the tree when ``gmx`` and the ctest file are already present.
    """
    if not rock_dir:
        pytest.skip("GROMACS build requires a ROCm install; pass --rock-dir / set ROCK_DIR")
    os_id = _os_id(cmake_executor)
    if os_id not in {"ubuntu", "debian"}:
        pytest.skip(f"gromacs functional build uses apt; this host is {os_id or 'unknown'}")

    build_timeout = float(framework_config.therock.build_timeout_secs)
    mpi = external_build.provision_openmpi_runtime(_OPENMPI_VERSION)
    mpi_home = mpi.env.get("MPI_HOME") or os.path.dirname(os.path.dirname(mpi.launcher))
    ucx_prefix = f"{ucx_build}/ucx"

    with external_build.build_lock("hpc-gromacs", timeout=build_timeout):
        source_dir = external_build.clone_repo(GROMACS_URL, f"{_PREFIX}/gromacs", ref=GROMACS_REF)
        external_build.assert_license_present(source_dir)
        source = str(source_dir).replace("\\", "/")
        build_dir = posixpath.join(source, "build-mpi")
        install_dir = posixpath.join(posixpath.dirname(source), "install")
        _bash(
            cmake_executor,
            _BUILD,
            {
                "ROCK": rock_dir,
                "MPI_HOME": str(mpi_home),
                "UCX_PREFIX": ucx_prefix,
                "GROMACS_SRC": source,
                "GROMACS_INSTALL": install_dir,
            },
            build_timeout,
        )

    return GromacsBuild(build_dir=build_dir, mpi_home=str(mpi_home), ucx_prefix=ucx_prefix)
