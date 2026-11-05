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
import pathlib
import posixpath
import shlex

import pytest

from framework.executors.cpu_executor import CpuExecutor
from tests.e2e.hpc.gromacs._workload import GROMACS_REF, GROMACS_URL, GromacsBuild

# Re-export so pytest registers the shared session fixture for this directory.
# A nested conftest cannot declare pytest_plugins.
from tests.e2e.hpc.ucx.conftest import ucx_build as ucx_build

_OPENMPI_VERSION = "4.1.4"

# Build and install dirs are keyed by ROCm version and GPU arch, so a new ROCm
# or a different GPU never reuses a GROMACS binary built against another stack.
_KEY_SCRIPT = r"""
set -euo pipefail
ver=$(head -n1 "${ROCK}/.info/version" 2>/dev/null || basename "$(readlink -f "${ROCK}")")
gfx=$("${ROCK}/bin/rocminfo" | awk '/Name:[[:space:]]+gfx/{print $2; exit}')
gfx=${gfx%%:*}
test -n "$gfx"
printf '%s-%s' "$ver" "$gfx" | tr -c 'A-Za-z0-9._-' '_'
"""

_BUILD = r"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PATH="${MPI_HOME}/bin:${ROCK}/bin:${ROCK}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCK}/lib:${UCX_PREFIX}/lib:${MPI_HOME}/lib:${MPI_HOME}/lib64:${LD_LIBRARY_PATH:-}"
src="${GROMACS_SRC}"
build="${src}/build-mpi-${BUILD_KEY}"
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
    """Run a bash script on the build node through the framework executors.

    Remote runs use the session ``cmake_executor``. Local runs use a ``CpuExecutor``,
    the same local executor the criu fixtures use, with ``env`` merged on top.
    """
    if cmake_executor is not None:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        command = (
            f"env {prefix} bash -lc {shlex.quote(script)}"
            if prefix
            else f"bash -lc {shlex.quote(script)}"
        )
        result = cmake_executor.run(command, timeout=timeout, stream=True)
    else:
        local = CpuExecutor(env_overrides=env, suppress_output_log=True)
        result = local.run(f"bash -lc {shlex.quote(script)}", timeout=timeout)
    if not result.ok:
        text = (result.stdout or "") + "\n" + (result.stderr or "")
        raise RuntimeError(f"command failed (exit={result.exit_code}):\n{text[-4000:]}")
    return result


def _os_id(cmake_executor) -> str:
    """Return the build node's OS id from /etc/os-release."""
    result = _bash(cmake_executor, '. /etc/os-release; printf %s "$ID"', {}, 60)
    return (result.stdout or "").strip()


def _build_key(cmake_executor, rock_dir: str) -> str:
    """Return the ROCm-version-and-GPU-arch key the build script uses for its dirs."""
    result = _bash(cmake_executor, _KEY_SCRIPT, {"ROCK": rock_dir}, 120)
    key = (result.stdout or "").strip()
    if not key:
        raise RuntimeError("could not derive a ROCm/GPU build key for GROMACS")
    return key


@pytest.fixture(scope="session")
def gromacs_build(
    rock_dir: str,
    ucx_build: str,
    external_build,
    framework_config,
    cmake_executor,
    compiler_build_dir: str,
) -> GromacsBuild:
    """Clone and build GROMACS HIP once for every test in the session.

    Requesting ``ucx_build`` builds UCX through the shared hpc/ucx fixture.
    Open MPI is the framework's private 4.1.4 prefix. The checkout lives under
    ``compiler_build_dir`` and is keyed by ROCm version and GPU arch, so a
    different stack never reuses a binary built against another ROCm. A later
    session on the same stack reuses the tree when ``gmx`` and the ctest file
    are already present.
    """
    if not rock_dir:
        pytest.skip(
            "GROMACS build requires a ROCm install; pass --rock-dir / set ROCK_DIR"
        )
    os_id = _os_id(cmake_executor)
    if os_id not in {"ubuntu", "debian"}:
        pytest.skip(
            f"gromacs functional build uses apt; this host is {os_id or 'unknown'}"
        )

    build_timeout = float(framework_config.therock.build_timeout_secs)
    build_key = _build_key(cmake_executor, rock_dir)
    mpi = external_build.provision_openmpi_runtime(_OPENMPI_VERSION)
    mpi_home = mpi.env.get("MPI_HOME") or os.path.dirname(os.path.dirname(mpi.launcher))
    ucx_prefix = f"{ucx_build}/ucx"

    dest = pathlib.Path(compiler_build_dir) / "hpc-gromacs" / build_key / "gromacs"
    with external_build.build_lock(f"hpc-gromacs-{build_key}", timeout=build_timeout):
        source_dir = external_build.clone_repo(GROMACS_URL, dest, ref=GROMACS_REF)
        external_build.assert_license_present(source_dir)
        source = str(source_dir).replace("\\", "/")
        build_dir = posixpath.join(source, f"build-mpi-{build_key}")
        install_dir = posixpath.join(posixpath.dirname(source), f"install-{build_key}")
        _bash(
            cmake_executor,
            _BUILD,
            {
                "ROCK": rock_dir,
                "MPI_HOME": str(mpi_home),
                "UCX_PREFIX": ucx_prefix,
                "GROMACS_SRC": source,
                "GROMACS_INSTALL": install_dir,
                "BUILD_KEY": build_key,
            },
            build_timeout,
        )

    return GromacsBuild(
        build_dir=build_dir, mpi_home=str(mpi_home), ucx_prefix=ucx_prefix
    )
