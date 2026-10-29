# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Functional GROMACS HIP tests.

Uses the functional-suite defaults: HIP feature-enablement branch, MPI on,
and build prerequisites (UCX plus Open MPI 4.1.4). ctest must pass. This is
not the GROMACS performance benchmark. Regression-test download stays off.
"""

from __future__ import annotations

import shlex

import pytest

from tests.e2e.hpc.gromacs._workload import (
    CTEST_TIMEOUT,
    GROMACS_REF,
    GROMACS_URL,
    OPENMPI_URL,
    RUN_TIMEOUT,
    UCX_REF,
    UCX_URL,
)

_SCRIPT = r"""
set -eu
export DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH="$ROCK/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ROCK/bin:$ROCK/llvm/bin:$PATH"
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
log=/tmp/hpc-gromacs.log
: > "$log"
trap 'echo "--- log tail ---"; tail -n 120 "$log"' EXIT
run() { echo "+ $*" | tee -a "$log"; "$@" >>"$log" 2>&1; }

gfx=$("$ROCK/bin/rocminfo" | awk '/Name:[[:space:]]+gfx/{print $2; exit}')
test -n "$gfx"
echo "GPU arch: $gfx" | tee -a "$log"
. /etc/os-release
case "$ID" in
  ubuntu|debian) ;;
  *) echo "gromacs functional port runs on Ubuntu/Debian, not $ID" >&2; exit 1 ;;
esac

apt-get update -qq
apt-get install -y --no-install-recommends \
  git cmake g++ make ca-certificates wget \
  autoconf automake libtool m4 pkg-config flex \
  libfftw3-dev libnuma-dev python3 openssh-client

prefix=/tmp/hpc-gromacs
rm -rf "$prefix"
mkdir -p "$prefix"
ucx_src="$prefix/ucx-src"
ucx_inst="$prefix/ucx"
git clone --depth 1 --branch __UCX_REF__ __UCX_URL__ "$ucx_src"
( cd "$ucx_src" && run ./autogen.sh )
mkdir -p "$ucx_src/build"
(
  cd "$ucx_src/build"
  run "$ucx_src/contrib/configure-opt" --prefix="$ucx_inst" --with-rocm="$ROCK"
  run make -j"$(nproc)"
  run make install
)

ompi_inst="$prefix/openmpi"
wget -q -O "$prefix/openmpi.tar.gz" __OPENMPI_URL__
tar -xzf "$prefix/openmpi.tar.gz" -C "$prefix"
ompi_src="$prefix/openmpi-4.1.4"
mkdir -p "$ompi_src/build"
(
  cd "$ompi_src/build"
  run "$ompi_src/configure" --prefix="$ompi_inst" \
    --with-ucx="$ucx_inst" --with-hwloc=internal --enable-mca-no-build=btl-uct
  run make -j"$(nproc)"
  run make install
)
export PATH="$ompi_inst/bin:$ucx_inst/bin:$PATH"
export LD_LIBRARY_PATH="$ompi_inst/lib:$ompi_inst/lib64:$ucx_inst/lib:${LD_LIBRARY_PATH:-}"

src="$prefix/gromacs"
git clone --branch __GROMACS_REF__ __GROMACS_URL__ "$src"
build="$src/build-mpi"
rm -rf "$build"
mkdir -p "$build"
install_dir="$prefix/install"
run cmake -S "$src" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGMX_BUILD_OWN_FFTW=ON \
  -DCMAKE_C_COMPILER=gcc \
  -DCMAKE_CXX_COMPILER=g++ \
  -DGMX_MPI=ON \
  -DGMX_HIP_TARGET_ARCH="$gfx" \
  -DGMX_GPU=HIP \
  -DCMAKE_PREFIX_PATH="$ROCK" \
  -DREGRESSIONTEST_DOWNLOAD=OFF \
  -DCMAKE_INSTALL_PREFIX="$install_dir" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
run cmake --build "$build" -j"$(nproc)"
run cmake --install "$build"
export PATH="$install_dir/bin:$PATH"
export LD_LIBRARY_PATH="$install_dir/lib:$install_dir/lib64:${LD_LIBRARY_PATH:-}"
# The container has no sshd. Open MPI's default rsh/ssh launcher aborts every
# MPI test. isolated forks ranks on this node.
export OMPI_MCA_plm=isolated
export OMPI_MCA_btl=tcp,self
export OMPI_MCA_pml=ob1
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_hwloc_base_binding_policy=none
run cmake --build "$build" --target tests -j"$(nproc)"
set +e
run ctest --test-dir "$build" --output-on-failure --timeout __CTEST_TIMEOUT__
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "===== ctest failure excerpt ====="
  grep -n -m 40 -E "MCA parameter|Error|FATAL|FAILED|hipError" "$log" || true
fi
exit "$rc"
"""


def _script() -> str:
    """Fill the pinned URLs and timeouts into the build script."""
    return (
        _SCRIPT.replace("__UCX_REF__", shlex.quote(UCX_REF))
        .replace("__UCX_URL__", shlex.quote(UCX_URL))
        .replace("__OPENMPI_URL__", shlex.quote(OPENMPI_URL))
        .replace("__GROMACS_REF__", shlex.quote(GROMACS_REF))
        .replace("__GROMACS_URL__", shlex.quote(GROMACS_URL))
        .replace("__CTEST_TIMEOUT__", shlex.quote(str(CTEST_TIMEOUT)))
    )


def _os_id(target_executor) -> str:
    """Return the host OS id from /etc/os-release."""
    result = target_executor.run(
        "bash -lc '. /etc/os-release; printf %s \"$ID\"'",
        timeout=60,
    )
    assert result.ok, f"could not read /etc/os-release:\n{result.stdout}\n{result.stderr}"
    return (result.stdout or "").strip()


@pytest.mark.runtime.soak
def test_gromacs_functional(target_executor, rock_dir: str):
    """GROMACS HIP MPI tests must build and pass ctest."""
    os_id = _os_id(target_executor)
    if os_id not in {"ubuntu", "debian"}:
        pytest.skip(f"gromacs functional build uses apt; this host is {os_id or 'unknown'}")
    result = target_executor.run(
        f"env ROCK={shlex.quote(rock_dir)} bash -lc {shlex.quote(_script())}",
        timeout=RUN_TIMEOUT,
    )
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.ok, f"gromacs functional tests failed:\n{text[-4000:]}"
    assert "The following tests FAILED" not in text, text[-4000:]
