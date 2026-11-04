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

from tests.e2e.hpc.gromacs._workload import CTEST_TIMEOUT, RUN_TIMEOUT, GromacsBuild


def _os_id(target_executor) -> str:
    """Return the host OS id from /etc/os-release."""
    result = target_executor.run(
        "bash -lc '. /etc/os-release; printf %s \"$ID\"'",
        timeout=60,
    )
    assert result.ok, f"could not read /etc/os-release:\n{result.stdout}\n{result.stderr}"
    return (result.stdout or "").strip()


@pytest.mark.runtime.soak
def test_gromacs_functional(target_executor, rock_dir: str, gromacs_build: GromacsBuild):
    """GROMACS HIP MPI tests must pass ctest. The session fixture already built."""
    os_id = _os_id(target_executor)
    if os_id not in {"ubuntu", "debian"}:
        pytest.skip(f"gromacs functional build uses apt; this host is {os_id or 'unknown'}")
    mpi_bin = shlex.quote(gromacs_build.mpi_home + "/bin")
    libraries = ":".join(
        [
            shlex.quote(rock_dir + "/lib"),
            shlex.quote(gromacs_build.ucx_prefix + "/lib"),
            shlex.quote(gromacs_build.mpi_home + "/lib"),
            shlex.quote(gromacs_build.mpi_home + "/lib64"),
            "${LD_LIBRARY_PATH:-}",
        ]
    )
    command = "\n".join(
        [
            "set -euo pipefail",
            f"export PATH={mpi_bin}:$PATH",
            f"export LD_LIBRARY_PATH={libraries}",
            "export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
            "export OMPI_MCA_plm=isolated OMPI_MCA_btl=tcp,self OMPI_MCA_pml=ob1",
            "export OMPI_MCA_rmaps_base_oversubscribe=1 OMPI_MCA_hwloc_base_binding_policy=none",
            "ctest --test-dir "
            f"{shlex.quote(gromacs_build.build_dir)} --output-on-failure --timeout {shlex.quote(str(CTEST_TIMEOUT))}",
        ]
    )
    result = target_executor.run(f"bash -lc {shlex.quote(command)}", timeout=RUN_TIMEOUT)
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.ok, f"gromacs functional tests failed:\n{text[-4000:]}"
    assert "The following tests FAILED" not in text, text[-4000:]
