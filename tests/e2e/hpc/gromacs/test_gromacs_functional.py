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

from tests.e2e.hpc.gromacs._workload import RUN_TIMEOUT


def _os_id(target_executor) -> str:
    """Return the host OS id from /etc/os-release."""
    result = target_executor.run(
        "bash -lc '. /etc/os-release; printf %s \"$ID\"'",
        timeout=60,
    )
    assert result.ok, f"could not read /etc/os-release:\n{result.stdout}\n{result.stderr}"
    return (result.stdout or "").strip()


@pytest.mark.runtime.soak
def test_gromacs_functional(target_executor, rock_dir: str, gromacs_build_script: str):
    """GROMACS HIP MPI tests must build and pass ctest."""
    os_id = _os_id(target_executor)
    if os_id not in {"ubuntu", "debian"}:
        pytest.skip(f"gromacs functional build uses apt; this host is {os_id or 'unknown'}")
    result = target_executor.run(
        f"env ROCK={shlex.quote(rock_dir)} bash -lc {shlex.quote(gromacs_build_script)}",
        timeout=RUN_TIMEOUT,
    )
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    assert result.ok, f"gromacs functional tests failed:\n{text[-4000:]}"
    assert "The following tests FAILED" not in text, text[-4000:]
