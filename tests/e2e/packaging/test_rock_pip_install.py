# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_rock_pip_install.py -- install a ROCm python package from the public index.

Realizes rock_pip_install_verification: create an isolated virtual environment,
``pip install`` the ``rocm-sdk`` package from the public per-family nightly index
(``https://rocm.nightlies.amd.com/v2/<family>/``), confirm the install with
``pip show``, then run ``rocm-sdk test`` — TheRock's bundled self check, which
loads the shared libraries and verifies the file/directory layout and API
contracts that a bare import does not exercise. The venv is created under a
``mktemp`` dir and removed on exit, so the host/base environment is never mutated.

hw.cpu_only (no GPU); runtime.medium (network install). The GPU family comes from
the ``artifact_group`` fixture, derived from the mandatory ``--gpu-arch``.
"""

import shlex

import pytest

_PACKAGE = "rocm-sdk"


@pytest.mark.runtime.medium
def test_rock_pip_install_in_venv(cpu_executor, artifact_group: str):
    """pip install rocm-sdk into a throwaway venv from the public index and verify it."""
    pip_index = f"https://rocm.nightlies.amd.com/v2/{artifact_group}/"
    script = (
        "set -e; "
        'work=$(mktemp -d); trap "rm -rf $work" EXIT; '
        "python3 -m venv $work/venv; "
        "$work/venv/bin/python -m pip install --quiet --disable-pip-version-check "
        f"--index-url {shlex.quote(pip_index)} {shlex.quote(_PACKAGE)}; "
        f"$work/venv/bin/python -m pip show {shlex.quote(_PACKAGE)}; "
        "$work/venv/bin/rocm-sdk test"
    )
    result = cpu_executor.run(f"bash -c {shlex.quote(script)}", timeout=1800)
    assert result.ok, (
        f"pip install/verify of {_PACKAGE!r} from {pip_index} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-1000:]}"
    )
    assert f"Name: {_PACKAGE}" in result.stdout, f"pip show did not confirm {_PACKAGE!r}:\n{result.stdout[-2000:]}"
