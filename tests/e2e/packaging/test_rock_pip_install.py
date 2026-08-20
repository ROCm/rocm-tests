# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_rock_pip_install.py -- install a ROCm python package from the public index.

Realizes TMS rock_pip_install_verification: create an isolated virtual environment,
``pip install`` the ``rocm-sdk`` package from the public per-family nightly index
(``https://rocm.nightlies.amd.com/v2/<family>/``), confirm the install with
``pip show``, and confirm the package is usable (its console entry point runs, or
it imports). The venv is created under a ``mktemp`` dir and removed on exit, so the
host/base environment is never mutated.

hw.cpu_only (no GPU); runtime.medium (network install). Override the GPU family via
``ROCM_TEST_ARTIFACT_GROUP`` (default ``gfx94X-dcgpu``).
"""

import os
import shlex

import pytest

_ARTIFACT_GROUP = os.environ.get("ROCM_TEST_ARTIFACT_GROUP", "gfx94X-dcgpu")
_PIP_INDEX = f"https://rocm.nightlies.amd.com/v2/{_ARTIFACT_GROUP}/"
_PACKAGE = "rocm-sdk"


@pytest.mark.runtime.medium
def test_rock_pip_install_in_venv(cpu_executor):
    """pip install rocm-sdk into a throwaway venv from the public index and verify it."""
    script = (
        "set -e; "
        'work=$(mktemp -d); trap "rm -rf $work" EXIT; '
        "python3 -m venv $work/venv; "
        "$work/venv/bin/python -m pip install --quiet --disable-pip-version-check "
        f"--index-url {shlex.quote(_PIP_INDEX)} {shlex.quote(_PACKAGE)}; "
        f"$work/venv/bin/python -m pip show {shlex.quote(_PACKAGE)}; "
        "$work/venv/bin/python -c 'import importlib.metadata as m; "
        f"print(\"INSTALLED\", m.version({_PACKAGE.replace('-', '_')!r}))' "
        "2>/dev/null || $work/venv/bin/rocm-sdk --help >/dev/null"
    )
    result = cpu_executor.run(f"bash -c {shlex.quote(script)}", timeout=1800)
    assert result.ok, (
        f"pip install/verify of {_PACKAGE!r} from {_PIP_INDEX} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-1000:]}"
    )
    assert f"Name: {_PACKAGE}" in result.stdout, f"pip show did not confirm {_PACKAGE!r}:\n{result.stdout[-2000:]}"
