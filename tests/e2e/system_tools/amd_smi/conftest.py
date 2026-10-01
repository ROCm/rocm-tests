# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- RVS fixtures for amd-smi tests that need a GPU load.

pytest only auto-loads ``conftest.py`` for its own directory tree, so the RVS
build and config-lookup fixtures from the sibling ``tests/e2e/system_tools/rvs``
suite are re-exported here to make them requestable from this suite.
"""

from __future__ import annotations

import shlex

import pytest
from tests.e2e.system_tools.rvs.conftest import (  # noqa: F401
    rvs_binary as _rvs_binary,
    rvs_env,
    rvs_find_conf,
    rvs_source as _rvs_source,
)

rvs_binary = _rvs_binary
rvs_source = _rvs_source

# Ceiling for resolving the binary. Both probes are sub-second, so anything
# approaching this means the connection to the node is wedged.
_PROBE_TIMEOUT = 60.0


@pytest.fixture
def amd_smi_binary(target_executor, rock_dir: str) -> str:
    """Return a shell-quoted ``amd-smi`` belonging to the ROCm under test.

    Prefers ``<rock_dir>/bin/amd-smi`` so a host carrying several ROCm installs
    reports telemetry for the one ``--rock-dir`` selected rather than whichever
    copy happens to be first on ``PATH``. Both probes run through
    ``target_executor``, so a ``--remote-node`` run resolves the binary on the
    node that will be sampled instead of on the machine running pytest.
    """
    if rock_dir:
        candidate = shlex.quote(f"{rock_dir.rstrip('/')}/bin/amd-smi")
        if target_executor.run(f"test -x {candidate}", timeout=_PROBE_TIMEOUT).ok:
            return candidate

    which = target_executor.run("command -v amd-smi", timeout=_PROBE_TIMEOUT)
    resolved = (which.stdout or "").strip().splitlines()
    if which.ok and resolved:
        return shlex.quote(resolved[-1].strip())

    pytest.fail(
        f"amd-smi found neither at {rock_dir}/bin/amd-smi nor on PATH. It ships with "
        "ROCm, so a node without it is misconfigured rather than out of scope."
    )
