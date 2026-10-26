# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Fixtures for tests/e2e/system_tools/amd_smi/.

Resolves the amd-smi binary, verifies metric/node subcommands are available,
and builds the CoralGemm workload binary for the power-under-load test.

pytest only auto-loads ``conftest.py`` for its own directory tree, so the RVS
build and config-lookup fixtures from the sibling ``tests/e2e/system_tools/rvs``
suite are re-exported here to make them requestable from this suite.
"""

from dataclasses import dataclass
import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess

import pytest

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
        if target_executor.run(f"test -x {candidate}").ok:
            return candidate

    which = target_executor.run("command -v amd-smi")
    resolved = (which.stdout or "").strip().splitlines()
    if which.ok and resolved:
        return shlex.quote(resolved[-1].strip())

    pytest.fail(
        f"amd-smi found neither at {rock_dir}/bin/amd-smi nor on PATH. It ships with "
        "ROCm, so a node without it is misconfigured rather than out of scope."
    )
