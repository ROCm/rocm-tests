# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- RVS fixtures for amd-smi tests that need a GPU load.

pytest only auto-loads ``conftest.py`` for its own directory tree, so the RVS
build and config-lookup fixtures from ``tests/e2e/rvs`` are re-exported here to
make them requestable from this suite.
"""

from __future__ import annotations

from tests.e2e.rvs.conftest import (  # noqa: F401
    rvs_binary as _rvs_binary,
    rvs_env,
    rvs_find_conf,
    rvs_source as _rvs_source,
)

rvs_binary = _rvs_binary
rvs_source = _rvs_source
