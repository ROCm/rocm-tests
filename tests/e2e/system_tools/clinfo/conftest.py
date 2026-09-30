# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- clinfo test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def clinfo_requires(target_executor, rock_dir: str) -> None:
    """Fail the test when clinfo is not available on the execution node.
    This fixture ensures the test only runs when clinfo can be found and executed.
    """
    check_cmd = f"[ -x {rock_dir}/bin/clinfo ] && echo found || which clinfo >/dev/null 2>&1"
    result = target_executor.run(check_cmd)
    if not result.ok:
        pytest.fail("clinfo not found in rock_dir/bin/ or on PATH")
