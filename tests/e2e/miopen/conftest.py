# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for the MIOpen driver test area."""

import pytest

from tests.common.miopen.runner import require_miopen_driver as _require_miopen_driver


@pytest.fixture(autouse=True)
def require_miopen_driver(rock_dir: str, target_executor) -> None:
    """Fail early if MIOpenDriver binary is absent or not executable.

    Function-scoped (not session) because target_executor is function-scoped;
    a session fixture cannot consume a function-scoped fixture.

    The check runs through target_executor so it works for both local and
    remote (SSH) executors — the binary lives on the execution node, which is
    not the controller host in remote-node mode.
    """
    _require_miopen_driver(rock_dir, target_executor)
