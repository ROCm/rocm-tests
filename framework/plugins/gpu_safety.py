# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skip GPU/driver state-changing tests during shared execution."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit opt-in for an externally reserved host."""
    group = parser.getgroup("rocm-gpu-safety")
    group.addoption(
        "--gpu-execution-mode",
        choices=("shared", "exclusive"),
        default="shared",
        help=(
            "shared (default) skips gpu_state.mutating tests. "
            "exclusive permits them on an externally reserved host; "
            "this option does not reserve hardware or grant permissions."
        ),
    )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip marked tests before fixture setup in shared execution."""
    if item.config.getoption("--gpu-execution-mode") == "shared" and any(item.iter_markers(name="gpu_state.mutating")):
        pytest.skip(
            "gpu_state.mutating: GPU/driver state changes are disabled in shared suites. "
            "Use --gpu-execution-mode=exclusive only on an exclusively reserved host."
        )
