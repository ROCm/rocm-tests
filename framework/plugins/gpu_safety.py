# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Skip GPU/driver state-changing tests unless mutations are explicitly allowed."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register explicit opt-in for GPU state mutations."""
    group = parser.getgroup("rocm-gpu-safety")
    group.addoption(
        "--gpu-state-mutations",
        choices=("skip", "allow"),
        default="skip",
        help=(
            "skip (default) skips gpu_state.mutating tests. "
            "allow permits them; use only on an exclusively reserved host. "
            "This option does not reserve hardware or grant permissions."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Reject mutation opt-in when xdist workers could run tests concurrently."""
    if config.getoption("--gpu-state-mutations") != "allow":
        return

    numprocesses = getattr(config.option, "numprocesses", None)
    if numprocesses not in (None, 0, "0", "no", ""):
        raise pytest.UsageError(
            "--gpu-state-mutations=allow cannot be used with pytest-xdist workers. "
            "Run mutation tests without -n (or with -n 0) on an exclusively reserved host."
        )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip marked tests before fixture setup when GPU state mutations are disabled."""
    if item.config.getoption("--gpu-state-mutations") == "skip" and any(item.iter_markers(name="gpu_state.mutating")):
        pytest.skip(
            "gpu_state.mutating: GPU/driver state mutations are disabled. "
            "Use --gpu-state-mutations=allow only on an exclusively reserved host."
        )
