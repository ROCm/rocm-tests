# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KFD device readiness and kfdtest invocation helpers.

Shared across KFD test suites. Provides device-node presence checks,
a uniform kfdtest invocation wrapper, and a crash-marker scanner that
catches hard GPU faults before the GTest summary is read.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_KFD_DEVICE = "/dev/kfd"

_CRASH_MARKERS: tuple[str, ...] = (
    "Memory access fault",
    "core dumped",
    "Segmentation fault",
    "HSA_STATUS_ERROR",
    "Aborted (",
    "Fatal error",
)


def ensure_kfd_device(executor) -> None:
    """Skip the test when ``/dev/kfd`` or the ``amdgpu`` kernel module is absent.

    Called at fixture setup time so suites skip cleanly rather than failing
    with an opaque device-not-found error mid-test.

    Args:
        executor: Any executor with a ``.run()`` method.
    """
    kfd_present = executor.run(f"test -c {_KFD_DEVICE}", timeout=15.0).ok
    if not kfd_present:
        pytest.skip(
            f"{_KFD_DEVICE} not found on the target node — " "the amdgpu kernel module must be loaded and KFD enabled."
        )
    module_loaded = executor.run("lsmod | grep -q amdgpu", timeout=15.0).ok
    if not module_loaded:
        pytest.skip("amdgpu kernel module is not loaded — KFD tests require it.")


def run_kfdtest(
    executor,
    binary: str,
    gtest_filter: str,
    ld_path: str,
    timeout: float = 600.0,
):
    """Invoke ``kfdtest`` with the given GTest filter and ``LD_LIBRARY_PATH``.

    Args:
        executor:     Any executor with a ``.run()`` method.
        binary:       Absolute path to the ``kfdtest`` binary.
        gtest_filter: GTest filter expression (e.g. ``"KFDSVMRangeTest.*"``).
        ld_path:      ``LD_LIBRARY_PATH`` value for the ROCm libraries.
        timeout:      Seconds before the run is abandoned.

    Returns:
        Execution result with ``.ok``, ``.stdout``, ``.stderr``, ``.exit_code``.
    """
    cmd = f"env LD_LIBRARY_PATH={ld_path} {binary} --gtest_filter={gtest_filter}"
    logger.info("Running kfdtest: %s", cmd)
    return executor.run(cmd, timeout=timeout)


def assert_kfd_no_crash_markers(output: str, label: str = "kfdtest") -> None:
    """Fail when the kfdtest output contains hard-crash signatures.

    GTest may exit 0 or 1 after a crash without surfacing the cause in the
    summary — this guard catches memory-access faults and similar hard GPU
    failures before the test result is interpreted.

    Args:
        output: Combined stdout + stderr from the kfdtest run.
        label:  Human-readable label for the failure message.

    Raises:
        AssertionError: When any crash marker is found in *output*.
    """
    found = [m for m in _CRASH_MARKERS if m in output]
    assert not found, f"{label} output contains hard-crash markers: {found}\n" f"Output tail:\n{output[-3000:]}"
