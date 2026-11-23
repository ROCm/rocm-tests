# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MIOpen driver invocation helpers shared across MIOpen test suites.

Provides a binary availability guard and an executor-transparent invocation
helper that always sets the standard MIOpen test environment.  Shared across
``miopen/`` and ``rocm_libs/test_miopen_np_nts_tensors.py``.
"""

from __future__ import annotations

import logging
import shlex

import pytest

logger = logging.getLogger(__name__)

_MIOPEN_DRIVER_REL = "bin/MIOpenDriver"


def require_miopen_driver(rock_dir: str, executor) -> str:
    """Fail early when ``MIOpenDriver`` is absent or non-executable on the executor.

    Works transparently for local and remote (SSH) executors because the check
    runs through the executor — the binary lives on the execution node, not the
    controller host in remote-node mode.

    Args:
        rock_dir:  TheRock/ROCm install root.
        executor:  Any executor with a ``.run()`` method.

    Returns:
        Absolute path to the ``MIOpenDriver`` binary.
    """
    driver = f"{rock_dir}/{_MIOPEN_DRIVER_REL}"
    result = executor.run(f"test -x {shlex.quote(driver)}", timeout=15.0)
    if not result.ok:
        pytest.fail(
            f"MIOpenDriver not found or not executable at {driver}. "
            "Ensure MIOpen is installed and --rock-dir points to the correct ROCm path."
        )
    return driver


def run_miopen_driver(
    executor,
    driver: str,
    ld_library_path: str,
    args: str,
    timeout: float = 300.0,
):
    """Invoke ``MIOpenDriver`` with the standard MIOpen test environment.

    Always sets ``MIOPEN_FIND_MODE=3`` (fastest find mode, avoids kernel-find
    lock contention under xdist) and ``MIOPEN_DISABLE_CACHE=1`` (prevents stale
    cache entries from masking regressions).

    Args:
        executor:          Any executor with a ``.run()`` method.
        driver:            Absolute path to ``MIOpenDriver`` binary.
        ld_library_path:   Value for ``LD_LIBRARY_PATH``.
        args:              Conv/pool/etc arguments forwarded verbatim to MIOpenDriver.
        timeout:           Seconds before the run is abandoned.

    Returns:
        Execution result with ``.ok``, ``.stdout``, ``.stderr``, ``.exit_code``.
    """
    ld = shlex.quote(ld_library_path)
    d = shlex.quote(driver)
    cmd = f"env LD_LIBRARY_PATH={ld} MIOPEN_FIND_MODE=3 MIOPEN_DISABLE_CACHE=1 {d} {args}"
    logger.info("Running MIOpenDriver: %s", cmd)
    return executor.run(cmd, timeout=timeout)
