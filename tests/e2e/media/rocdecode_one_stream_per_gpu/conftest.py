# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixture for the rocDecode one-stream-per-partition suite.

This directory lives at ``tests/e2e/media/rocdecode_one_stream_per_gpu/``. The
fixture builds the three mode binaries in one CMake invocation:

    output/test-binaries/rocdecode_one_stream_per_gpu/build/rocdecode_spx_one_stream_per_card
    output/test-binaries/rocdecode_one_stream_per_gpu/build/rocdecode_dpx_one_stream_per_half
    output/test-binaries/rocdecode_one_stream_per_gpu/build/rocdecode_cpx_one_stream_per_slice

``xcd_burn`` is built beside them and is used only by the optional ``smi_one_xcd``
check, which the pytest module does not run.
"""

from __future__ import annotations

import logging
import os
import time

import pytest

logger = logging.getLogger(__name__)

_SRC_DIR = "tests/e2e/media/rocdecode_one_stream_per_gpu"
_COMPUTE_FILE = "current_compute_partition"
_SWITCH_TIMEOUT_SECS = 180.0
_SWITCH_AFTER_ERROR_SECS = 30.0
_POLL_SECS = 2.0
_BINARIES = (
    "rocdecode_spx_one_stream_per_card",
    "rocdecode_dpx_one_stream_per_half",
    "rocdecode_cpx_one_stream_per_slice",
)


@pytest.fixture(scope="session")
def rocdecode_one_stream_binaries(
    gpu_arch: str | None,
    cmake_build_dir,
    built_binary,
) -> dict[str, str]:
    """Build the SPX, DPX, and CPX binaries; return ``{name: absolute path}``."""
    build_dir = cmake_build_dir(
        src=_SRC_DIR,
        subdir="rocdecode_one_stream_per_gpu",
        gpu_arch=gpu_arch,
        compiler_mode="auto",
        label="rocdecode_one_stream_per_gpu",
        sync_dirs=[_SRC_DIR],
        artifact=_BINARIES[-1],
    )
    return {name: built_binary(os.path.join(build_dir, name), name) for name in _BINARIES}


def _clear_partition_cache(config: pytest.Config) -> None:
    """Drop the plugin's session cache so the next test re-reads sysfs.

    ``partition_plugin`` stores the first read for the whole session. After this
    suite changes the mode, that cached value would skip the next group.
    """
    cache = getattr(config, "_partition_mode_cache", None)
    if isinstance(cache, dict):
        cache.clear()


def _node_run(node_pool, command: str, timeout: float):
    """Run ``command`` on the remote node, or locally when pytest is not remote."""
    if node_pool is not None:
        for spec in node_pool.node_specs:
            ssh = node_pool._ssh_sessions.get(spec.label)
            if ssh is not None:
                return ssh.run(command, timeout=timeout)
    from framework.common.helpers import ExecutionResult  # pylint: disable=import-outside-toplevel
    from framework.executors.local_executor import (  # pylint: disable=import-outside-toplevel
        run_cmd_get_stdout_stderr,
    )

    code, out, err = run_cmd_get_stdout_stderr("bash", "-lc", command, timeout=timeout)
    return ExecutionResult(exit_code=code, stdout=out, stderr=err)


def _read_compute_modes(node_pool) -> set[str]:
    """Return the distinct compute-partition values reported by sysfs."""
    command = f'for f in /sys/bus/pci/devices/*/{_COMPUTE_FILE}; do [ -r "$f" ] && cat "$f"; done'
    result = _node_run(node_pool, command, timeout=30.0)
    return {line.strip().lower() for line in (result.stdout or "").splitlines() if line.strip()}


def _apply_compute_mode(node_pool, mode: str) -> None:
    """Set every GPU to ``mode`` and wait until sysfs agrees. No-op if already set."""
    if mode not in {"spx", "dpx", "tpx", "qpx", "cpx"}:
        pytest.fail(f"refusing to set unknown compute partition {mode!r}")
    current = _read_compute_modes(node_pool)
    if current == {mode}:
        logger.info("compute partition already %s; not calling amd-smi", mode)
        return
    amd_smi = '"${ROCM_PATH:-/opt/rocm}/bin/amd-smi"'
    command = (
        f'if [ "$(id -u)" -eq 0 ]; then {amd_smi} set -C {mode.upper()}; '
        f"else sudo -n {amd_smi} set -C {mode.upper()}; fi"
    )
    result = _node_run(node_pool, command, timeout=_SWITCH_TIMEOUT_SECS)
    amd_smi_text = f"{result.stdout or ''}\n{result.stderr or ''}".strip()
    if not result.ok and "sudo:" in amd_smi_text:
        pytest.fail(
            f"could not set compute partition to {mode.upper()}: amd-smi set needs root or "
            f"passwordless sudo on the assigned node.\n{amd_smi_text[-1500:]}"
        )
    # amd-smi exits non-zero when it cannot repartition a non-primary partition,
    # for example DPX to CPX reports NOT_SUPPORTED on each card's second half
    # although the card itself switched. sysfs is the verdict, not the exit code.
    wait_secs = _SWITCH_TIMEOUT_SECS if result.ok else _SWITCH_AFTER_ERROR_SECS
    deadline = time.monotonic() + wait_secs
    last: set[str] = set()
    while True:
        last = _read_compute_modes(node_pool)
        if last == {mode}:
            logger.info("compute partition is %s (amd-smi exit %s)", mode, result.exit_code)
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_SECS)
    pytest.fail(
        f"compute partition did not become {mode.upper()} (amd-smi exit {result.exit_code}); "
        f"sysfs reported {sorted(last) or 'nothing'}.\n{amd_smi_text[-1500:]}"
    )


@pytest.fixture(scope="module", autouse=True)
def _rocdecode_restore_compute_partition(request: pytest.FixtureRequest, node_pool):
    """Remember the node's compute partition and put it back after this folder.

    Setup only records the mode. The class fixtures switch to SPX, then DPX,
    then CPX. Teardown runs after the last test in this directory, before pytest
    moves to any other folder, so a later test sees the original mode.
    ``--no-gpu`` does not touch the machine. A node with no sysfs partition
    file is left alone. Mixed modes are a failure: there is no single mode to
    restore.
    """
    config = request.config
    if config.getoption("--no-gpu", default=False):
        yield
        return
    original = _read_compute_modes(node_pool)
    if not original:
        logger.info("no compute-partition sysfs on the assigned node; not switching")
        yield
        return
    if len(original) != 1:
        pytest.fail(
            f"GPUs report mixed compute partitions {sorted(original)}. "
            "Set one mode on every GPU before running this suite."
        )
    saved = next(iter(original))
    config._rocdecode_saved_compute = saved  # type: ignore[attr-defined]
    logger.info("saving compute partition %s for restore after rocdecode_one_stream_per_gpu", saved)
    yield
    logger.info("restoring compute partition %s", saved)
    try:
        _apply_compute_mode(node_pool, saved)
    finally:
        _clear_partition_cache(config)


@pytest.fixture(scope="module")
def rocdecode_set_compute_partition(request: pytest.FixtureRequest, node_pool, _rocdecode_restore_compute_partition):
    """Switch the assigned node to one compute mode and invalidate the plugin cache."""

    def _set(mode: str) -> None:
        if getattr(request.config, "_rocdecode_saved_compute", None) is None:
            return
        logger.info("rocdecode_one_stream_per_gpu: switching compute partition to %s", mode)
        _apply_compute_mode(node_pool, mode)
        _clear_partition_cache(request.config)

    return _set
