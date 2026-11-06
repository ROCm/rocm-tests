# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
partition_plugin.py -- skip tests whose partition marker does not match the host.

Compute markers (``partition.spx``, ``dpx``, ``tpx``, ``qpx``, ``cpx``) are compared
with ``current_compute_partition``. Memory markers (``partition.nps1``, ``nps2``,
``nps4``) are compared with ``current_memory_partition``. A test may carry one of
each. It runs only when every GPU reports the marked mode. Other tests are unchanged.

The marker does not change the partition. Set it before pytest, for example
``sudo amd-smi set -C CPX`` or ``sudo amd-smi set -M NPS1``.

When the matching sysfs file is not readable, that half of the check does not skip.
"""

from __future__ import annotations

import logging
import pathlib

import pytest

logger = logging.getLogger(__name__)

_COMPUTE_MODES = ("spx", "dpx", "tpx", "qpx", "cpx")
_MEMORY_MODES = ("nps1", "nps2", "nps4")
_COMPUTE_FILE = "current_compute_partition"
_MEMORY_FILE = "current_memory_partition"


def _read_partition_file(filename: str) -> set[str]:
    """Return distinct values of one sysfs partition file, lower-cased."""
    modes: set[str] = set()
    for path in pathlib.Path("/sys/bus/pci/devices").glob(f"*/{filename}"):
        try:
            text = path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if text:
            modes.add(text)
    return modes


def _marked_modes(item: pytest.Item, allowed: tuple[str, ...]) -> list[str]:
    """Return partition marker values on ``item`` that are in ``allowed``."""
    found: list[str] = []
    for marker in item.iter_markers():
        name = marker.name
        if not name.startswith("partition."):
            continue
        mode = name.split(".", 1)[1]
        if mode in allowed:
            found.append(mode)
    return found


def _skip_unless(item: pytest.Item, kind: str, filename: str, wanted: list[str], set_flag: str) -> None:
    """Skip ``item`` when ``filename`` does not report one of ``wanted`` on every GPU."""
    modes = _read_partition_file(filename)
    if not modes:
        logger.info(
            "partition.%s on %s but no %s file; not skipping",
            wanted[0],
            item.nodeid,
            filename,
        )
        return
    if len(modes) > 1:
        pytest.skip(
            f"partition test skipped: GPUs report mixed {kind} partitions {sorted(modes)}. "
            "Set one mode on every GPU before running."
        )
    host = next(iter(modes))
    if host not in wanted:
        pytest.skip(
            f"partition.{wanted[0]} test skipped: host {kind} partition is {host.upper()}. "
            f"Set it with `sudo amd-smi set {set_flag} {wanted[0].upper()}` to run this test."
        )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a test when a partition marker does not match the host.

    Args:
        item: The collected test item about to be set up.
    """
    compute = _marked_modes(item, _COMPUTE_MODES)
    memory = _marked_modes(item, _MEMORY_MODES)
    if compute:
        _skip_unless(item, "compute", _COMPUTE_FILE, compute, "-C")
    if memory:
        _skip_unless(item, "memory", _MEMORY_FILE, memory, "-M")
