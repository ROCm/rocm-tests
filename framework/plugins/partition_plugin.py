# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
partition_plugin.py -- skip tests whose partition marker does not match the node.

Compute markers (``partition.spx``, ``dpx``, ``tpx``, ``qpx``, ``cpx``) are compared
with ``current_compute_partition``. Memory markers (``partition.nps1``, ``nps2``,
``nps4``) are compared with ``current_memory_partition``. A test may carry one of
each. It runs only when every GPU on the assigned node reports the marked mode.
Other tests are unchanged.

A test that uses ``target_executor`` is checked on the node that fixture acquired.
The sysfs read goes through that executor, so ``--remote-node`` reads the GPU host
rather than the controller running pytest. A partition-marked test with no
``target_executor`` is checked on the controller. The result is cached per node for
the session. The marker does not change the partition.

Set the mode before pytest, for example ``sudo amd-smi set -C CPX`` or
``sudo amd-smi set -M NPS1``.

When the matching sysfs file is not readable, that half of the check does not skip.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
import pathlib

import pytest

from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

_COMPUTE_MODES = ("spx", "dpx", "tpx", "qpx", "cpx")
_MEMORY_MODES = ("nps1", "nps2", "nps4")
_COMPUTE_FILE = "current_compute_partition"
_MEMORY_FILE = "current_memory_partition"
_PROBE_TIMEOUT_SECS = 30.0
_CACHE_ATTR = "_partition_mode_cache"


def _modes_from_text(text: str) -> set[str]:
    """Return distinct non-empty lines of partition text, lower-cased."""
    modes: set[str] = set()
    for line in text.splitlines():
        token = line.strip().lower()
        if token:
            modes.add(token)
    return modes


def _read_partition_file(filename: str) -> set[str]:
    """Return distinct values of one sysfs partition file on this machine."""
    modes: set[str] = set()
    for path in pathlib.Path("/sys/bus/pci/devices").glob(f"*/{filename}"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        modes.update(_modes_from_text(text))
    return modes


def _probe_command(filename: str) -> str:
    """Shell command that prints one partition file from every PCI device."""
    return f'for f in /sys/bus/pci/devices/*/{filename}; do [ -r "$f" ] && cat "$f"; done'


def _probe_executor(executor: AbstractExecutor, filename: str) -> tuple[set[str], bool]:
    """Read ``filename`` through ``executor``.

    Returns:
        A pair of the modes found and whether that result is safe to cache.
        ``DryRunExecutor`` reports no modes. A command that prints nothing and
        writes no stderr is treated as "no partition file" and cached. A command
        that fails with stderr is not cached, so a later test can probe again.
    """
    from framework.executors.dry_run_executor import DryRunExecutor  # pylint: disable=import-outside-toplevel

    if isinstance(executor, DryRunExecutor):
        return set(), True

    result = executor.run(_probe_command(filename), timeout=_PROBE_TIMEOUT_SECS)
    modes = _modes_from_text(result.stdout)
    if modes or result.ok or not result.stderr.strip():
        return modes, True
    logger.info(
        "partition probe of %s failed (exit %s); not skipping",
        filename,
        result.exit_code,
    )
    return set(), False


def _cache_key(executor: AbstractExecutor) -> str:
    """Identity of the machine ``executor`` runs on."""
    host = getattr(executor, "host", None)
    if isinstance(host, str) and host:
        return f"ssh:{host}"
    return f"local:{type(executor).__name__}"


def _mode_cache(config: pytest.Config) -> dict[tuple[str, str], set[str]]:
    """Session cache of partition modes, keyed by node and sysfs filename."""
    cache = getattr(config, _CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(config, _CACHE_ATTR, cache)
    return cache


def _modes_on_executor(config: pytest.Config, executor: AbstractExecutor, filename: str) -> set[str]:
    """Return cached partition modes for ``executor``'s node."""
    cache = _mode_cache(config)
    key = (_cache_key(executor), filename)
    if key not in cache:
        modes, cacheable = _probe_executor(executor, filename)
        if not cacheable:
            return modes
        cache[key] = modes
    return set(cache[key])


def _modes_on_group(config: pytest.Config, group: Iterable[AbstractExecutor], filename: str) -> set[str]:
    """Return the union of partition modes across every executor in ``group``."""
    modes: set[str] = set()
    for executor in group:
        modes.update(_modes_on_executor(config, executor, filename))
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


def _skip_unless(
    item: pytest.Item,
    kind: str,
    filename: str,
    wanted: list[str],
    set_flag: str,
    modes: set[str],
) -> None:
    """Skip ``item`` when ``modes`` is not exactly one of ``wanted``."""
    if not modes:
        logger.info(
            "partition.%s on %s but no %s value; not skipping",
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
    node_mode = next(iter(modes))
    if node_mode not in wanted:
        pytest.skip(
            f"partition.{wanted[0]} test skipped: node {kind} partition is {node_mode.upper()}. "
            f"Set it with `sudo amd-smi set {set_flag} {wanted[0].upper()}` to run this test."
        )


def _apply_partition_markers(item: pytest.Item, modes_for: Callable[[str], set[str]]) -> None:
    """Apply compute and memory marker checks using ``modes_for(filename)``."""
    compute = _marked_modes(item, _COMPUTE_MODES)
    memory = _marked_modes(item, _MEMORY_MODES)
    if compute:
        _skip_unless(item, "compute", _COMPUTE_FILE, compute, "-C", modes_for(_COMPUTE_FILE))
    if memory:
        _skip_unless(item, "memory", _MEMORY_FILE, memory, "-M", modes_for(_MEMORY_FILE))


@pytest.fixture(autouse=True)
def _partition_on_execution_node(request: pytest.FixtureRequest) -> None:
    """Skip a test whose partition marker does not match the assigned node.

    Tests without a ``partition.*`` marker return immediately and do not acquire
    a GPU. When the test already requests ``target_executor``, the sysfs files
    are read through that executor after the node is acquired. Otherwise the
    controller's own sysfs is read.
    """
    item = request.node
    if not _marked_modes(item, _COMPUTE_MODES) and not _marked_modes(item, _MEMORY_MODES):
        return

    if "target_executor" in request.fixturenames:
        group = request.getfixturevalue("target_executor")
        _apply_partition_markers(
            item,
            lambda filename: _modes_on_group(request.config, group, filename),
        )
        return

    cache = _mode_cache(request.config)

    def _local_modes(filename: str) -> set[str]:
        key = ("controller", filename)
        if key not in cache:
            cache[key] = _read_partition_file(filename)
        return set(cache[key])

    _apply_partition_markers(item, _local_modes)
