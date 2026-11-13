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
``target_executor`` is checked on the controller. The marker does not change the
partition. Set it before pytest, for example ``sudo amd-smi set -C CPX`` or
``sudo amd-smi set -M NPS1``.

The ``/sys`` probe runs only on Linux. On any other platform, including Windows,
the probe is not attempted and the test is not skipped. On Linux, a missing or
unreadable partition file skips the marked test: the partition state is
unavailable, so the marker cannot be enforced.
"""

from __future__ import annotations

from collections.abc import Callable
import enum
import logging
import pathlib

import pytest

logger = logging.getLogger(__name__)

_COMPUTE_MODES = ("spx", "dpx", "tpx", "qpx", "cpx")
_MEMORY_MODES = ("nps1", "nps2", "nps4")
_COMPUTE_FILE = "current_compute_partition"
_MEMORY_FILE = "current_memory_partition"
_PROBE_TIMEOUT_SECS = 30.0


class _SysfsNotApplicable(enum.Enum):
    """Sentinel: this node has no Linux sysfs partition files.

    An empty read there is not a failed probe, so the marked test is allowed to run.
    This is an enum rather than ``object()``. ``set[str] | object`` collapses to
    ``object``, and mypy then rejects ``len``, ``sorted``, and ``iter`` on the result.
    """

    VALUE = "not_applicable"


_SYSFS_NOT_APPLICABLE = _SysfsNotApplicable.VALUE


def _modes_from_text(text: str) -> set[str]:
    """Return distinct non-empty lines of partition text, lower-cased."""
    return {line.strip().lower() for line in text.splitlines() if line.strip()}


def _local_has_sysfs() -> bool:
    """True when this pytest process is Linux and ``/sys`` can be read."""
    from framework.os_adapter import os_adapter_factory  # pylint: disable=import-outside-toplevel

    return os_adapter_factory().get_platform_name() in ("linux", "wsl")


def _executor_has_sysfs(executor: object) -> bool | None:
    """True when ``executor``'s node is Linux.

    ``None`` means ``uname`` failed and the caller must not cache the result.
    ``False`` means another platform, such as Windows, so ``/sys`` is not read.
    """
    host = str(getattr(executor, "host", None) or type(executor).__name__)
    cache: dict[str, bool] = getattr(_executor_has_sysfs, "cache", {})
    if host in cache:
        return cache[host]
    result = executor.run("uname -s", timeout=_PROBE_TIMEOUT_SECS)  # type: ignore[attr-defined]
    if not result.ok:
        logger.info("partition probe could not run uname (exit %s); partition state unavailable", result.exit_code)
        return None
    cache[host] = result.stdout.strip().lower().startswith("linux")
    _executor_has_sysfs.cache = cache  # type: ignore[attr-defined]
    return cache[host]


def _read_local(filename: str) -> set[str] | _SysfsNotApplicable:
    """Return distinct values of one sysfs partition file on this machine."""
    if not _local_has_sysfs():
        logger.info("partition sysfs check does not apply on this platform; not skipping")
        return _SYSFS_NOT_APPLICABLE
    modes: set[str] = set()
    for path in pathlib.Path("/sys/bus/pci/devices").glob(f"*/{filename}"):
        try:
            modes.update(_modes_from_text(path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return modes


def _read_executor(executor: object, filename: str) -> set[str] | _SysfsNotApplicable | None:
    """Read ``filename`` through ``executor``.

    Returns ``None`` when the command failed and the result must not be cached.
    ``_SYSFS_NOT_APPLICABLE`` means the node is not Linux, so the caller does not
    skip. An empty set on Linux means the partition state is unavailable.
    """
    from framework.executors.dry_run_executor import DryRunExecutor  # pylint: disable=import-outside-toplevel

    if isinstance(executor, DryRunExecutor):
        return _SYSFS_NOT_APPLICABLE
    has_sysfs = _executor_has_sysfs(executor)
    if has_sysfs is None:
        return None
    if not has_sysfs:
        logger.info("partition sysfs check does not apply on the assigned node; not skipping")
        return _SYSFS_NOT_APPLICABLE
    command = f'for f in /sys/bus/pci/devices/*/{filename}; do [ -r "$f" ] && cat "$f"; done'
    result = executor.run(command, timeout=_PROBE_TIMEOUT_SECS)  # type: ignore[attr-defined]
    modes = _modes_from_text(result.stdout)
    if modes or result.ok or not result.stderr.strip():
        return modes
    logger.info("partition probe of %s failed (exit %s); partition state unavailable", filename, result.exit_code)
    return None


def _cached_modes(
    config: pytest.Config,
    key: tuple[str, str],
    read: Callable[[], set[str] | _SysfsNotApplicable | None],
) -> set[str] | _SysfsNotApplicable:
    """Return partition modes for ``key``, reading them once per session."""
    cache: dict[tuple[str, str], set[str] | _SysfsNotApplicable]
    stored = getattr(config, "_partition_mode_cache", None)
    if stored is None:
        cache = {}
        config._partition_mode_cache = cache  # type: ignore[attr-defined]
    else:
        cache = stored
    if key not in cache:
        modes = read()
        if modes is None:
            return set()
        cache[key] = modes
    found = cache[key]
    if found is _SYSFS_NOT_APPLICABLE:
        return found
    return set(found)


def _partition_modes(request: pytest.FixtureRequest, filename: str) -> set[str] | _SysfsNotApplicable:
    """Read ``filename`` on the assigned node, or on the controller."""
    if "target_executor" not in request.fixturenames:
        return _cached_modes(request.config, ("controller", filename), lambda: _read_local(filename))

    group = request.getfixturevalue("target_executor")
    modes: set[str] = set()
    saw_node = False
    for executor in group:
        host = getattr(executor, "host", None) or type(executor).__name__

        def _read_this(bound=executor) -> set[str] | _SysfsNotApplicable | None:
            return _read_executor(bound, filename)

        found = _cached_modes(request.config, (str(host), filename), _read_this)
        if found is _SYSFS_NOT_APPLICABLE:
            continue
        saw_node = True
        modes.update(found)
    if not saw_node:
        return _SYSFS_NOT_APPLICABLE
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
    modes: set[str] | _SysfsNotApplicable,
) -> None:
    """Skip ``item`` when ``modes`` is not exactly one of ``wanted``."""
    marked = ",".join(wanted)
    if modes is _SYSFS_NOT_APPLICABLE:
        logger.info("partition.%s on %s: sysfs check does not apply; not skipping", marked, item.nodeid)
        return
    if not modes:
        pytest.skip(
            f"partition.{marked} test skipped: {kind} partition state unavailable "
            f"({filename} missing or unreadable on the assigned node)."
        )
    if len(modes) > 1:
        pytest.skip(
            f"partition test skipped: GPUs report mixed {kind} partitions {sorted(modes)}. "
            "Set one mode on every GPU before running."
        )
    node_mode = next(iter(modes))
    if node_mode not in wanted:
        pytest.skip(
            f"partition.{marked} test skipped: node {kind} partition is {node_mode.upper()}. "
            f"Set it with `sudo amd-smi set {set_flag} {marked.upper()}` to run this test."
        )


@pytest.fixture(autouse=True)
def _partition_on_execution_node(request: pytest.FixtureRequest) -> None:
    """Skip a test whose partition marker does not match the assigned node.

    Tests without a ``partition.*`` marker return immediately and do not acquire
    a GPU. When the test already requests ``target_executor``, the sysfs files
    are read through that executor after the node is acquired.
    """
    item = request.node
    compute = _marked_modes(item, _COMPUTE_MODES)
    memory = _marked_modes(item, _MEMORY_MODES)
    if compute:
        _skip_unless(item, "compute", _COMPUTE_FILE, compute, "-C", _partition_modes(request, _COMPUTE_FILE))
    if memory:
        _skip_unless(item, "memory", _MEMORY_FILE, memory, "-M", _partition_modes(request, _MEMORY_FILE))
