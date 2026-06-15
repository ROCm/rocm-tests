# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
dynamic_scheduler.py -- Resource-aware xdist test ordering and group assignment.

Runs during pytest_collection_modifyitems. Assigns xdist_group to multinode and
multi-GPU tests, then sorts by schedule policy (resource-most: multinode →
multi-GPU DESC → single-GPU; resource-least: reversed). No-op when --no-gpu.
VRAM headroom applied per --vram-headroom-gb flag.
"""

from __future__ import annotations

from enum import Enum
import logging

import pytest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy enum
# ---------------------------------------------------------------------------


class SchedulePolicy(str, Enum):
    """GPU resource scheduling policy for ``--schedule-policy``."""

    RESOURCE_MOST = "resource-most"
    """Highest GPU demand first: multinode → multi_gpu DESC → single_gpu.

    Multi-GPU workers block inside fixture acquisition while single_gpu tests
    fill remaining free slots via xdist worksteal.
    """

    RESOURCE_LEAST = "resource-least"
    """Lowest GPU demand first: single_gpu → multi_gpu ASC → multinode.

    Maximises time-to-first-result; heavy tests wait until lightweight tests finish.
    """


# ---------------------------------------------------------------------------
# Item classification helpers (pure functions — no pytest imports at call time)
# ---------------------------------------------------------------------------


def _is_multinode(item) -> bool:
    """Return True if *item* carries ``@pytest.mark.e2e.multinode``."""
    return any(m.name == "e2e.multinode" for m in item.iter_markers())


def _multi_gpu_count(item) -> int:
    """Return the gpu_count for a multi-GPU test, or 0 for single-GPU tests.

    Reads ``@pytest.mark.gpu_count(N)`` first; falls back to 2 when the test
    carries ``@pytest.mark.hw.multi_gpu`` without an explicit count.

    Returns:
        Integer N > 1 for multi-GPU tests; 0 for single-GPU tests.
    """
    gpu_count_marker = item.get_closest_marker("gpu_count")
    if gpu_count_marker and gpu_count_marker.args:
        n = int(gpu_count_marker.args[0])
        if n > 1:
            return n
    if any(m.name == "hw.multi_gpu" for m in item.iter_markers()):
        return 2  # default gpu_count for hw.multi_gpu without explicit marker
    return 0


# ---------------------------------------------------------------------------
# Sort key functions
# ---------------------------------------------------------------------------


def _gpu_indices_count(item) -> int:
    """Return the number of pinned GPU indices, or 0 if no gpu_indices marker."""
    m = item.get_closest_marker("gpu_indices")
    if m and m.args:
        raw = m.args[0]
        if isinstance(raw, int):
            return 1
        return len(list(raw))
    return 0


def resource_sort_key_most(item) -> tuple:
    """Sort key for ``resource-most``: lower tuple = runs first.

    Tier 0: multinode (cross-node, highest total demand).
    Tier 1: gpu_indices / multi_gpu sorted by GPU count DESC (higher = earlier).
    Tier 2: single_gpu (fills free slots via xdist worksteal).

    Args:
        item: pytest ``Item`` from ``pytest_collection_modifyitems``.

    Returns:
        Tuple ``(tier, sub_key)`` suitable for ``list.sort(key=...)``.
    """
    if _is_multinode(item):
        return (0, 0)
    k = _gpu_indices_count(item)
    if k > 0:
        return (1, -k)
    n = _multi_gpu_count(item)
    if n > 0:
        return (1, -n)  # negate so higher count → smaller sub_key → runs earlier
    return (2, 0)


def resource_sort_key_least(item) -> tuple:
    """Sort key for ``resource-least``: lower tuple = runs first.

    Tier 0: single_gpu (lowest demand, immediate results).
    Tier 1: gpu_indices / multi_gpu sorted by GPU count ASC (lower = earlier).
    Tier 2: multinode (highest demand; starts last).

    Args:
        item: pytest ``Item`` from ``pytest_collection_modifyitems``.

    Returns:
        Tuple ``(tier, sub_key)`` suitable for ``list.sort(key=...)``.
    """
    if _is_multinode(item):
        return (2, 0)
    k = _gpu_indices_count(item)
    if k > 0:
        return (1, k)
    n = _multi_gpu_count(item)
    if n > 0:
        return (1, n)  # lower count → smaller sub_key → runs earlier
    return (0, 0)


# ---------------------------------------------------------------------------
# DynamicScheduler
# ---------------------------------------------------------------------------


class DynamicScheduler:
    """Collection-time scheduler: xdist_group assignment + resource-policy sort.

    Step 1 (``_assign_xdist_groups``): annotate multi-GPU and multinode tests
    with ``xdist_group`` so each goes to a dedicated xdist worker that holds all
    required GPU file locks simultaneously.

    Step 2 (``schedule``): sort *items* in-place so the xdist work queue yields
    the desired slot-filling behaviour at runtime.

    Attributes:
        pool:   ``NodePool`` used for ``recommended_workers()``.
        policy: ``SchedulePolicy`` to apply when sorting items.
    """

    def __init__(self, pool, policy: SchedulePolicy = SchedulePolicy.RESOURCE_MOST) -> None:
        """Initialise the scheduler.

        Args:
            pool:   Active ``NodePool`` (used only for ``recommended_workers()``).
            policy: ``SchedulePolicy.RESOURCE_MOST`` or ``SchedulePolicy.RESOURCE_LEAST``.
        """
        self.pool = pool
        self.policy = policy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule(self, items: list) -> None:
        """Assign xdist_group markers and sort *items* in-place.

        Modifies *items* directly (pytest ``pytest_collection_modifyitems`` contract).

        Step 1: assign ``xdist_group`` markers for multi-GPU and multinode tests.
        Step 2: stable-sort by the policy sort key (``resource_sort_key_most`` or
                ``resource_sort_key_least``).

        Args:
            items: Pytest ``Item`` list from ``pytest_collection_modifyitems``.
        """
        self._assign_xdist_groups(items)

        key_fn = resource_sort_key_most if self.policy == SchedulePolicy.RESOURCE_MOST else resource_sort_key_least
        items.sort(key=key_fn)

        multi = sum(1 for i in items if _is_multinode(i) or _multi_gpu_count(i) > 0)
        single = len(items) - multi
        logger.info(
            "DynamicScheduler [%s]: %d items — %d multi-resource, %d single-gpu",
            self.policy.value,
            len(items),
            multi,
            single,
        )

    def recommended_workers(self) -> int:
        """Return the recommended ``-n`` value for optimal parallelism.

        Returns:
            Total GPU slot count across all nodes in the pool.
        """
        return self.pool.total_gpu_slots() if self.pool is not None else 1

    # ------------------------------------------------------------------
    # Internal: xdist_group assignment
    # ------------------------------------------------------------------

    def _assign_xdist_groups(self, items: list) -> None:
        """Annotate multi-GPU and multinode tests with ``xdist_group`` markers.

        Each test gets its own unique group name so separate xdist workers can run
        different multi-GPU tests in parallel.  Single-GPU tests receive no group
        and distribute via xdist worksteal.

        Group naming:
            - multinode tests  → ``"multinode_N"`` (N = per-test counter).
            - multi_gpu tests  → ``"multi_gpu_{count}_{N}"`` (N = per-test counter).
            - single_gpu tests → no group assigned.

        Args:
            items: Pytest ``Item`` list to annotate in-place.
        """
        gpu_indices_idx = 0
        multi_gpu_idx = 0
        multi_node_idx = 0

        for item in items:
            # gpu_indices path: must be checked before multinode/multi_gpu so that
            # tests using gpu_indices get their own deterministic group and are not
            # accidentally treated as plain multi-GPU tests.
            gpu_idx_marker = item.get_closest_marker("gpu_indices")
            if gpu_idx_marker and gpu_idx_marker.args:
                if item.get_closest_marker("gpu_count"):
                    pytest.fail(
                        f"{item.nodeid}: @pytest.mark.gpu_indices and @pytest.mark.gpu_count "
                        "are mutually exclusive — remove gpu_count when using gpu_indices.",
                        pytrace=False,
                    )
                if item.get_closest_marker("multi_gpu"):
                    pytest.fail(
                        f"{item.nodeid}: @pytest.mark.gpu_indices and @pytest.mark.hw.multi_gpu "
                        "are mutually exclusive — use gpu_indices alone to pin specific indices.",
                        pytrace=False,
                    )
                if isinstance(gpu_idx_marker.args[0], int):
                    pytest.fail(
                        f"{item.nodeid}: @pytest.mark.gpu_indices takes a list, not a bare int. "
                        f"Use @pytest.mark.gpu_indices([{gpu_idx_marker.args[0]}]) instead.",
                        pytrace=False,
                    )
                raw: list[int] = list(gpu_idx_marker.args[0])
                if not raw:
                    pytest.fail(
                        f"{item.nodeid}: @pytest.mark.gpu_indices requires at least one index — got empty list.",
                        pytrace=False,
                    )
                # Normalize order so [2,0] and [0,2] get the same group name.
                group_name = "gpu_indices_" + "_".join(str(i) for i in sorted(raw)) + f"_{gpu_indices_idx}"
                gpu_indices_idx += 1
                item.add_marker(pytest.mark.xdist_group(group_name))
                logger.debug("xdist_group assigned (gpu_indices): %s → %s", item.nodeid, group_name)
                continue

            if _is_multinode(item):
                group_name = f"multinode_{multi_node_idx}"
                multi_node_idx += 1
                item.add_marker(pytest.mark.xdist_group(group_name))
                logger.debug("xdist_group assigned: %s → %s", item.nodeid, group_name)

            else:
                n = _multi_gpu_count(item)
                if n > 0:
                    group_name = f"multi_gpu_{n}_{multi_gpu_idx}"
                    multi_gpu_idx += 1
                    item.add_marker(pytest.mark.xdist_group(group_name))
                    logger.debug("xdist_group assigned: %s → %s", item.nodeid, group_name)
                # single_gpu: no group — xdist worksteal
