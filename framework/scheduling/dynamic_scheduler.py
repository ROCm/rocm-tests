# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
dynamic_scheduler.py -- Resource-aware test scheduling for GPU test suites.

Replaces ``TestScheduler`` (``framework/nodes/test_scheduler.py``) with a unified
engine that:

1. Assigns ``xdist_group`` markers so multi-GPU and multinode tests are routed to a
   dedicated xdist worker that holds all required GPU file locks simultaneously.

2. Sorts the collected test items by a resource-priority key so that the xdist work
   queue produces the desired slot-filling behaviour at runtime.

Scheduling policies
-------------------
``resource-most`` (default):
    Tests are ordered by descending GPU demand:
    multinode (tier 0) → multi_gpu by count DESC (tier 1) → single_gpu (tier 2).

    The workers that grab multi_gpu or multinode items may *block* inside their fixture
    while waiting for enough GPU slots.  Other workers continue stealing single_gpu items
    from the queue and run on whatever GPUs are free.  This means single_gpu tests fill
    available slots *emergently* — no special interleaving is required in the static sort.

``resource-least``:
    single_gpu (tier 0) → multi_gpu by count ASC (tier 1) → multinode (tier 2).
    Maximises time-to-first-result; heavy tests wait until lightweight ones clear.

xdist_group assignment
----------------------
- ``@pytest.mark.e2e.multinode`` → ``xdist_group = "multinode_N"`` (unique per test).
- ``@pytest.mark.gpu_count(N>1)`` or ``hw.multi_gpu`` → ``xdist_group = "multi_gpu_{count}_{N}"``.
- All other tests → no group (xdist worksteal distributes across free workers).

Each multi-GPU / multinode test gets its own unique group name, so separate xdist workers
can run different multi-GPU tests in parallel (each worker holds its own set of GPU locks).

Usage (via ``scheduling_plugin.pytest_collection_modifyitems`` — never called directly)::

    pool = NodePool(...)
    scheduler = DynamicScheduler(pool=pool, policy=SchedulePolicy.RESOURCE_MOST)
    scheduler.schedule(items)       # modifies items in-place
    print(scheduler.recommended_workers())
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


def resource_sort_key_most(item) -> tuple:
    """Sort key for ``resource-most``: lower tuple = runs first.

    Tier 0: multinode (cross-node, highest total demand).
    Tier 1: multi_gpu sorted by gpu_count DESC (higher count = earlier).
    Tier 2: single_gpu (fills free slots via xdist worksteal).

    Args:
        item: pytest ``Item`` from ``pytest_collection_modifyitems``.

    Returns:
        Tuple ``(tier, sub_key)`` suitable for ``list.sort(key=...)``.
    """
    if _is_multinode(item):
        return (0, 0)
    n = _multi_gpu_count(item)
    if n > 0:
        return (1, -n)  # negate so higher count → smaller sub_key → runs earlier
    return (2, 0)


def resource_sort_key_least(item) -> tuple:
    """Sort key for ``resource-least``: lower tuple = runs first.

    Tier 0: single_gpu (lowest demand, immediate results).
    Tier 1: multi_gpu sorted by gpu_count ASC (lower count = earlier).
    Tier 2: multinode (highest demand; starts last).

    Args:
        item: pytest ``Item`` from ``pytest_collection_modifyitems``.

    Returns:
        Tuple ``(tier, sub_key)`` suitable for ``list.sort(key=...)``.
    """
    if _is_multinode(item):
        return (2, 0)
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
        multi_gpu_idx = 0
        multi_node_idx = 0

        for item in items:
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
