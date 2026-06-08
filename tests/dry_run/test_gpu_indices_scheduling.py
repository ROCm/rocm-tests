# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_gpu_indices_scheduling.py -- Unit tests for DynamicScheduler gpu_indices handling.

Verifies xdist_group assignment, sort-order normalization, and the gpu_count
conflict guard.  No GPU hardware required (hw.cpu_only, ci.pr).
"""

import pytest

from framework.scheduling.dynamic_scheduler import DynamicScheduler, SchedulePolicy

# ---------------------------------------------------------------------------
# Helpers: lightweight mock item
# ---------------------------------------------------------------------------


class _MockItem:
    """Minimal pytest Item stand-in for scheduler unit tests."""

    def __init__(self, nodeid: str, markers: list):
        self.nodeid = nodeid
        self._markers: dict = {}
        self._added: list = []
        for m in markers:
            self._markers[m.name] = m

    def get_closest_marker(self, name: str):
        return self._markers.get(name)

    def iter_markers(self):
        return list(self._markers.values())

    def add_marker(self, marker):
        self._added.append(marker)


class _FakeMarker:
    def __init__(self, name: str, *args):
        self.name = name
        self.args = args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_gpu_indices_group_assigned():
    """DynamicScheduler assigns a gpu_indices_* xdist_group for gpu_indices marker."""
    item = _MockItem("test_foo", [_FakeMarker("gpu_indices", [0, 2])])
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    scheduler._assign_xdist_groups([item])

    assert len(item._added) == 1
    group_marker = item._added[0]
    # The marker is pytest.mark.xdist_group("gpu_indices_0_2_0")
    assert "gpu_indices_0_2" in str(group_marker)


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_gpu_indices_sort_normalization():
    """[2, 0] and [0, 2] produce the same group name prefix (sorted indices)."""
    item_a = _MockItem("test_a", [_FakeMarker("gpu_indices", [2, 0])])
    item_b = _MockItem("test_b", [_FakeMarker("gpu_indices", [0, 2])])
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    scheduler._assign_xdist_groups([item_a, item_b])

    assert len(item_a._added) == 1
    assert len(item_b._added) == 1
    name_a = str(item_a._added[0])
    name_b = str(item_b._added[0])
    # Both contain "gpu_indices_0_2" — they differ only in the trailing counter
    assert "gpu_indices_0_2" in name_a
    assert "gpu_indices_0_2" in name_b


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_single_gpu_indices_gets_own_group():
    """A single-element gpu_indices([3]) gets a gpu_indices_3_* group, not a multi_gpu group."""
    item = _MockItem("test_single_idx", [_FakeMarker("gpu_indices", [3])])
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    scheduler._assign_xdist_groups([item])

    assert len(item._added) == 1
    assert "gpu_indices_3" in str(item._added[0])


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_gpu_indices_conflict_with_gpu_count_fails():
    """gpu_indices + gpu_count on the same test raises pytest.fail at collection."""
    item = _MockItem(
        "test_conflict",
        [
            _FakeMarker("gpu_indices", [0, 1]),
            _FakeMarker("gpu_count", 2),
        ],
    )
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    with pytest.raises(pytest.fail.Exception, match="mutually exclusive"):
        scheduler._assign_xdist_groups([item])


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_no_group_for_plain_single_gpu():
    """Plain single-GPU tests (no gpu_indices, no multi_gpu) receive no xdist_group."""
    item = _MockItem("test_plain", [_FakeMarker("hw.gpu")])
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    scheduler._assign_xdist_groups([item])
    assert item._added == []


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_gpu_indices_bare_int_fails():
    """gpu_indices(0) — bare int instead of a list — raises a friendly pytest.fail at collection."""
    item = _MockItem("test_bare_int", [_FakeMarker("gpu_indices", 0)])  # int, not list
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    with pytest.raises(pytest.fail.Exception, match="takes a list"):
        scheduler._assign_xdist_groups([item])


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_gpu_indices_empty_list_fails():
    """gpu_indices([]) — empty list — raises pytest.fail at collection."""
    item = _MockItem("test_empty_list", [_FakeMarker("gpu_indices", [])])
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    with pytest.raises(pytest.fail.Exception, match="empty"):
        scheduler._assign_xdist_groups([item])


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_gpu_indices_with_multi_gpu_fails():
    """gpu_indices + hw.multi_gpu (without gpu_count) raises pytest.fail at collection."""
    item = _MockItem(
        "test_multi_gpu_conflict",
        [
            _FakeMarker("gpu_indices", [0, 1]),
            _FakeMarker("multi_gpu"),
        ],
    )
    scheduler = DynamicScheduler(pool=None, policy=SchedulePolicy.RESOURCE_MOST)
    with pytest.raises(pytest.fail.Exception, match="mutually exclusive"):
        scheduler._assign_xdist_groups([item])
