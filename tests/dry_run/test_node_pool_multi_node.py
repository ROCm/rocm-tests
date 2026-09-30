# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_node_pool_multi_node.py -- Unit tests for slot release in NodePool.acquire_multi_node.

Multi-node acquisition takes one node at a time, so a later node timing out must
return the slots already taken on earlier nodes.  No GPU hardware or SSH required
(hw.cpu_only, ci.pr).
"""

import pytest

from framework.gpu.detector import GpuInfo
from framework.nodes.node_pool import NodePool
from framework.nodes.node_spec import NodeSpec


def _two_node_pool(monkeypatch, tmp_path, failing_label: str | None):
    """Return (pool, per-node groups, released list) with acquire_slots/release_multi stubbed."""
    specs = [NodeSpec(hostname="node-a", label="node-a"), NodeSpec(hostname="node-b", label="node-b")]
    pool = NodePool(
        node_specs=specs,
        prefilled_gpus={spec.label: [GpuInfo(0, "gfx942", 32768, 0)] for spec in specs},
        artifact_dir=str(tmp_path),
    )
    groups = {spec.label: object() for spec in specs}

    def _acquire_slots(*, node_label, **_kwargs):
        if node_label == failing_label:
            raise RuntimeError(f"NodePool: cannot acquire 1 GPU slots from {node_label} after 0.0s")
        return groups[node_label]

    released: list = []
    monkeypatch.setattr(pool, "acquire_slots", _acquire_slots)
    monkeypatch.setattr(pool, "release_multi", released.append)
    return pool, groups, released


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_partial_acquisition_releases_earlier_nodes(monkeypatch, tmp_path):
    """Slots taken on node-a are released when node-b cannot be acquired."""
    pool, groups, released = _two_node_pool(monkeypatch, tmp_path, failing_label="node-b")

    with pytest.raises(RuntimeError, match="cannot acquire"):
        pool.acquire_multi_node(gpu_count_per_node=1, wait_timeout_secs=0.0, test_id="test_stub")

    assert released == [groups["node-a"]], "slots taken on node-a were not returned to the pool"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_full_acquisition_keeps_slots(monkeypatch, tmp_path):
    """A successful acquisition returns every node's slots and releases none of them."""
    pool, groups, released = _two_node_pool(monkeypatch, tmp_path, failing_label=None)

    assert pool.acquire_multi_node(gpu_count_per_node=1, wait_timeout_secs=0.0, test_id="test_stub") == [
        groups["node-a"],
        groups["node-b"],
    ]
    assert released == []
