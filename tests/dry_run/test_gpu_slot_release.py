# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_gpu_slot_release.py -- Unit tests for GPU slot release in _acquire_and_yield and
target_executor's single-GPU path.

Slots are acquired before the executors and container wrappers are built, so a
failure while building them must still release them.  Leaked slots are never
returned, and every later GPU test then waits out its acquisition timeout and
skips.  No GPU hardware required (hw.cpu_only, ci.pr).
"""

import pytest

from framework.plugins import remote_node_plugin

# ---------------------------------------------------------------------------
# Helpers: lightweight stand-ins for the slot/pool objects
# ---------------------------------------------------------------------------


class _FakeGpuInfo:
    """Minimal GpuInfo stand-in carrying just the ordinal."""

    def __init__(self, index: int):
        self.index = index


class _FakeSlot:
    """Minimal NodeSlot stand-in."""

    def __init__(self, index: int):
        self.gpu_info = _FakeGpuInfo(index)
        self.gpu_label = f"GPU-{index}"
        self.node_spec = _FakeNodeSpec()

    def make_executor(self, **_kwargs):
        return object()


class _FakeNodeSpec:
    """Minimal NodeSpec stand-in."""

    def __init__(self, label: str = "localhost"):
        self.label = label


class _FakeMulti:
    """Minimal MultiGpuSlots stand-in."""

    def __init__(self, indices: list[int]):
        self.slots = [_FakeSlot(i) for i in indices]
        self.gpu_indices = list(indices)
        self.node_spec = _FakeNodeSpec()
        self._log_path: str | None = None
        self._session_log_path: str | None = None

    def make_executor(self, **_kwargs):
        return object()


class _FakePool:
    """NodePool stand-in recording which slots were released."""

    def __init__(self, total: int):
        self._total = total
        self.released: list = []

    def pool_status(self):
        return self._total - len(self.released), self._total

    def release(self, slots):
        self.released.extend(slots)

    def release_multi(self, multi):
        self.release(multi.slots)


class _FakeConfig:
    """pytest.Config stand-in for the sequential (non-xdist) drain path."""

    class _Option:
        numprocesses = None

    option = _Option()

    def getoption(self, _name, default=None):
        return default


class _FakeFrameworkConfig:
    """FrameworkConfig stand-in supplying only the container pull timeout."""

    class _TheRock:
        build_timeout_secs = 60

    therock = _TheRock()


class _FakeRequest:
    """pytest.FixtureRequest stand-in for a test without a gpu_indices marker."""

    class _Node:
        def get_closest_marker(self, _name):
            return None

    config = _FakeConfig()
    node = _Node()


def _drive(monkeypatch, single_gpu: bool, container_wrappers_raise: bool):
    """Start one acquisition path and return (generator, pool, slots it acquired)."""
    monkeypatch.setattr(remote_node_plugin, "_setup_monitoring", lambda *a, **k: ([], [], []))
    monkeypatch.setattr(remote_node_plugin, "_teardown_monitoring", lambda *a, **k: None)
    monkeypatch.setattr(remote_node_plugin, "_drain_gpu_slots", lambda *a, **k: None)
    monkeypatch.setattr(remote_node_plugin, "_write_session_separator", lambda *a, **k: None)

    def _boom(*_a, **_k):
        raise RuntimeError("Failed to pull container image: docker: not found")

    monkeypatch.setattr(
        remote_node_plugin,
        "_start_container_wrappers",
        _boom if container_wrappers_raise else (lambda *a, **k: []),
    )

    ctx = {
        "test_name": "test_stub",
        "log_path": "/tmp/stub.log",
        "session_log": None,
        "rock_dir": None,
        "gpu_count_marker": None,
        "is_multi_node": False,
        "is_multi_gpu": False,
    }
    container_opts = {"image": "stub:latest"}

    if single_gpu:
        # The default hw.gpu path sets up inline in target_executor, not via _acquire_and_yield.
        slot = _FakeSlot(0)
        monkeypatch.setattr(remote_node_plugin, "_resolve_test_context", lambda *a, **k: ctx)
        monkeypatch.setattr(remote_node_plugin, "_container_marker_opts", lambda *a, **k: container_opts)
        monkeypatch.setattr(remote_node_plugin, "_acquire_single", lambda *a, **k: slot)
        pool = _FakePool(total=1)
        gen = remote_node_plugin.target_executor.__wrapped__(_FakeRequest(), _FakeFrameworkConfig(), pool)
        return gen, pool, [slot]

    multi = _FakeMulti([0, 1])
    pool = _FakePool(total=2)
    gen = remote_node_plugin._acquire_and_yield(
        multi_list=[multi],
        ctx=ctx,
        framework_config=_FakeFrameworkConfig(),
        config=_FakeConfig(),
        node_pool=pool,
        is_multi_node=False,
        container_opts=container_opts,
    )
    return gen, pool, multi.slots


# ---------------------------------------------------------------------------
# A setup failure must not strand the slots
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.parametrize("single_gpu", [False, True], ids=["multi_gpu", "single_gpu"])
def test_setup_failure_releases_slots(monkeypatch, single_gpu):
    """Slots are released when container startup raises before the test body runs."""
    gen, pool, held = _drive(monkeypatch, single_gpu, container_wrappers_raise=True)

    with pytest.raises(RuntimeError, match="docker: not found"):
        next(gen)

    assert pool.released == held, "acquired slots were not returned to the pool"


# ---------------------------------------------------------------------------
# The normal path still releases
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.parametrize("single_gpu", [False, True], ids=["multi_gpu", "single_gpu"])
def test_normal_path_releases_slots(monkeypatch, single_gpu):
    """Slots are released once the test body completes and the generator closes."""
    gen, pool, held = _drive(monkeypatch, single_gpu, container_wrappers_raise=False)

    next(gen)
    assert pool.released == [], "slots released while the test body was still running"

    gen.close()
    assert pool.released == held, "acquired slots were not returned to the pool"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.parametrize("single_gpu", [False, True], ids=["multi_gpu", "single_gpu"])
def test_teardown_failure_releases_slots(monkeypatch, single_gpu):
    """Slots are released even when an earlier teardown step raises."""
    gen, pool, held = _drive(monkeypatch, single_gpu, container_wrappers_raise=False)

    def _boom(*_a, **_k):
        raise RuntimeError("teardown step failed")

    monkeypatch.setattr(remote_node_plugin, "_drain_gpu_slots", _boom)

    next(gen)
    with pytest.raises(RuntimeError, match="teardown step failed"):
        gen.close()

    assert pool.released == held, "acquired slots were not returned to the pool"
