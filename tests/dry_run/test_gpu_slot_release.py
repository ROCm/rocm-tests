# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_gpu_slot_release.py -- Unit tests for GPU slot release in _acquire_and_yield.

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
    """NodePool stand-in recording which groups were released."""

    def __init__(self, total: int):
        self._total = total
        self.released: list = []

    def pool_status(self):
        return self._total - len(self.released), self._total

    def release_multi(self, multi):
        self.released.append(multi)


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


def _drive(monkeypatch, container_wrappers_raise: bool):
    """Run _acquire_and_yield once and return (pool, multi, raised_exception)."""
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

    multi = _FakeMulti([0, 1])
    pool = _FakePool(total=2)
    gen = remote_node_plugin._acquire_and_yield(
        multi_list=[multi],
        ctx={
            "test_name": "test_stub",
            "log_path": "/tmp/stub.log",
            "session_log": None,
            "rock_dir": None,
        },
        framework_config=_FakeFrameworkConfig(),
        config=_FakeConfig(),
        node_pool=pool,
        is_multi_node=False,
        container_opts={"image": "stub:latest"},
    )
    return gen, pool, multi


# ---------------------------------------------------------------------------
# A setup failure must not strand the slots
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_setup_failure_releases_slots(monkeypatch):
    """Slots are released when container startup raises before the test body runs."""
    gen, pool, multi = _drive(monkeypatch, container_wrappers_raise=True)

    with pytest.raises(RuntimeError, match="docker: not found"):
        next(gen)

    assert pool.released == [multi], "acquired slots were not returned to the pool"


# ---------------------------------------------------------------------------
# The normal path still releases
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_normal_path_releases_slots(monkeypatch):
    """Slots are released once the test body completes and the generator closes."""
    gen, pool, multi = _drive(monkeypatch, container_wrappers_raise=False)

    next(gen)
    assert pool.released == [], "slots released while the test body was still running"

    gen.close()
    assert pool.released == [multi], "acquired slots were not returned to the pool"
