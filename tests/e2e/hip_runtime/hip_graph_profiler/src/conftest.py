# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for the HIP graph profiler worker test scripts."""

from __future__ import annotations

import gc
import os
import shutil
import signal

import pytest

_HAS_SIGALRM = hasattr(signal, "SIGALRM")
_DEADLOCK_TIMEOUT_SEC = 120


class _DeadlockTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _DeadlockTimeoutError("Test exceeded timeout — potential deadlock detected")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "timeout(seconds): per-test deadlock watchdog timeout")
    config.addinivalue_line("filterwarnings", "ignore:.*Profiler clears events.*:UserWarning")


@pytest.fixture(autouse=True)
def deadlock_watchdog(request: pytest.FixtureRequest):
    if not _HAS_SIGALRM:
        yield
        return
    marker = request.node.get_closest_marker("timeout")
    seconds = marker.args[0] if marker else _DEADLOCK_TIMEOUT_SEC
    prev = signal.signal(signal.SIGALRM, _alarm_handler)  # type: ignore[attr-defined]
    signal.alarm(seconds)  # type: ignore[attr-defined]
    try:
        yield
    finally:
        signal.alarm(0)  # type: ignore[attr-defined]
        signal.signal(signal.SIGALRM, prev)  # type: ignore[attr-defined]


@pytest.fixture
def device():
    torch = pytest.importorskip("torch", reason="PyTorch not installed")
    if not torch.cuda.is_available():
        pytest.skip("No CUDA/HIP GPU available")
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    yield dev
    gc.collect()
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()


@pytest.fixture
def profiler_dir(tmp_path):
    d = tmp_path / "profiler_traces"
    d.mkdir()
    return d


@pytest.fixture
def clean_extension_cache():
    cache = os.path.expanduser("~/.cache/torch_extensions")
    if os.path.isdir(cache):
        shutil.rmtree(cache, ignore_errors=True)
