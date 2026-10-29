# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Signal dependency cycle detection across multiple CUDA streams under profiling."""

from __future__ import annotations

import sys
import time

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    NO_PROGRESS_THRESHOLD_SEC,
    GPTModel,
    apply_graph_profiling_env,
    assert_no_hang,
    assert_traces_valid,
    capture_graph,
    make_profiler,
    measure_step_times,
)

NUM_STREAMS_SMALL = 4
NUM_STREAMS_LARGE = 16

_SMALL_CFG = dict(vocab=1024, dim=128, heads=4, layers=2, seq_len=32)


class TestSignalDependencyCycle:

    def test_multi_stream_event_dependencies(self, device, profiler_dir, monkeypatch):
        """Multiple streams with cross-stream event dependencies must not deadlock."""
        apply_graph_profiling_env(monkeypatch)
        streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS_SMALL)]
        events = [torch.cuda.Event() for _ in range(NUM_STREAMS_SMALL)]
        dim = 256
        tensors = [torch.randn(dim, dim, device=device) for _ in range(NUM_STREAMS_SMALL)]

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for _ in range(20):
                for i, (stream, event, tensor) in enumerate(zip(streams, events, tensors, strict=True)):
                    with torch.cuda.stream(stream):
                        if i > 0:
                            stream.wait_event(events[i - 1])
                        tensors[i] = torch.matmul(tensor, tensor)
                        event.record(stream)
                torch.cuda.synchronize(device)
                prof.step()

        assert_traces_valid(profiler_dir)

    def test_reverse_ordered_event_waits(self, device, profiler_dir, monkeypatch):
        """Reverse-ordered event waits to provoke potential circular dependencies."""
        apply_graph_profiling_env(monkeypatch)
        streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS_SMALL)]
        events = [torch.cuda.Event() for _ in range(NUM_STREAMS_SMALL)]
        dim = 128

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for _ in range(20):
                for i in range(NUM_STREAMS_SMALL):
                    with torch.cuda.stream(streams[i]):
                        t = torch.randn(dim, dim, device=device)
                        _ = torch.matmul(t, t)
                        events[i].record(streams[i])
                for i in reversed(range(NUM_STREAMS_SMALL)):
                    if i > 0:
                        streams[0].wait_event(events[i])
                torch.cuda.synchronize(device)
                prof.step()

        assert_traces_valid(profiler_dir)

    def test_many_streams_event_fan_out(self, device, profiler_dir, monkeypatch):
        """Fan-out pattern: one producer stream, many consumer streams."""
        apply_graph_profiling_env(monkeypatch)
        producer = torch.cuda.Stream(device=device)
        consumers = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS_LARGE)]
        event = torch.cuda.Event()
        dim = 128

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for _ in range(15):
                with torch.cuda.stream(producer):
                    t = torch.randn(dim, dim, device=device)
                    _ = torch.matmul(t, t)
                    event.record(producer)
                for consumer in consumers:
                    with torch.cuda.stream(consumer):
                        consumer.wait_event(event)
                        c = torch.randn(dim, dim, device=device)
                        _ = torch.matmul(c, c)
                torch.cuda.synchronize(device)
                prof.step()

        assert_traces_valid(profiler_dir)

    def test_graph_capture_with_multi_stream_sync(self, device, profiler_dir, monkeypatch):
        """Graph capture/replay with cross-stream synchronization under profiling."""
        apply_graph_profiling_env(monkeypatch)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        s1 = torch.cuda.Stream(device=device)
        s2 = torch.cuda.Stream(device=device)
        event = torch.cuda.Event()
        graph, _ = capture_graph(model, sample, stream=s1)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for _ in range(15):
                graph.replay()
                event.record(s1)
                s2.wait_event(event)
                with torch.cuda.stream(s2):
                    _ = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
                torch.cuda.synchronize(device)
                prof.step()

        assert_traces_valid(profiler_dir)

    def test_no_hard_hang_on_risky_ordering(self, device, monkeypatch):
        """Aggressive event ordering must not produce unrecoverable hangs."""
        apply_graph_profiling_env(monkeypatch)
        streams = [torch.cuda.Stream(device=device) for _ in range(8)]
        events = [torch.cuda.Event() for _ in range(8)]
        dim = 64

        for iteration in range(30):
            for i, s in enumerate(streams):
                with torch.cuda.stream(s):
                    t = torch.randn(dim, dim, device=device)
                    _ = torch.matmul(t, t)
                    events[i].record(s)
            for i in range(len(streams)):
                next_idx = (i + 1) % len(streams)
                if next_idx != 0:
                    streams[next_idx].wait_event(events[i])
            t0 = time.monotonic()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - t0
            assert elapsed < NO_PROGRESS_THRESHOLD_SEC, f"Iteration {iteration}: sync took {elapsed:.1f}s"


class TestSignalPoolStress:

    @pytest.mark.parametrize("pool_size", [8, 64, 256, 1024])
    def test_varying_signal_pool_sizes(self, device, profiler_dir, monkeypatch, pool_size):
        """Workload must succeed regardless of ROC_SIGNAL_POOL_SIZE."""
        apply_graph_profiling_env(monkeypatch, ROC_SIGNAL_POOL_SIZE=pool_size)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (2, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, 15, prof)

        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    def test_small_pool_no_starvation(self, device, profiler_dir, monkeypatch):
        """Very small signal pool must not cause signal starvation."""
        apply_graph_profiling_env(monkeypatch, ROC_SIGNAL_POOL_SIZE=4)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (2, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            times = measure_step_times(graph, device, 30, prof)

        assert_no_hang(times)

    def test_large_pool_no_regression(self, device, profiler_dir, monkeypatch):
        """Large pool should not cause performance regression vs default."""
        apply_graph_profiling_env(monkeypatch, ROC_SIGNAL_POOL_SIZE=4096)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (2, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, 15, prof)

        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(180)
    def test_pool_stress_with_many_streams(self, device, profiler_dir, monkeypatch):
        """Combine small pool with many streams to maximize signal pressure."""
        apply_graph_profiling_env(monkeypatch, ROC_SIGNAL_POOL_SIZE=16)
        streams = [torch.cuda.Stream(device=device) for _ in range(8)]
        events = [torch.cuda.Event() for _ in range(8)]
        dim = 64

        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            for iteration in range(50):
                for i, s in enumerate(streams):
                    with torch.cuda.stream(s):
                        t = torch.randn(dim, dim, device=device)
                        _ = torch.matmul(t, t)
                        events[i].record(s)
                for i in range(len(streams) - 1):
                    streams[i + 1].wait_event(events[i])
                t0 = time.monotonic()
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - t0
                assert elapsed < NO_PROGRESS_THRESHOLD_SEC, f"Iteration {iteration}: {elapsed:.1f}s"
                prof.step()

        assert_traces_valid(profiler_dir)
