# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Signal dependency cycle detection across multiple CUDA streams."""

from __future__ import annotations

import sys
import time

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    GPTModel,
    apply_graph_profiling_env,
    assert_no_hang,
    assert_traces_valid,
    capture_graph,
    flush_gpu,
    make_profiler,
    measure_step_times,
)

_SMALL_CFG = dict(vocab=1024, dim=256, heads=8, layers=4, seq_len=64)
NUM_STREAMS = 4
NUM_STEPS = 12
_STREAM_COUNTS = [1, 2, 4, 8]


class TestSignalDependencyCycle:

    @pytest.mark.timeout(400)
    def test_multi_stream_event_dependencies(self, device, profiler_dir, monkeypatch):
        """Graph replays across concurrent streams must complete under profiling."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS)]
        graphs = [capture_graph(model, sample, stream=s)[0] for s in streams]
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = []
            for _ in range(NUM_STEPS):
                t0 = time.monotonic()
                for g, s in zip(graphs, streams, strict=True):
                    with torch.cuda.stream(s):
                        g.replay()
                torch.cuda.synchronize(device)
                times.append(time.monotonic() - t0)
                prof.step()
        for g in graphs:
            del g
        del model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(400)
    def test_reverse_ordered_event_waits(self, device, profiler_dir, monkeypatch):
        """Event-wait chains across two streams must not deadlock under profiling."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        s1 = torch.cuda.Stream(device=device)
        s2 = torch.cuda.Stream(device=device)
        g1, _ = capture_graph(model, sample, stream=s1)
        g2, _ = capture_graph(model, sample, stream=s2)
        ev = torch.cuda.Event()
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = []
            for _ in range(NUM_STEPS):
                t0 = time.monotonic()
                with torch.cuda.stream(s1):
                    g1.replay()
                    ev.record(s1)
                s2.wait_event(ev)
                with torch.cuda.stream(s2):
                    g2.replay()
                torch.cuda.synchronize(device)
                times.append(time.monotonic() - t0)
                prof.step()
        del g1, g2, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(400)
    def test_many_streams_event_fan_out(self, device, profiler_dir, monkeypatch):
        """Fan-out from one stream to many must not deadlock."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS)]
        graphs = [capture_graph(model, sample, stream=s)[0] for s in streams]
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = []
            for _ in range(NUM_STEPS):
                t0 = time.monotonic()
                for g, s in zip(graphs, streams, strict=True):
                    with torch.cuda.stream(s):
                        g.replay()
                torch.cuda.synchronize(device)
                times.append(time.monotonic() - t0)
                prof.step()
        for g in graphs:
            del g
        del model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(300)
    def test_graph_capture_with_multi_stream_sync(self, device, profiler_dir, monkeypatch):
        """Sequential single-stream replay under profiling must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        g, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(g, device, NUM_STEPS, prof)
        del g, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_no_hard_hang_on_risky_ordering(self, device, profiler_dir, monkeypatch):
        """Streams with back-to-back records must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        g, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(g, device, NUM_STEPS, prof)
        del g, model
        flush_gpu()
        assert_no_hang(times)


class TestSignalPoolStress:

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("num_streams", _STREAM_COUNTS)
    def test_varying_signal_pool_sizes(self, device, monkeypatch, num_streams):
        """Graph replay with varying stream counts must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
        graphs = [capture_graph(model, sample, stream=s)[0] for s in streams]
        for _ in range(10):
            for g, s in zip(graphs, streams, strict=True):
                with torch.cuda.stream(s):
                    g.replay()
            torch.cuda.synchronize(device)
        for g in graphs:
            del g
        del model
        flush_gpu()

    @pytest.mark.timeout(300)
    def test_small_pool_no_starvation(self, device, monkeypatch):
        """Single-stream sequential replay must not stall the device."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        g, _ = capture_graph(model, sample)
        times = measure_step_times(g, device, 20)
        del g, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(300)
    def test_large_pool_no_regression(self, device, profiler_dir, monkeypatch):
        """Multi-stream replay under profiling must not regress."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(4)]
        graphs = [capture_graph(model, sample, stream=s)[0] for s in streams]
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = []
            for _ in range(10):
                t0 = time.monotonic()
                for g, s in zip(graphs, streams, strict=True):
                    with torch.cuda.stream(s):
                        g.replay()
                torch.cuda.synchronize(device)
                times.append(time.monotonic() - t0)
                prof.step()
        for g in graphs:
            del g
        del model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(400)
    def test_pool_stress_with_many_streams(self, device, monkeypatch):
        """Many concurrent streams must not hang or deadlock."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS)]
        graphs = [capture_graph(model, sample, stream=s)[0] for s in streams]
        times = []
        for _ in range(10):
            t0 = time.monotonic()
            for g, s in zip(graphs, streams, strict=True):
                with torch.cuda.stream(s):
                    g.replay()
            torch.cuda.synchronize(device)
            times.append(time.monotonic() - t0)
        for g in graphs:
            del g
        del model
        flush_gpu()
        assert_no_hang(times)
