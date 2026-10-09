# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capture abort with profiling active and forced timeout resilience tests."""

from __future__ import annotations

import sys

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
    replay_and_sync,
    safe_abort_capture,
)

_SMALL_CFG = dict(vocab=1024, dim=256, heads=8, layers=4, seq_len=64)
NUM_STEPS = 8


class TestCaptureAbort:

    @pytest.mark.timeout(300)
    def test_invalid_op_aborts_capture_cleanly(self, device, profiler_dir, monkeypatch):
        """Deliberate abort during profiling must leave device healthy."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=device)
        with make_profiler(profiler_dir, wait=0, warmup=0, active=3) as prof:
            try:
                with torch.cuda.graph(g, stream=s):
                    raise RuntimeError("Deliberate abort during profiling")
            except Exception:
                safe_abort_capture(g, stream=s)
            prof.step()
        g2, _ = capture_graph(model, sample)
        replay_and_sync(g2, device, n=3)
        del g2, model
        flush_gpu()
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_subsequent_capture_succeeds_after_abort(self, device, profiler_dir, monkeypatch):
        """Subsequent capture must succeed after a failed one."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream(device=device)
        try:
            with torch.cuda.graph(g, stream=s):
                raise RuntimeError("Simulated failure")
        except Exception:
            safe_abort_capture(g, stream=s)
        g2, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=3) as prof:
            times = measure_step_times(g2, device, NUM_STEPS, prof)
        del g2, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_no_lingering_signals_after_abort(self, device, monkeypatch):
        """Device must be fully responsive after abort — no lingering GPU signals."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        for _ in range(3):
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(device=device)
            try:
                with torch.cuda.graph(g, stream=s):
                    raise RuntimeError("Repeated abort")
            except Exception:
                safe_abort_capture(g, stream=s)
            flush_gpu()
        g_good, _ = capture_graph(model, sample)
        times = measure_step_times(g_good, device, NUM_STEPS)
        del g_good, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(600)
    def test_repeated_abort_recovery_cycles(self, device, profiler_dir, monkeypatch):
        """Multiple abort+recovery cycles must not corrupt device state."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        for _ in range(5):
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream(device=device)
            try:
                with torch.cuda.graph(g, stream=s):
                    raise RuntimeError("Repeated abort")
            except Exception:
                safe_abort_capture(g, stream=s)
            flush_gpu()
        g_good, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=3) as prof:
            times = measure_step_times(g_good, device, NUM_STEPS, prof)
        del g_good, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)


class TestWatchdogTimeout:

    @pytest.mark.timeout(300)
    def test_system_remains_responsive_under_load(self, device, profiler_dir, monkeypatch):
        """Device must remain responsive under repeated fast replays."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_concurrent_streams_no_indefinite_hang(self, device, monkeypatch):
        """Concurrent-stream replay must not hang indefinitely."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        s1 = torch.cuda.Stream(device=device)
        s2 = torch.cuda.Stream(device=device)
        g1, _ = capture_graph(model, sample, stream=s1)
        g2, _ = capture_graph(model, sample, stream=s2)
        for _ in range(10):
            with torch.cuda.stream(s1):
                g1.replay()
            with torch.cuda.stream(s2):
                g2.replay()
            torch.cuda.synchronize(device)
        del g1, g2, model
        flush_gpu()

    @pytest.mark.timeout(300)
    def test_graceful_timeout_with_artificial_cpu_delay(self, device, profiler_dir, monkeypatch):
        """Small CPU delays between replays must not cause hang."""
        import time

        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = []
            for _ in range(NUM_STEPS):
                t0 = time.monotonic()
                graph.replay()
                torch.cuda.synchronize(device)
                time.sleep(0.01)
                times.append(time.monotonic() - t0)
                prof.step()
        del graph, model
        flush_gpu()
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_rapid_sync_cycles_no_queue_stuck(self, device, monkeypatch):
        """Rapid sync cycles must not stall the GPU command queue."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        for _ in range(50):
            graph.replay()
            torch.cuda.synchronize(device)
        del graph, model
        flush_gpu()

    @pytest.mark.timeout(300)
    def test_mixed_graph_eager_with_profiling(self, device, profiler_dir, monkeypatch):
        """Interleaving graph replay and eager inference under profiling must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for i in range(NUM_STEPS):
                if i % 2 == 0:
                    graph.replay()
                else:
                    with torch.no_grad():
                        model(sample)
                torch.cuda.synchronize(device)
                prof.step()
        del graph, model
        flush_gpu()
        assert_traces_valid(profiler_dir)
