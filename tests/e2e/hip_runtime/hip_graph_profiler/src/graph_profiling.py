# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Graph replay and profiler stability tests."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    GPT_1B_CONFIG,
    GPTModel,
    apply_graph_profiling_env,
    assert_no_hang,
    assert_traces_valid,
    capture_graph,
    flush_gpu,
    make_profiler,
    measure_step_times,
)

NUM_STEPS = 15


class TestGraphReplayUnderProfiling:

    @pytest.mark.timeout(300)
    def test_graph_replay_completes_with_profiling(self, device, profiler_dir, monkeypatch, clean_extension_cache):
        """All 15 steps must complete with profiling active; traces must be generated."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=1, active=5, repeat=2) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_profiler_artifacts_are_parsable(self, device, profiler_dir, monkeypatch):
        """Profiler output must be valid parsable JSON traces."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (1, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            measure_step_times(graph, device, 8, prof)
        del graph, model
        flush_gpu()
        traces = assert_traces_valid(profiler_dir)
        assert len(traces) >= 1

    @pytest.mark.timeout(300)
    def test_no_watchdog_trigger_during_replay(self, device, profiler_dir, monkeypatch):
        """No individual step should exceed the no-progress threshold."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(300)
    def test_iteration_time_comparable_across_steps(self, device, profiler_dir, monkeypatch):
        """No single step should regress beyond 10x the median step time."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        for _ in range(5):
            graph.replay()
        torch.cuda.synchronize(device)
        with make_profiler(profiler_dir) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        median = sorted(times)[len(times) // 2]
        threshold = max(median * 10, 1.0)
        for i, t in enumerate(times[2:], start=2):
            assert t < threshold, f"Step {i}: {t:.4f}s exceeds {threshold:.4f}s (median {median:.4f}s)"


class TestProfilerGraphStability:

    @pytest.mark.timeout(300)
    def test_graph_profiling_bfloat16_1b(self, device, profiler_dir, monkeypatch, clean_extension_cache):
        """~1B-param bfloat16 model, batch=2, 15 steps with profiling+graphs."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=2, active=5, repeat=2) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_profiling_without_graph_baseline(self, device, profiler_dir, monkeypatch):
        """Profiling with USE_GRAPH=0 must always succeed."""
        apply_graph_profiling_env(use_graph=0)
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for _ in range(NUM_STEPS):
                with torch.no_grad():
                    model(sample)
                torch.cuda.synchronize(device)
                prof.step()
        del model
        flush_gpu()
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_graph_without_profiling_baseline(self, device, monkeypatch):
        """Graph replay with PROFILING=0 must always succeed."""
        apply_graph_profiling_env(profiling=0)
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        times = measure_step_times(graph, device, NUM_STEPS)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
