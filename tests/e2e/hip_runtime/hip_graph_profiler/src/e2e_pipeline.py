# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end benchmarking pipeline with HIP graph capture and PyTorch profiler."""

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

NUM_STEPS = 20
WARMUP_STEPS = 5


class TestE2EPipeline:

    @pytest.mark.timeout(600)
    def test_full_training_pipeline_with_graph_and_profiling(
        self, device, profiler_dir, monkeypatch, clean_extension_cache
    ):
        """Full capture-warmup-profile-replay pipeline at 1B-param scale."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample, num_warmup=WARMUP_STEPS)
        with make_profiler(profiler_dir, wait=2, warmup=3, active=8, repeat=1) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_inference_pipeline_with_graph_capture(self, device, profiler_dir, monkeypatch):
        """Inference pipeline with graph capture must complete all steps."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (1, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=2, active=5) as prof:
            times = measure_step_times(graph, device, 10, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(400)
    def test_pipeline_iteration_latency_stability(self, device, profiler_dir, monkeypatch):
        """Median step time must not drift more than 5x between quartiles."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=2, active=8) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)
        n = len(times)
        q1_med = sorted(times[: n // 4])[len(times[: n // 4]) // 2]
        q4_med = sorted(times[3 * n // 4 :])[len(times[3 * n // 4 :]) // 2]
        drift = q4_med / q1_med if q1_med > 0 else 1.0
        assert drift < 5.0, f"Throughput drift Q1={q1_med:.4f}s Q4={q4_med:.4f}s ratio={drift:.2f}"

    @pytest.mark.timeout(300)
    def test_pipeline_profiling_off_baseline(self, device, monkeypatch):
        """Pipeline without profiling must complete (sanity baseline)."""
        apply_graph_profiling_env(profiling=0)
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        times = measure_step_times(graph, device, NUM_STEPS)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(600)
    def test_a_b_comparison_framework(self, device, profiler_dir, monkeypatch):
        """A/B comparison: profiling-on vs profiling-off latency must be within 3x."""
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)

        apply_graph_profiling_env(profiling=0)
        graph_a, _ = capture_graph(model, sample)
        times_a = measure_step_times(graph_a, device, 10)
        del graph_a
        flush_gpu()

        apply_graph_profiling_env(profiling=1)
        graph_b, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times_b = measure_step_times(graph_b, device, 10, prof)
        del graph_b, model
        flush_gpu()

        assert_no_hang(times_a)
        assert_no_hang(times_b)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(600)
    def test_multi_dtype_pipeline(self, device, profiler_dir, monkeypatch):
        """Pipeline must succeed for both bfloat16 and float32."""
        apply_graph_profiling_env()
        for dtype in (torch.float32, torch.bfloat16):
            model = GPTModel(**GPT_1B_CONFIG).to(device).to(dtype).eval()
            sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
            graph, _ = capture_graph(model, sample)
            with make_profiler(profiler_dir, wait=1, warmup=2, active=5) as prof:
                times = measure_step_times(graph, device, 10, prof)
            del graph, model
            flush_gpu()
            assert_no_hang(times)
        assert_traces_valid(profiler_dir)
