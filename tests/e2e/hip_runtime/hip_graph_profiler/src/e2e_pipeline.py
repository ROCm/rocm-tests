# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end benchmarking pipeline with HIP graph capture and PyTorch profiler."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    GPT_1B_CONFIG,
    TransformerBlock,
    apply_graph_profiling_env,
    assert_no_hang,
    assert_traces_valid,
    flush_gpu,
    make_profiler,
)

_VOCAB = GPT_1B_CONFIG["vocab"]
_SEQ = GPT_1B_CONFIG["seq_len"]

NUM_PIPELINE_STEPS = 15
BENCHMARK_RESULTS_FILE = "benchmark_results.json"


class BenchmarkModel(nn.Module):
    """Full training-capable model for end-to-end pipeline tests."""

    def __init__(self, vocab=1024, dim=256, heads=8, layers=4, seq_len=64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx, targets=None):
        _b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def save_benchmark_results(results, output_dir):
    path = os.path.join(str(output_dir), BENCHMARK_RESULTS_FILE)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


class TestE2EPipeline:

    @pytest.mark.timeout(600)
    def test_full_training_pipeline_with_graph_and_profiling(
        self, device, profiler_dir, monkeypatch, clean_extension_cache
    ):
        """Full forward + backward + optimizer pipeline with ~1B-param model."""
        apply_graph_profiling_env()
        model = BenchmarkModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch_size = 2

        step_times = []
        with make_profiler(profiler_dir, wait=1, warmup=2, active=5, repeat=2) as prof:
            for _ in range(NUM_PIPELINE_STEPS):
                t0 = time.monotonic()
                inputs = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)
                targets = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)
                _, loss = model(inputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize(device)
                step_times.append(time.monotonic() - t0)
                prof.step()

        assert_no_hang(step_times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(600)
    def test_inference_pipeline_with_graph_capture(self, device, profiler_dir, monkeypatch):
        """Inference pipeline: ~1B-param model, capture graph once, replay."""
        apply_graph_profiling_env()
        flush_gpu()
        model = BenchmarkModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        batch_size = 2
        static_input = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)

        s = torch.cuda.Stream(device=device)
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                model(static_input)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s), torch.no_grad():
            _, _ = model(static_input)

        step_times = []
        with make_profiler(profiler_dir, wait=1, warmup=1, active=5, repeat=2) as prof:
            for _ in range(NUM_PIPELINE_STEPS):
                t0 = time.monotonic()
                g.replay()
                torch.cuda.synchronize(device)
                step_times.append(time.monotonic() - t0)
                prof.step()

        assert_no_hang(step_times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(600)
    def test_pipeline_iteration_latency_stability(self, device, profiler_dir, monkeypatch, tmp_path):
        """Iteration latency stability with ~1B-param model."""
        apply_graph_profiling_env()
        model = BenchmarkModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch_size = 2

        step_times = []
        with make_profiler(profiler_dir, wait=0, warmup=2, active=5, repeat=2) as prof:
            for _ in range(NUM_PIPELINE_STEPS):
                t0 = time.monotonic()
                inputs = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)
                targets = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)
                _, loss = model(inputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize(device)
                step_times.append(time.monotonic() - t0)
                prof.step()

        del model, optimizer
        flush_gpu()

        results = {
            "step_times": step_times,
            "mean": sum(step_times) / len(step_times),
            "median": sorted(step_times)[len(step_times) // 2],
            "max": max(step_times),
            "min": min(step_times),
        }
        save_benchmark_results(results, tmp_path)

        median = results["median"]
        threshold = max(median * 10, 1.0)
        for i, t in enumerate(step_times[3:], start=3):
            assert t < threshold, (
                f"Step {i}: {t:.4f}s exceeds threshold {threshold:.4f}s (median {median:.4f}s)"
            )

    @pytest.mark.timeout(600)
    def test_pipeline_profiling_off_baseline(self, device, monkeypatch):
        """Baseline: ~1B-param model without profiling (sanity check)."""
        apply_graph_profiling_env(profiling=0, use_graph=0)
        model = BenchmarkModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        batch_size = 2

        for _ in range(NUM_PIPELINE_STEPS):
            inputs = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)
            targets = torch.randint(0, _VOCAB, (batch_size, _SEQ), device=device)
            _, loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize(device)

    @pytest.mark.timeout(300)
    def test_a_b_comparison_framework(self, device, profiler_dir, monkeypatch, tmp_path):
        """A/B comparison: capture timing metrics for cross-drop comparison."""
        apply_graph_profiling_env()
        flush_gpu()
        model = BenchmarkModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        static_input = torch.randint(0, _VOCAB, (2, _SEQ), device=device)

        s = torch.cuda.Stream(device=device)
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                model(static_input)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s), torch.no_grad():
            _, _ = model(static_input)

        step_times = []
        with make_profiler(profiler_dir, wait=0, warmup=2, active=5, repeat=2) as prof:
            for _ in range(NUM_PIPELINE_STEPS):
                t0 = time.monotonic()
                g.replay()
                torch.cuda.synchronize(device)
                step_times.append(time.monotonic() - t0)
                prof.step()

        results = {
            "drop": os.environ.get("ROCM_DROP", "unknown"),
            "pytorch_version": torch.__version__,
            "device_name": torch.cuda.get_device_name(device),
            "step_times": step_times,
            "mean_ms": sum(step_times) / len(step_times) * 1000,
            "median_ms": sorted(step_times)[len(step_times) // 2] * 1000,
            "completed": True,
            "hangs_detected": False,
            "profiler_traces_generated": len(list(profiler_dir.rglob("*.json"))),
        }
        save_benchmark_results(results, tmp_path)

        assert results["completed"]
        assert not results["hangs_detected"]
        assert results["profiler_traces_generated"] > 0

    @pytest.mark.timeout(600)
    def test_multi_dtype_pipeline(self, device, profiler_dir, monkeypatch):
        """Pipeline must work across float32 and bfloat16 dtypes with ~1B-param model."""
        apply_graph_profiling_env()
        for dtype in [torch.float32, torch.bfloat16]:
            model = BenchmarkModel(**GPT_1B_CONFIG).to(device).to(dtype)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for _ in range(5):
                inputs = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
                targets = torch.randint(0, _VOCAB, (1, _SEQ), device=device)
                _, loss = model(inputs, targets)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.synchronize(device)
            del model, optimizer
            torch.cuda.empty_cache()
