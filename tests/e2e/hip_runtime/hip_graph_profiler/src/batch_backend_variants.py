# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Batch size sensitivity and attention backend variant tests."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    AttentionModel,
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
_BATCH_SIZES = [1, 2, 4, 8]
NUM_STEPS = 10


class TestBatchSizeSensitivity:

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("batch", _BATCH_SIZES)
    def test_graph_profiling_across_batch_sizes(self, device, profiler_dir, monkeypatch, batch):
        """GPT graph replay must succeed for all batch sizes under profiling."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (batch, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("batch", _BATCH_SIZES)
    def test_profiler_completeness_per_batch_size(self, device, profiler_dir, monkeypatch, batch):
        """Profiler must generate valid traces for every batch size."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (batch, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        traces = assert_traces_valid(profiler_dir)
        assert len(traces) >= 1

    @pytest.mark.timeout(300)
    def test_batch_size_1_minimal_topology(self, device, profiler_dir, monkeypatch):
        """Batch=1 minimal topology must complete without hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(300)
    def test_batch_size_8_larger_topology(self, device, profiler_dir, monkeypatch):
        """Batch=8 larger topology must complete without hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (8, _SMALL_CFG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)


class TestBackendVariants:

    @pytest.mark.timeout(400)
    @pytest.mark.parametrize("use_flash", [True, False])
    def test_flash_attention_toggle(self, device, profiler_dir, monkeypatch, use_flash):
        """Flash attention toggle must not hang under graph + profiling."""
        apply_graph_profiling_env()
        cfg = dict(dim=256, heads=8, layers=4, seq_len=64)
        model = AttentionModel(**cfg).to(device).to(torch.float16).eval()
        sample = torch.randn(2, cfg["seq_len"], cfg["dim"], device=device, dtype=torch.float16)
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=use_flash, enable_math=not use_flash, enable_mem_efficient=False
            ):
                graph, _ = capture_graph(model, sample)
        except Exception as exc:
            pytest.skip(f"Flash SDP backend not supported: {exc}")
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(400)
    @pytest.mark.parametrize("use_mem_eff", [True, False])
    def test_memory_efficient_attention_toggle(self, device, profiler_dir, monkeypatch, use_mem_eff):
        """Memory-efficient attention toggle must not hang under graph + profiling."""
        apply_graph_profiling_env()
        cfg = dict(dim=256, heads=8, layers=4, seq_len=64)
        model = AttentionModel(**cfg).to(device).eval()
        sample = torch.randn(2, cfg["seq_len"], cfg["dim"], device=device)
        try:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=not use_mem_eff, enable_mem_efficient=use_mem_eff
            ):
                graph, _ = capture_graph(model, sample)
        except Exception as exc:
            pytest.skip(f"Mem-efficient SDP backend not supported: {exc}")
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_attention_dtype_toggle(self, device, profiler_dir, monkeypatch, dtype):
        """Attention model graph replay must succeed for both float16 and bfloat16."""
        apply_graph_profiling_env()
        cfg = dict(dim=256, heads=8, layers=4, seq_len=64)
        model = AttentionModel(**cfg).to(device).to(dtype).eval()
        sample = torch.randn(2, cfg["seq_len"], cfg["dim"], device=device, dtype=dtype)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(600)
    def test_all_backends_sequential(self, device, profiler_dir, monkeypatch):
        """Default SDPA backend must complete all steps under graph + profiling."""
        apply_graph_profiling_env()
        cfg = dict(dim=256, heads=8, layers=4, seq_len=64)
        model = AttentionModel(**cfg).to(device).eval()
        sample = torch.randn(2, cfg["seq_len"], cfg["dim"], device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)
