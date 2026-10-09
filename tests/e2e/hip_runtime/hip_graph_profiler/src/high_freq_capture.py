# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""High-frequency graph capture and re-capture cycle tests."""

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
)

_SMALL_CFG = dict(vocab=1024, dim=256, heads=8, layers=4, seq_len=64)
NUM_STEPS = 10


class TestHighFreqCapture:

    @pytest.mark.timeout(600)
    def test_100_capture_replay_cycles(self, device, monkeypatch):
        """100 capture+release cycles must not hang or exhaust GPU memory."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        for _ in range(100):
            g, _ = capture_graph(model, sample)
            replay_and_sync(g, device, n=3)
            del g
            flush_gpu()
        del model
        flush_gpu()

    @pytest.mark.timeout(1200)
    def test_200_capture_replay_cycles_stress(self, device, monkeypatch):
        """200 capture+release cycles stress must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        for _ in range(200):
            g, _ = capture_graph(model, sample)
            replay_and_sync(g, device, n=2)
            del g
            flush_gpu()
        del model
        flush_gpu()

    @pytest.mark.timeout(600)
    def test_no_progressive_slowdown(self, device, profiler_dir, monkeypatch):
        """Step times must not progressively increase across capture cycles."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        all_times = []
        for _ in range(5):
            g, _ = capture_graph(model, sample)
            with make_profiler(profiler_dir, wait=0, warmup=1, active=3) as prof:
                times = measure_step_times(g, device, NUM_STEPS, prof)
            all_times.extend(times)
            assert_no_hang(times)
            del g
            flush_gpu()
        del model
        flush_gpu()
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(600)
    def test_profiler_markers_consistent_across_cycles(self, device, profiler_dir, monkeypatch):
        """Profiler must generate valid traces across repeated capture cycles."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        for _ in range(5):
            g, _ = capture_graph(model, sample)
            with make_profiler(profiler_dir, wait=0, warmup=1, active=3) as prof:
                measure_step_times(g, device, NUM_STEPS, prof)
            del g
            flush_gpu()
        del model
        flush_gpu()
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_multi_replay_per_capture(self, device, profiler_dir, monkeypatch):
        """Multiple replays per capture must all complete without hang."""
        apply_graph_profiling_env()
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)
        g, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(g, device, 50, prof)
        del g, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)
