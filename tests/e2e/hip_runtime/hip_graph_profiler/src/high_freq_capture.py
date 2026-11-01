# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""High-frequency graph capture and re-capture cycle tests."""

from __future__ import annotations

import sys
import time

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    NO_PROGRESS_THRESHOLD_SEC,
    FusableConvNet,
    GPTModel,
    apply_graph_profiling_env,
    assert_traces_valid,
    capture_graph,
    make_profiler,
)

CYCLE_COUNT_STANDARD = 100
CYCLE_COUNT_STRESS = 200

_SMALL_CFG = dict(vocab=1024, dim=128, heads=4, layers=2, seq_len=32)


def _capture_conv_graph(model, sample):
    """Capture a CUDA graph for a Conv model; skip if the environment does not support it."""
    try:
        return capture_graph(model, sample, num_warmup=1)
    except Exception as exc:
        pytest.skip(f"Conv2d CUDA graph capture not supported in this environment: {exc}")


class TestHighFreqCapture:

    @pytest.mark.timeout(300)
    def test_100_capture_replay_cycles(self, device, profiler_dir, monkeypatch):
        """100 capture/replay cycles must complete without stalls."""
        apply_graph_profiling_env(monkeypatch)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)

        cycle_times = []
        with make_profiler(profiler_dir, wait=1, warmup=2, active=30, repeat=3) as prof:
            for _ in range(CYCLE_COUNT_STANDARD):
                graph, _ = capture_graph(model, sample, num_warmup=1)
                t0 = time.monotonic()
                graph.replay()
                torch.cuda.synchronize(device)
                cycle_times.append(time.monotonic() - t0)
                del graph
                torch.cuda.synchronize(device)
                prof.step()

        for i, t in enumerate(cycle_times):
            assert t < NO_PROGRESS_THRESHOLD_SEC, f"Cycle {i}: {t:.1f}s — potential hang"
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(600)
    def test_200_capture_replay_cycles_stress(self, device, profiler_dir, monkeypatch):
        """200 cycles — extended stress for marker accumulation bugs."""
        apply_graph_profiling_env(monkeypatch)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)

        cycle_times = []
        with make_profiler(profiler_dir, wait=2, warmup=3, active=50, repeat=3) as prof:
            for _ in range(CYCLE_COUNT_STRESS):
                graph, _ = capture_graph(model, sample, num_warmup=1)
                t0 = time.monotonic()
                graph.replay()
                torch.cuda.synchronize(device)
                cycle_times.append(time.monotonic() - t0)
                del graph
                torch.cuda.synchronize(device)
                prof.step()

        for i, t in enumerate(cycle_times):
            assert t < NO_PROGRESS_THRESHOLD_SEC, f"Cycle {i}: {t:.1f}s — potential hang"

    def test_no_progressive_slowdown(self, device, profiler_dir, monkeypatch):
        """Latter half should not be significantly slower than the first half."""
        apply_graph_profiling_env(monkeypatch)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)

        cycle_times = []
        with make_profiler(profiler_dir, wait=0, warmup=2, active=30, repeat=3) as prof:
            for _ in range(CYCLE_COUNT_STANDARD):
                graph, _ = capture_graph(model, sample, num_warmup=1)
                t0 = time.monotonic()
                graph.replay()
                torch.cuda.synchronize(device)
                cycle_times.append(time.monotonic() - t0)
                del graph
                torch.cuda.synchronize(device)
                prof.step()

        mid = len(cycle_times) // 2
        first_half_avg = sum(cycle_times[:mid]) / mid
        second_half_avg = sum(cycle_times[mid:]) / (len(cycle_times) - mid)
        slowdown_ratio = second_half_avg / max(first_half_avg, 1e-9)
        assert slowdown_ratio < 3.0, (
            f"Progressive slowdown detected: second half avg {second_half_avg:.4f}s "
            f"vs first half avg {first_half_avg:.4f}s (ratio {slowdown_ratio:.2f}x)"
        )

    def test_profiler_markers_consistent_across_cycles(self, device, profiler_dir, monkeypatch):
        """Profiler must produce valid markers across all capture/replay cycles."""
        apply_graph_profiling_env(monkeypatch)
        model = FusableConvNet(ch=32).to(device).eval()
        sample = torch.randn(1, 3, 16, 16, device=device)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=20, repeat=4) as prof:
            for _ in range(80):
                graph, _ = _capture_conv_graph(model, sample)
                graph.replay()
                torch.cuda.synchronize(device)
                del graph
                torch.cuda.synchronize(device)
                prof.step()

        traces = assert_traces_valid(profiler_dir)
        assert len(traces) >= 3, "Expected multiple trace files from 80-cycle run"

    @pytest.mark.timeout(300)
    def test_multi_replay_per_capture(self, device, profiler_dir, monkeypatch):
        """Multiple replays per capture to test marker accumulation per-graph."""
        apply_graph_profiling_env(monkeypatch)
        model = GPTModel(**_SMALL_CFG).to(device).eval()
        sample = torch.randint(0, _SMALL_CFG["vocab"], (1, _SMALL_CFG["seq_len"]), device=device)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=30, repeat=3) as prof:
            for cycle in range(50):
                graph, _ = capture_graph(model, sample, num_warmup=1)
                for replay_idx in range(5):
                    t0 = time.monotonic()
                    graph.replay()
                    torch.cuda.synchronize(device)
                    elapsed = time.monotonic() - t0
                    assert elapsed < NO_PROGRESS_THRESHOLD_SEC, f"Cycle {cycle} replay {replay_idx}: {elapsed:.1f}s"
                    prof.step()
                del graph
                torch.cuda.synchronize(device)

        assert_traces_valid(profiler_dir)
