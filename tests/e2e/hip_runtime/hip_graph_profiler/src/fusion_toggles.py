# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BN/Conv/Linear fusion toggle combination tests."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    FusableConvNet,
    GPTModel,
    apply_graph_profiling_env,
    assert_no_hang,
    assert_traces_valid,
    capture_graph,
    flush_gpu,
    make_profiler,
    measure_step_times,
)

_TOGGLE_MATRIX = [
    pytest.param(False, False, False, id="bn=False_convact=False_linact=False"),
    pytest.param(False, False, True, id="bn=False_convact=False_linact=True"),
    pytest.param(False, True, False, id="bn=False_convact=True_linact=False"),
    pytest.param(False, True, True, id="bn=False_convact=True_linact=True"),
    pytest.param(True, False, False, id="bn=True_convact=False_linact=False"),
    pytest.param(True, False, True, id="bn=True_convact=False_linact=True"),
    pytest.param(True, True, False, id="bn=True_convact=True_linact=False"),
    pytest.param(True, True, True, id="bn=True_convact=True_linact=True"),
]

NUM_STEPS = 10


class TestFusionToggleGraph:

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(("use_bn", "use_conv_act", "use_linear_act"), _TOGGLE_MATRIX)
    def test_fusion_combo_completes_with_profiling(
        self, device, profiler_dir, monkeypatch, use_bn, use_conv_act, use_linear_act
    ):
        """Each fusion combo must complete all steps without hang under profiling."""
        apply_graph_profiling_env()
        model = FusableConvNet(ch=64, use_bn=use_bn, use_conv_act=use_conv_act, use_linear_act=use_linear_act)
        model = model.to(device).eval()
        sample = torch.randn(4, 3, 32, 32, device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_all_fusions_enabled(self, device, profiler_dir, monkeypatch):
        """All fusion flags enabled must complete without hang."""
        apply_graph_profiling_env()
        model = FusableConvNet(ch=64, use_bn=True, use_conv_act=True, use_linear_act=True).to(device).eval()
        sample = torch.randn(4, 3, 32, 32, device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(180)
    def test_no_fusions_enabled(self, device, monkeypatch):
        """All fusion flags disabled with profiling=0 must complete cleanly."""
        apply_graph_profiling_env(profiling=0)
        model = FusableConvNet(ch=64, use_bn=False, use_conv_act=False, use_linear_act=False).to(device).eval()
        sample = torch.randn(4, 3, 32, 32, device=device)
        graph, _ = capture_graph(model, sample)
        times = measure_step_times(graph, device, NUM_STEPS)
        del graph, model
        flush_gpu()
        assert_no_hang(times)


class TestHeavyBatchFusion:

    @pytest.mark.timeout(300)
    def test_attention_fusion_50_iterations(self, device, profiler_dir, monkeypatch):
        """50-step replay with profiling must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(vocab=1024, dim=256, heads=8, layers=4, seq_len=64).to(device).eval()
        sample = torch.randint(0, 1024, (4, 64), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=2, active=10) as prof:
            times = measure_step_times(graph, device, 50, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_transformer_fusion_heavy_replay(self, device, profiler_dir, monkeypatch):
        """Deeper transformer heavy replay under profiling must not hang."""
        apply_graph_profiling_env()
        model = GPTModel(vocab=2048, dim=512, heads=8, layers=6, seq_len=128).to(device).eval()
        sample = torch.randint(0, 2048, (2, 128), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=8) as prof:
            times = measure_step_times(graph, device, 30, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(300)
    def test_mixed_conv_attention_fusion(self, device, profiler_dir, monkeypatch):
        """Conv model replay under profiling must not hang."""
        apply_graph_profiling_env()
        model = FusableConvNet(ch=128, use_bn=True, use_conv_act=True, use_linear_act=True).to(device).eval()
        sample = torch.randn(8, 3, 32, 32, device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=8) as prof:
            times = measure_step_times(graph, device, 30, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)
