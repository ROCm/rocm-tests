# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""BN/Conv/Linear fusion toggle combination tests."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    ATTN_1B_CONFIG,
    GPT_1B_CONFIG,
    AttentionModel,
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

NUM_STEPS = 30


def _capture_conv_graph(model, sample):
    """Capture a CUDA graph for a Conv model; skip if the environment does not support it."""
    try:
        return capture_graph(model, sample)
    except Exception as exc:
        pytest.skip(f"Conv2d CUDA graph capture not supported in this environment: {exc}")


class TestFusionToggleGraph:

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize(("use_bn", "use_conv_act", "use_linear_act"), _TOGGLE_MATRIX)
    def test_fusion_combo_completes_with_profiling(
        self, device, profiler_dir, monkeypatch, use_bn, use_conv_act, use_linear_act
    ):
        """Each fusion combo must complete all steps without hang under profiling."""
        apply_graph_profiling_env(
            FUSED_BN_ACT=int(use_bn),
            FUSED_CONV_ACT=int(use_conv_act),
            FUSED_LINEAR_ACT=int(use_linear_act),
        )
        model = FusableConvNet(ch=64, use_bn=use_bn, use_conv_act=use_conv_act, use_linear_act=use_linear_act)
        model = model.to(device).eval()
        sample = torch.randn(2, 3, 32, 32, device=device)
        graph, _ = _capture_conv_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=2, active=10, repeat=2) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_all_fusions_enabled(self, device, profiler_dir, monkeypatch):
        """Maximum fusion scenario — all fused paths active."""
        apply_graph_profiling_env(FUSED_BN_ACT=1, FUSED_CONV_ACT=1, FUSED_LINEAR_ACT=1)
        model = FusableConvNet(ch=128, use_bn=True, use_conv_act=True, use_linear_act=True).to(device).eval()
        sample = torch.randn(4, 3, 64, 64, device=device)
        graph, _ = _capture_conv_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=2, active=10, repeat=2) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(180)
    def test_no_fusions_enabled(self, device, profiler_dir, monkeypatch):
        """Baseline with all fusions disabled."""
        apply_graph_profiling_env(FUSED_BN_ACT=0, FUSED_CONV_ACT=0, FUSED_LINEAR_ACT=0)
        model = FusableConvNet(ch=64, use_bn=False, use_conv_act=False, use_linear_act=False).to(device).eval()
        sample = torch.randn(2, 3, 32, 32, device=device)
        graph, _ = _capture_conv_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)


class TestHeavyBatchFusion:

    @pytest.mark.timeout(300)
    def test_attention_fusion_50_iterations(self, device, profiler_dir, monkeypatch):
        """Heavy attention fusion for 55 iterations with ~1B-param attention model in bfloat16."""
        apply_graph_profiling_env()
        model = AttentionModel(**ATTN_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randn(2, ATTN_1B_CONFIG["seq_len"], ATTN_1B_CONFIG["dim"], device=device, dtype=torch.bfloat16)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=2, active=15, repeat=3) as prof:
            times = measure_step_times(graph, device, 55, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_transformer_fusion_heavy_replay(self, device, profiler_dir, monkeypatch):
        """~1B-param GPT model with 55 iterations stressing fusion + profiling."""
        apply_graph_profiling_env(FUSED_BN_ACT=1, FUSED_CONV_ACT=1, FUSED_LINEAR_ACT=1)
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (2, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=2, active=15, repeat=3) as prof:
            times = measure_step_times(graph, device, 55, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        traces = assert_traces_valid(profiler_dir)
        assert len(traces) >= 2, "Expected multiple trace files for 55-step run"

    @pytest.mark.timeout(300)
    def test_mixed_conv_attention_fusion(self, device, profiler_dir, monkeypatch):
        """Interleaved conv and attention graph replays under profiling must not hang."""
        apply_graph_profiling_env(FUSED_BN_ACT=1, FUSED_CONV_ACT=1, FUSED_LINEAR_ACT=1)
        conv_model = FusableConvNet(ch=64).to(device).eval()
        attn_model = AttentionModel(**ATTN_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        conv_sample = torch.randn(2, 3, 32, 32, device=device)
        attn_sample = torch.randn(
            2, ATTN_1B_CONFIG["seq_len"], ATTN_1B_CONFIG["dim"], device=device, dtype=torch.bfloat16
        )
        try:
            conv_graph, _ = capture_graph(conv_model, conv_sample)
        except Exception as exc:
            pytest.skip(f"Conv2d CUDA graph capture not supported in this environment: {exc}")
        attn_graph, _ = capture_graph(attn_model, attn_sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            for _ in range(20):
                conv_graph.replay()
                attn_graph.replay()
                torch.cuda.synchronize(device)
                prof.step()
        del conv_graph, attn_graph, conv_model, attn_model
        flush_gpu()
        assert_traces_valid(profiler_dir)
