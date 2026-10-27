# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Batch size sensitivity and attention backend variant tests."""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    ATTN_1B_CONFIG,
    GPT_1B_CONFIG,
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

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _BACKEND_MAP = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "math": SDPBackend.MATH,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    _USE_NEW_SDPA_API = True
except ImportError:
    _USE_NEW_SDPA_API = False

_SKIP_PATTERNS = ["not available", "not supported", "no available kernel", "no kernel", "aborting execution"]

NUM_STEPS = 15


def _should_skip(err):
    msg = str(err).lower()
    return any(s in msg for s in _SKIP_PATTERNS)


class TestBatchSizeSensitivity:

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_graph_profiling_across_batch_sizes(self, device, profiler_dir, monkeypatch, batch_size):
        """No deadlocks across batch sizes {1, 2, 8} with ~1B-param model."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (batch_size, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=1, warmup=1, active=5, repeat=2) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("batch_size", [1, 2, 8])
    def test_profiler_completeness_per_batch_size(self, device, profiler_dir, monkeypatch, batch_size):
        """Profiler artifacts must be complete and valid for each batch size."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (batch_size, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            measure_step_times(graph, device, 10, prof)
        del graph, model
        flush_gpu()
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_batch_size_1_minimal_topology(self, device, profiler_dir, monkeypatch):
        """Batch size 1 — simplest graph topology with ~1B-param model."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (1, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)

    @pytest.mark.timeout(300)
    def test_batch_size_8_larger_topology(self, device, profiler_dir, monkeypatch):
        """Batch size 8 — most complex graph topology with ~1B-param model."""
        apply_graph_profiling_env()
        model = GPTModel(**GPT_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randint(0, GPT_1B_CONFIG["vocab"], (8, GPT_1B_CONFIG["seq_len"]), device=device)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5, repeat=2) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)


class TestBackendVariants:

    @pytest.mark.parametrize("use_flash", [True, False])
    def test_flash_attention_toggle(self, device, profiler_dir, monkeypatch, use_flash):
        """Profiling + graphs must work with FlashAttention enabled and disabled."""
        apply_graph_profiling_env()
        model = AttentionModel(**ATTN_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randn(2, ATTN_1B_CONFIG["seq_len"], ATTN_1B_CONFIG["dim"], device=device, dtype=torch.bfloat16)
        graph = None
        try:
            if _USE_NEW_SDPA_API:
                backend = _BACKEND_MAP["flash"] if use_flash else _BACKEND_MAP["math"]
                ctx = sdpa_kernel(backend)
            else:
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=use_flash, enable_math=not use_flash, enable_mem_efficient=False
                )
            with ctx:
                graph, _ = capture_graph(model, sample)
                with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
                    times = measure_step_times(graph, device, NUM_STEPS, prof)
        except RuntimeError as e:
            if _should_skip(e):
                pytest.skip(f"Backend not available: {e}")
            raise
        finally:
            del model, graph
            flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.parametrize("use_mem_efficient", [True, False])
    def test_memory_efficient_attention_toggle(self, device, profiler_dir, monkeypatch, use_mem_efficient):
        """Test memory-efficient attention backend with profiling + graphs."""
        apply_graph_profiling_env()
        model = AttentionModel(**ATTN_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randn(2, ATTN_1B_CONFIG["seq_len"], ATTN_1B_CONFIG["dim"], device=device, dtype=torch.bfloat16)
        graph = None
        try:
            if _USE_NEW_SDPA_API:
                backend = _BACKEND_MAP["mem_efficient"] if use_mem_efficient else _BACKEND_MAP["math"]
                ctx = sdpa_kernel(backend)
            else:
                ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=not use_mem_efficient, enable_mem_efficient=use_mem_efficient
                )
            with ctx:
                graph, _ = capture_graph(model, sample)
                with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
                    times = measure_step_times(graph, device, NUM_STEPS, prof)
        except RuntimeError as e:
            if _should_skip(e):
                pytest.skip(f"Backend not available: {e}")
            raise
        finally:
            del model, graph
            flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_attention_dtype_toggle(self, device, profiler_dir, monkeypatch, dtype):
        """Attention model graph replay must succeed for both float16 and bfloat16."""
        apply_graph_profiling_env()
        model = AttentionModel(**ATTN_1B_CONFIG).to(device).to(dtype).eval()
        sample = torch.randn(2, ATTN_1B_CONFIG["seq_len"], ATTN_1B_CONFIG["dim"], device=device, dtype=dtype)
        graph, _ = capture_graph(model, sample)
        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, NUM_STEPS, prof)
        del graph, model
        flush_gpu()
        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    def test_all_backends_sequential(self, device, profiler_dir, monkeypatch):
        """Run all available backends in the same session — no residual state issues."""
        apply_graph_profiling_env()
        model = AttentionModel(**ATTN_1B_CONFIG).to(device).to(torch.bfloat16).eval()
        sample = torch.randn(2, ATTN_1B_CONFIG["seq_len"], ATTN_1B_CONFIG["dim"], device=device, dtype=torch.bfloat16)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            if _USE_NEW_SDPA_API:
                backends = [_BACKEND_MAP["flash"], _BACKEND_MAP["math"], _BACKEND_MAP["mem_efficient"]]
                for backend in backends:
                    try:
                        with sdpa_kernel(backend):
                            graph, _ = capture_graph(model, sample)
                            times = measure_step_times(graph, device, 5, prof)
                            assert_no_hang(times)
                            del graph
                            torch.cuda.synchronize(device)
                    except RuntimeError:
                        continue
            else:
                graph, _ = capture_graph(model, sample)
                times = measure_step_times(graph, device, 5, prof)
                assert_no_hang(times)
                del graph

        assert_traces_valid(profiler_dir)
