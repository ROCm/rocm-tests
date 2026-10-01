# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared model definitions and utilities for HIP graph profiling worker scripts."""

from __future__ import annotations

import contextlib
import gc
import json
import multiprocessing as _mp
import time
import traceback

import torch
import torch.nn as nn

DEADLOCK_TIMEOUT_SEC = 120
NO_PROGRESS_THRESHOLD_SEC = 60

GPT_1B_CONFIG = dict(vocab=8192, dim=2048, heads=16, layers=20, seq_len=256)


class DeadlockTimeoutError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise DeadlockTimeoutError("Test exceeded timeout — potential deadlock detected")


def _is_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_current_stream_capturing", None)
    if fn is None:
        return False
    with contextlib.suppress(Exception):
        return bool(fn())
    return False


def safe_abort_capture(graph=None, stream=None) -> None:
    """Force-release any in-progress graph capture and reset GPU state."""
    if not torch.cuda.is_available():
        return
    if stream is not None and graph is not None:
        with contextlib.suppress(Exception), torch.cuda.stream(stream):
            if _is_stream_capturing():
                with contextlib.suppress(Exception):
                    graph.capture_end()
    if graph is not None:
        with contextlib.suppress(Exception):
            graph.reset()
        with contextlib.suppress(Exception):
            del graph
    if stream is not None:
        with contextlib.suppress(Exception):
            del stream
    gc.collect()
    gc.collect()
    with contextlib.suppress(Exception):
        torch.cuda.synchronize()
    with contextlib.suppress(Exception):
        torch.cuda.empty_cache()


def flush_gpu() -> None:
    """Synchronize and free GPU memory; abort any stray capture."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        if _is_stream_capturing():
            safe_abort_capture()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def apply_graph_profiling_env(profiling: int = 1, use_graph: int = 1, **extra) -> None:
    """Set standard HIP graph + profiling environment variables in os.environ."""
    import os

    env = {
        "PROFILING": str(profiling),
        "USE_GRAPH": str(use_graph),
        "HIP_LAUNCH_BLOCKING": "0",
        "ROCBLAS_USE_HIPBLASLT": "1",
        "MIOPEN_FIND_MODE": "3",
        "MIOPEN_FIND_ENFORCE": "3",
    }
    env.update({k: str(v) for k, v in extra.items()})
    for k, v in env.items():
        os.environ[k] = v


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """Scaled-down GPT-like model for exercising transformer code paths under graph capture."""

    def __init__(self, vocab: int = 1024, dim: int = 256, heads: int = 8, layers: int = 4, seq_len: int = 64) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.seq_len = seq_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _b, t = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(t, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


class FusableConvNet(nn.Module):
    """Conv/BN/Linear model for fusion toggle testing."""

    def __init__(
        self, ch: int = 64, use_bn: bool = True, use_conv_act: bool = True, use_linear_act: bool = True
    ) -> None:
        super().__init__()
        feat: list[nn.Module] = [nn.Conv2d(3, ch, 3, padding=1)]
        if use_bn:
            feat.append(nn.BatchNorm2d(ch))
        if use_conv_act:
            feat.append(nn.ReLU(inplace=True))
        feat.append(nn.Conv2d(ch, ch, 3, padding=1))
        if use_bn:
            feat.append(nn.BatchNorm2d(ch))
        if use_conv_act:
            feat.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*feat)
        self.pool = nn.AdaptiveAvgPool2d(1)
        cls: list[nn.Module] = [nn.Linear(ch, ch)]
        if use_linear_act:
            cls.append(nn.ReLU(inplace=True))
        cls.append(nn.Linear(ch, 10))
        self.classifier = nn.Sequential(*cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


class AttentionModel(nn.Module):
    """Multi-layer attention model exercising scaled_dot_product_attention backends."""

    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 4, seq_len: int = 64) -> None:
        super().__init__()
        self.layers_list = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ln": nn.LayerNorm(dim),
                        "qkv": nn.Linear(dim, dim * 3),
                        "proj": nn.Linear(dim, dim),
                        "ffn_ln": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)),
                    }
                )
                for _ in range(layers)
            ]
        )
        self.heads = heads
        self.head_dim = dim // heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        for layer in self.layers_list:
            h = layer["ln"](x)
            qkv = layer["qkv"](h).reshape(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            x = x + layer["proj"](attn_out.transpose(1, 2).reshape(b, t, c))
            x = x + layer["ffn"](layer["ffn_ln"](x))
        return x


def capture_graph(model: nn.Module, sample_input: torch.Tensor, num_warmup: int = 3, stream=None):
    """Warmup then capture a CUDA graph for model(sample_input). Returns (graph, static_out)."""
    flush_gpu()
    s = stream or torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(num_warmup):
            model(sample_input)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:
        try:
            ctx = torch.cuda.graph(g, stream=s, capture_error_mode="thread_local")
        except TypeError:
            ctx = torch.cuda.graph(g, stream=s)
        with ctx:
            static_out = model(sample_input)
    except Exception:
        safe_abort_capture(g, stream=s)
        raise
    return g, static_out


def replay_and_sync(graph, device, n: int = 1) -> None:
    for _ in range(n):
        graph.replay()
    torch.cuda.synchronize(device)


def make_profiler(trace_dir, wait: int = 1, warmup: int = 1, active: int = 3, repeat: int = 1):
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        with_stack=True,
    )


def measure_step_times(graph, device, num_steps: int, profiler=None) -> list[float]:
    times = []
    for _ in range(num_steps):
        t0 = time.monotonic()
        graph.replay()
        torch.cuda.synchronize(device)
        times.append(time.monotonic() - t0)
        if profiler is not None:
            profiler.step()
    return times


def assert_no_hang(step_times: list[float], threshold: float = NO_PROGRESS_THRESHOLD_SEC) -> None:
    for i, t in enumerate(step_times):
        assert t < threshold, f"Step {i} took {t:.1f}s — exceeds threshold of {threshold}s"


def assert_traces_valid(profiler_dir) -> list:
    import pathlib

    traces = list(pathlib.Path(profiler_dir).rglob("*.json"))
    assert len(traces) > 0, "No profiler trace files generated"
    for f in traces:
        with open(f) as fh:
            data = json.load(fh)
        assert isinstance(data, (dict, list)), f"Trace {f.name} is not valid JSON"
    return traces


def _subprocess_entry(target, args, kwargs, conn):
    try:
        conn.send(("ok", target(*args, **kwargs)))
    except BaseException as exc:
        conn.send(("err", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def run_in_subprocess(target, args=(), kwargs=None, timeout: int = 120):
    """Run target in a fresh spawn subprocess and return its result; raises on timeout or error."""
    if kwargs is None:
        kwargs = {}
    ctx = _mp.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_subprocess_entry, args=(target, args, kwargs, child))
    proc.start()
    child.close()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        raise TimeoutError(f"Subprocess {getattr(target, '__name__', target)!r} exceeded {timeout}s")
    payload = parent.recv() if parent.poll() else None
    parent.close()
    if payload is None:
        raise RuntimeError(f"Subprocess died without result (exitcode={proc.exitcode})")
    if payload[0] == "ok":
        return payload[1]
    _kind, msg, tb = payload
    raise RuntimeError(f"Subprocess raised:\n{msg}\n\nTraceback:\n{tb}")


class _UncapturableOpWorker(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.ones(1, device="cpu").to(x.device)


def bad_capture_worker(layer_dims: list[int], in_shape: tuple[int, ...], num_warmup: int = 3) -> bool:
    """Deliberately-failing graph capture; returns True if it raised as expected."""
    import torch as _torch
    import torch.nn as _nn

    if not _torch.cuda.is_available():
        return False
    dev = _torch.device("cuda", 0)
    _torch.cuda.set_device(dev)
    layers: list[_nn.Module] = []
    for i in range(len(layer_dims) - 1):
        layers.append(_nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i == 0:
            layers.append(_UncapturableOpWorker())
    model = _nn.Sequential(*layers).to(dev).eval()
    sample = _torch.randn(*in_shape, device=dev)
    s = _torch.cuda.Stream(device=dev)
    with _torch.cuda.stream(s):
        for _ in range(num_warmup):
            with contextlib.suppress(Exception):
                model(sample)
    _torch.cuda.synchronize()
    g = _torch.cuda.CUDAGraph()
    raised = False
    try:
        try:
            ctx = _torch.cuda.graph(g, stream=s, capture_error_mode="thread_local")
        except TypeError:
            ctx = _torch.cuda.graph(g, stream=s)
        with ctx:
            _ = model(sample)
    except Exception:
        raised = True
    finally:
        with contextlib.suppress(Exception):
            g.reset()
        del g
        with contextlib.suppress(Exception):
            _torch.cuda.synchronize()
    return raised
