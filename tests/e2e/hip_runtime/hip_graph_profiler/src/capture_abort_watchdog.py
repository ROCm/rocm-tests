# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Capture abort with profiling active and forced timeout resilience tests."""

from __future__ import annotations

import sys
import time

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from helpers import (
    NO_PROGRESS_THRESHOLD_SEC,
    GPTModel,
    _bad_capture_worker,
    apply_graph_profiling_env,
    assert_no_hang,
    assert_traces_valid,
    capture_graph,
    make_profiler,
    measure_step_times,
    run_in_subprocess,
)

# On ROCm/HIP, a failed torch.cuda.graph() capture leaves the HIP driver's
# process-global stream-capture flag set. No in-process API clears it, so any
# subsequent host-to-device copy or allocation raises
# "HIP error: operation not permitted when stream is capturing".
# All TestCaptureAbort tests therefore run the deliberately-failing capture in a
# freshly-spawned subprocess (run_in_subprocess). The wedged HIP state dies with
# the child; the parent retains a clean device for recovery assertions.


class TestCaptureAbort:

    def test_invalid_op_aborts_capture_cleanly(self, device, monkeypatch):
        """Graph capture with an uncapturable op must raise, not deadlock."""
        apply_graph_profiling_env()
        raised = run_in_subprocess(_bad_capture_worker, args=([64, 64, 64], (2, 64)), timeout=60)
        assert raised, (
            "Subprocess capture with UncapturableOp did not raise — "
            "graph-capture abort path is not exercising the failure mode."
        )
        torch.cuda.synchronize(device)

    def test_subsequent_capture_succeeds_after_abort(self, device, monkeypatch):
        """After a failed capture (in an isolated subprocess), a valid capture must still succeed."""
        apply_graph_profiling_env()
        raised = run_in_subprocess(_bad_capture_worker, args=([64, 64], (2, 64)), timeout=60)
        assert raised
        torch.cuda.synchronize(device)

        good_model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).to(device).eval()
        good_sample = torch.randn(2, 64, device=device)
        graph, _ = capture_graph(good_model, good_sample)
        graph.replay()
        torch.cuda.synchronize(device)

    def test_no_lingering_signals_after_abort(self, device, profiler_dir, monkeypatch):
        """After capture failure, profiled graph replay must work without residual signal interference."""
        apply_graph_profiling_env()
        raised = run_in_subprocess(_bad_capture_worker, args=([32, 32], (1, 32)), timeout=60)
        assert raised
        torch.cuda.synchronize(device)

        good_model = GPTModel(vocab=1024, dim=128, heads=4, layers=2, seq_len=16).to(device).eval()
        good_sample = torch.randint(0, 1024, (1, 16), device=device)
        graph, _ = capture_graph(good_model, good_sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            times = measure_step_times(graph, device, 10, prof)

        assert_no_hang(times)
        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(300)
    def test_repeated_abort_recovery_cycles(self, device, monkeypatch):
        """Multiple abort-then-recover cycles must not accumulate failures."""
        apply_graph_profiling_env()
        good_model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)).to(device).eval()
        sample = torch.randn(1, 32, device=device)

        for cycle in range(10):
            raised = run_in_subprocess(_bad_capture_worker, args=([32, 32], (1, 32)), timeout=60)
            assert raised, f"Cycle {cycle}: bad capture did not raise"
            torch.cuda.synchronize(device)

            graph, _ = capture_graph(good_model, sample)
            t0 = time.monotonic()
            graph.replay()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - t0
            assert elapsed < NO_PROGRESS_THRESHOLD_SEC, (
                f"Cycle {cycle}: recovery replay took {elapsed:.1f}s"
            )
            del graph


class TestWatchdogTimeout:

    @pytest.mark.timeout(90)
    def test_system_remains_responsive_under_load(self, device, profiler_dir, monkeypatch):
        """Heavy workload with profiling must not make the system unresponsive."""
        apply_graph_profiling_env()
        model = GPTModel(vocab=1024, dim=256, heads=8, layers=4, seq_len=64).to(device).eval()
        sample = torch.randint(0, 1024, (2, 64), device=device)
        graph, _ = capture_graph(model, sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            for step in range(20):
                t0 = time.monotonic()
                graph.replay()
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - t0
                assert elapsed < NO_PROGRESS_THRESHOLD_SEC, (
                    f"Step {step}: {elapsed:.1f}s — system may be unresponsive"
                )
                prof.step()

        assert_traces_valid(profiler_dir)

    @pytest.mark.timeout(90)
    def test_concurrent_streams_no_indefinite_hang(self, device, profiler_dir, monkeypatch):
        """Multiple concurrent streams must not produce indefinite queue stalls."""
        apply_graph_profiling_env()
        streams = [torch.cuda.Stream(device=device) for _ in range(8)]
        dim = 128

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for iteration in range(20):
                for s in streams:
                    with torch.cuda.stream(s):
                        a = torch.randn(dim, dim, device=device)
                        for _ in range(10):
                            a = torch.matmul(a, a)
                            a = a / a.norm()
                t0 = time.monotonic()
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - t0
                assert elapsed < NO_PROGRESS_THRESHOLD_SEC, (
                    f"Iteration {iteration}: sync took {elapsed:.1f}s"
                )
                prof.step()

        assert_traces_valid(profiler_dir)

    def test_graceful_timeout_with_artificial_cpu_delay(self, device, profiler_dir, monkeypatch):
        """CPU-side delays should not trigger GPU-side deadlocks."""
        apply_graph_profiling_env()
        model = GPTModel(vocab=1024, dim=128, heads=4, layers=2, seq_len=32).to(device).eval()
        sample = torch.randint(0, 1024, (1, 32), device=device)
        graph, _ = capture_graph(model, sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=5) as prof:
            for _ in range(10):
                time.sleep(0.1)
                t0 = time.monotonic()
                graph.replay()
                torch.cuda.synchronize(device)
                gpu_time = time.monotonic() - t0
                assert gpu_time < NO_PROGRESS_THRESHOLD_SEC
                prof.step()

        assert_traces_valid(profiler_dir)

    def test_rapid_sync_cycles_no_queue_stuck(self, device, monkeypatch):
        """Rapid synchronize() cycles must not leave kernel queues stuck."""
        apply_graph_profiling_env()
        model = GPTModel(vocab=1024, dim=64, heads=4, layers=1, seq_len=16).to(device).eval()
        sample = torch.randint(0, 1024, (1, 16), device=device)
        graph, _ = capture_graph(model, sample)

        for step in range(100):
            graph.replay()
            t0 = time.monotonic()
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - t0
            assert elapsed < 10.0, f"Step {step}: sync took {elapsed:.1f}s — queue may be stuck"

    @pytest.mark.timeout(120)
    def test_mixed_graph_eager_with_profiling(self, device, profiler_dir, monkeypatch):
        """Alternating between graph replay and eager execution under profiling."""
        apply_graph_profiling_env()
        model = GPTModel(vocab=1024, dim=128, heads=4, layers=2, seq_len=32).to(device).eval()
        sample = torch.randint(0, 1024, (1, 32), device=device)
        graph, _ = capture_graph(model, sample)

        with make_profiler(profiler_dir, wait=0, warmup=1, active=10) as prof:
            for step in range(20):
                if step % 2 == 0:
                    graph.replay()
                else:
                    with torch.no_grad():
                        _ = model(sample)
                t0 = time.monotonic()
                torch.cuda.synchronize(device)
                elapsed = time.monotonic() - t0
                assert elapsed < NO_PROGRESS_THRESHOLD_SEC
                prof.step()

        assert_traces_valid(profiler_dir)
