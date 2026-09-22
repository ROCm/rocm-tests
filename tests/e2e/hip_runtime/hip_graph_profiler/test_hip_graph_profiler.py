# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HIP Graph + PyTorch Profiler stability test suite.

Validates graph capture, replay, and profiler trace integrity across
fusion toggles, high-frequency captures, multi-stream signal dependencies,
batch/backend variants, abort resilience, and end-to-end pipeline scenarios.
"""

from __future__ import annotations

import pathlib
import shlex

import pytest

_SRC_DIR = pathlib.Path(__file__).parent / "src"


def _resolve(target_executor, script_name: str) -> str:
    executor = next(iter(target_executor))
    if hasattr(executor, "upload_tree"):
        return executor.upload_tree(str(_SRC_DIR)) + "/" + script_name
    return str(_SRC_DIR / script_name)


def _run(target_executor, torch_python: str, ld_path: dict, script_name: str, timeout: int) -> None:
    script = _resolve(target_executor, script_name)
    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    python = shlex.quote(torch_python)
    result = target_executor.run(
        f"{python} -m pip install pytest -q && "
        f"env LD_LIBRARY_PATH={ld} {python} -m pytest {shlex.quote(script)} -v",
        timeout=timeout,
    )
    assert result.ok, (
        f"{script_name} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:500]}"
    )


@pytest.mark.ci.pr
@pytest.mark.runtime.medium
def test_graph_profiling(require_torch, torch_python: str, target_executor, ld_path: dict):
    """Graph replay completion and profiler trace integrity — Tier-1."""
    _run(target_executor, torch_python, ld_path, "graph_profiling.py", timeout=900)


@pytest.mark.runtime.soak
def test_fusion_toggles(require_torch, torch_python: str, target_executor, ld_path: dict):
    """BN/Conv/Linear fusion toggle matrix must complete all combinations without hang."""
    _run(target_executor, torch_python, ld_path, "fusion_toggles.py", timeout=600)


@pytest.mark.runtime.soak
def test_high_freq_capture(require_torch, torch_python: str, target_executor, ld_path: dict):
    """Repeated capture+release cycles must not hang or exhaust GPU resources."""
    _run(target_executor, torch_python, ld_path, "high_freq_capture.py", timeout=1800)


@pytest.mark.runtime.soak
def test_signal_dependency_stress(require_torch, torch_python: str, target_executor, ld_path: dict):
    """Inter-stream event signaling under profiling must not produce deadlocks."""
    _run(target_executor, torch_python, ld_path, "signal_dependency_stress.py", timeout=1200)


@pytest.mark.runtime.medium
def test_batch_backend_variants(require_torch, torch_python: str, target_executor, ld_path: dict):
    """Graph replay across batch sizes and attention backends must complete under profiling."""
    _run(target_executor, torch_python, ld_path, "batch_backend_variants.py", timeout=600)


@pytest.mark.runtime.medium
def test_capture_abort_watchdog(require_torch, torch_python: str, target_executor, ld_path: dict):
    """Deliberate capture abort during profiling must leave device healthy and recoverable."""
    _run(target_executor, torch_python, ld_path, "capture_abort_watchdog.py", timeout=600)


@pytest.mark.ci.pr
@pytest.mark.runtime.soak
def test_e2e_pipeline(require_torch, torch_python: str, target_executor, ld_path: dict):
    """Full capture-warmup-profile-replay pipeline at 1B-param scale — Tier-1."""
    _run(target_executor, torch_python, ld_path, "e2e_pipeline.py", timeout=1800)
