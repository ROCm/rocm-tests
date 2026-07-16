# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hecbench.py -- HeCBench HIP benchmark suite compile + run validation.

Validates:
    For each HeCBench benchmark ``<name>`` that ships a ``src/<name>-hip`` folder,
    this test:
        1. Compiles the benchmark's Makefile (``make clean`` + ``make``) — the
           Makefiles invoke ``hipcc`` directly, so the ROCm toolchain must be on
           PATH (supplied from ``--rock-dir``).
        2. Runs it (``make run``) on an AMD GPU.
        3. Parses stdout with the benchmark's timing/throughput regex from
           ``subset.json`` and requires a nonzero numeric total.

    A benchmark passes only when BOTH the compile succeeds AND the regex extracts
    a nonzero metric

    Benchmarks whose ``subset.json`` regex is the empty string carry no metric to
    validate; for those the test asserts only that compile + run exit cleanly

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/compiler/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    - test_hecbench_smoke:      runtime.medium (curated subset, keeps ci.nightly)
    - test_hecbench_full_suite: ci.weekly + runtime.soak (overrides ci.nightly)

Prerequisites:
    - ``--rock-dir`` / ``ROCK_DIR`` pointing to a TheRock/ROCm install providing
      ``bin/hipcc`` and ``lib/``.
    - Network access to clone https://github.com/zjin-lcf/HeCBench (see conftest).
    - Real AMD GPU hardware.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Benchmark catalog — loaded once at import time so it can drive parametrization.
# subset.json lives next to this file
# Format: {"<benchmark-name>": ["<regex-with-one-capture-group>"]}.
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).parent
_SUBSET_PATH = _HERE / "subset.json"

with _SUBSET_PATH.open(encoding="utf-8") as _f:
    _BENCHMARKS: dict[str, list[str]] = json.load(_f)

# Full weekly/soak catalog: every benchmark declared in subset.json.
_ALL_BENCHMARKS: list[str] = sorted(_BENCHMARKS)

# Curated nightly smoke subset: small, self-contained benchmarks with a clear,
# non-empty metric regex.  Kept intentionally short so the nightly gate stays
# within the runtime.medium budget.
_SMOKE_BENCHMARKS: list[str] = [
    "accuracy",
    "adam",
    "bitonic-sort",
    "black-scholes",
    "xsbench",
]

# Compile with plain hipcc, or patch the Makefile for amdgcnspirv codegen.
_MODES: list[str] = ["hip", "spirv"]

# Generous per-benchmark timeouts (seconds).  Some HeCBench Makefiles pull large
# templates; runs iterate kernels many times.
_COMPILE_TIMEOUT_S = 1800.0
_RUN_TIMEOUT_S = 900.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_total(regex: str, output: str) -> float:
    """Sum every numeric value the *regex* captures from *output*.

    Args:
        regex:  Per-benchmark capture regex from subset.json.
        output: Combined stdout/stderr from ``make run``.

    Returns:
        Sum of all parsed numeric captures (``0.0`` when nothing matched).
    """
    total = 0.0
    for match in re.findall(regex, output):
        candidate = next((g for g in match if g.strip()), "") if isinstance(match, tuple) else match
        if candidate.strip():
            try:
                total += float(candidate)
            except ValueError:
                # Non-numeric capture (e.g. a checksum string)
                continue
    return total


def _run_benchmark(
    *,
    benchmark: str,
    mode: str,
    hecbench_repo: pathlib.Path,
    target_executor,
    rock_dir: str,
    ld_path: dict,
) -> None:
    """Compile and run one HeCBench benchmark.

    Skips (rather than fails) when the ``<benchmark>-hip`` folder is absent from
    the checkout

    Args:
        benchmark:       Benchmark key from subset.json (e.g. ``"accuracy"``).
        mode:            ``"hip"`` (plain hipcc) or ``"spirv"`` (amdgcnspirv).
        hecbench_repo:   Session-scoped checkout path (``hecbench_repo`` fixture).
        target_executor: GPU executor group (compile + run on the same node).
        rock_dir:        TheRock/ROCm install path (provides ``bin/hipcc``).
        ld_path:         ``LD_LIBRARY_PATH`` env dict for TheRock-linked binaries.
    """
    hip_folder = f"{benchmark}-hip"
    folder = os.path.join(str(hecbench_repo), "src", hip_folder)
    q_folder = shlex.quote(folder)

    # Missing benchmark folder
    if not target_executor.run(f"test -d {q_folder}").ok:
        pytest.skip(f"benchmark folder not present in checkout: src/{hip_folder}")

    # hipcc from --rock-dir must be on PATH for the Makefile; LD_LIBRARY_PATH lets
    # the built binary find TheRock libs at run time.  env applies to the command
    # immediately following it; $PATH is expanded by the executor's shell.
    env = f"env PATH={shlex.quote(rock_dir)}/bin:$PATH LD_LIBRARY_PATH={shlex.quote(ld_path['LD_LIBRARY_PATH'])}"

    # Always start from a pristine Makefile so a prior SPIR-V patch on the shared
    # session checkout never leaks into a later run (either mode, any order).
    target_executor.run(f"cd {q_folder} && git checkout -- Makefile 2>/dev/null || true")

    if mode == "spirv":
        patch = target_executor.run(f"cd {q_folder} && sed -i 's/hipcc/hipcc --offload-arch=amdgcnspirv/g' Makefile")
        assert patch.ok, f"SPIR-V Makefile patch failed for {hip_folder}:\n{patch.stderr[:500]}"

    # Compile: tolerate `make clean` failure assert on `make`.
    compile_res = target_executor.run(
        f"cd {q_folder} && {env} make clean >/dev/null 2>&1 ; {env} make",
        timeout=_COMPILE_TIMEOUT_S,
    )
    assert compile_res.ok, (
        f"{hip_folder} ({mode}) failed to compile (exit={compile_res.exit_code}):\n"
        f"stdout: {compile_res.stdout[:2000]}\nstderr: {compile_res.stderr[:1000]}"
    )

    # Run on the GPU.
    run_res = target_executor.run(
        f"cd {q_folder} && {env} make run",
        timeout=_RUN_TIMEOUT_S,
    )
    assert run_res.ok, (
        f"{hip_folder} ({mode}) run failed (exit={run_res.exit_code}):\n"
        f"stdout: {run_res.stdout[:2000]}\nstderr: {run_res.stderr[:1000]}"
    )

    regex = _BENCHMARKS[benchmark][0] if _BENCHMARKS[benchmark] else ""

    # Empty regex -> no metric to validate; compile + run success is the signal.
    if not regex.strip():
        logger.info("%s (%s): no metric regex in subset.json; validated on exit code only.", hip_folder, mode)
        return

    total = _extract_total(regex, f"{run_res.stdout}\n{run_res.stderr}")
    report_metric(f"HECBENCH_{benchmark}_{mode}", total)
    assert total, (
        f"{hip_folder} ({mode}) ran but metric regex matched no nonzero value.\n"
        f"regex: {regex!r}\nstdout: {run_res.stdout[:2000]}"
    )


# ---------------------------------------------------------------------------
# Nightly smoke: curated subset, keeps the profile-injected ci.nightly.
# ---------------------------------------------------------------------------


@pytest.mark.runtime.medium
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("benchmark", _SMOKE_BENCHMARKS)
def test_hecbench_smoke(
    benchmark: str,
    mode: str,
    hecbench_repo: pathlib.Path,
    target_executor,
    rock_dir: str,
    ld_path: dict,
):
    """Compile + run a curated HeCBench benchmark (nightly signal)."""
    _run_benchmark(
        benchmark=benchmark,
        mode=mode,
        hecbench_repo=hecbench_repo,
        target_executor=target_executor,
        rock_dir=rock_dir,
        ld_path=ld_path,
    )


@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("benchmark", _ALL_BENCHMARKS)
def test_hecbench_full_suite(
    benchmark: str,
    mode: str,
    hecbench_repo: pathlib.Path,
    target_executor,
    rock_dir: str,
    ld_path: dict,
):
    """Compile + run every HeCBench benchmark in both hip and spirv modes."""
    _run_benchmark(
        benchmark=benchmark,
        mode=mode,
        hecbench_repo=hecbench_repo,
        target_executor=target_executor,
        rock_dir=rock_dir,
        ld_path=ld_path,
    )
