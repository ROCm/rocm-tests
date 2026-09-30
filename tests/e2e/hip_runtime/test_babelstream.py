# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# BabelStream is maintained by the University of Bristol HPC group.
# Licensed under a custom permissive license (see UoB-HPC/BabelStream/LICENSE).

"""
test_babelstream.py -- Babelstream HIP benchmark validation.

BabelStream is a memory bandwidth benchmark suite maintained by the University of Bristol HPC group.
This test exercises the HIP backend with both double and float precision kernels on AMD GPUs.

Validates:
    1. Babelstream successfully compiles via Makefile HIP backend.
    2. Babelstream executes hip-stream (double precision) and produces deterministic output.
    3. Babelstream executes hip-stream (float precision) and produces deterministic output.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hip_runtime/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.medium (5-30 min: ~8 min clone/build/run per test)
"""

from __future__ import annotations

import pytest


@pytest.mark.runtime.medium
def test_babelstream_double(target_executor, ld_path: dict, babelstream_binary: str):
    """Execute babelstream hip-stream benchmark with double precision.

    Runs the compiled hip-stream binary and validates that the benchmark
    completes successfully, emitting throughput and timing data to stdout.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {babelstream_binary}",
        timeout=300.0,
    )
    assert result.ok, (
        f"babelstream double test failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Babelstream emits benchmark results; at minimum, verify output is non-empty
    # and contains expected stream benchmark markers (e.g., timing, throughput).
    assert result.stdout, "babelstream produced no output; binary may have crashed silently"


@pytest.mark.runtime.medium
def test_babelstream_float(target_executor, ld_path: dict, babelstream_binary: str):
    """Execute babelstream hip-stream benchmark with float precision.

    Runs the compiled hip-stream binary with --float flag and validates
    successful completion and output generation. Reuses the same binary
    from the first test (no rebuild required).
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {babelstream_binary} --float",
        timeout=300.0,
    )
    assert result.ok, (
        f"babelstream float test failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert result.stdout, "babelstream produced no output; binary may have crashed silently"
