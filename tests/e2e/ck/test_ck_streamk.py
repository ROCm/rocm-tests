# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CK tile stream-k GEMM correctness tests for MI300X/MI350X (gfx942/gfx950)."""

import pytest

_BINARY = "bin/tile_example_streamk_gemm_basic"

_FP16_ARGS = "-m=3840 -n=4096 -k=4096 -prec=fp16 -v=2 -a_layout=R -b_layout=C -c_layout=R"
_FP8_ARGS = "-m=3840 -n=4096 -k=4096 -prec=fp8 -v=2 -a_layout=R -b_layout=C -c_layout=R"


@pytest.mark.runtime.medium
def test_ck_streamk_gemm_fp16(
    target_executor,
    ld_path: dict,
    ck_streamk_build: str,
):
    """Verify CK tile stream-k GEMM correctness in fp16 precision."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = f"{ck_streamk_build}/{_BINARY}"
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} {_FP16_ARGS}",
        timeout=300,
    )
    assert result.ok, (
        f"ck_streamk fp16 failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "correct" in result.stdout, f"ck_streamk fp16 did not report correct result:\n{result.stdout[:2000]}"


@pytest.mark.runtime.medium
def test_ck_streamk_gemm_fp8(
    target_executor,
    ld_path: dict,
    ck_streamk_build: str,
):
    """Verify CK tile stream-k GEMM correctness in fp8 precision."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = f"{ck_streamk_build}/{_BINARY}"
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} {_FP8_ARGS}",
        timeout=300,
    )
    assert result.ok, (
        f"ck_streamk fp8 failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "correct" in result.stdout, f"ck_streamk fp8 did not report correct result:\n{result.stdout[:2000]}"
