# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CK tile FMHA dropout correctness tests for MI300X/MI350X (gfx942/gfx950)."""

import pytest

_FWD_BINARY = "bin/tile_example_fmha_fwd"
_BWD_BINARY = "bin/tile_example_fmha_bwd"

_DROP_ARGS = "-b=1 -h=8 -s=4096 -d=64 -drop_seed=10 -drop_offset=1234"


@pytest.mark.runtime.medium
def test_ck_fmha_fwd_drop_prefs_1(
    target_executor,
    ld_path: dict,
    ck_fmha_build: str,
):
    """Verify CK tile FMHA forward dropout with drop_prefs=1."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = f"{ck_fmha_build}/{_FWD_BINARY}"
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} {_DROP_ARGS} -drop_prefs=1",
        timeout=300,
    )
    assert result.ok, (
        f"fmha_fwd drop_prefs=1 failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "valid:y" in result.stdout, f"fmha_fwd drop_prefs=1 did not report valid:y:\n{result.stdout[:2000]}"


@pytest.mark.runtime.medium
def test_ck_fmha_fwd_drop_prefs_0(
    target_executor,
    ld_path: dict,
    ck_fmha_build: str,
):
    """Verify CK tile FMHA forward dropout with drop_prefs=0."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = f"{ck_fmha_build}/{_FWD_BINARY}"
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} {_DROP_ARGS} -drop_prefs=0",
        timeout=300,
    )
    assert result.ok, (
        f"fmha_fwd drop_prefs=0 failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "valid:y" in result.stdout, f"fmha_fwd drop_prefs=0 did not report valid:y:\n{result.stdout[:2000]}"


@pytest.mark.runtime.medium
def test_ck_fmha_bwd_drop_prefs_1(
    target_executor,
    ld_path: dict,
    ck_fmha_build: str,
):
    """Verify CK tile FMHA backward dropout with drop_prefs=1."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = f"{ck_fmha_build}/{_BWD_BINARY}"
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} {_DROP_ARGS} -drop_prefs=1",
        timeout=300,
    )
    assert result.ok, (
        f"fmha_bwd drop_prefs=1 failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "valid:y" in result.stdout, f"fmha_bwd drop_prefs=1 did not report valid:y:\n{result.stdout[:2000]}"


@pytest.mark.runtime.medium
def test_ck_fmha_bwd_drop_prefs_0(
    target_executor,
    ld_path: dict,
    ck_fmha_build: str,
):
    """Verify CK tile FMHA backward dropout with drop_prefs=0."""
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = f"{ck_fmha_build}/{_BWD_BINARY}"
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} {_DROP_ARGS} -drop_prefs=0",
        timeout=300,
    )
    assert result.ok, (
        f"fmha_bwd drop_prefs=0 failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "valid:y" in result.stdout, f"fmha_bwd drop_prefs=0 did not report valid:y:\n{result.stdout[:2000]}"
