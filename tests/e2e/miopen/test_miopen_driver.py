# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_miopen_driver.py -- MIOpen convolution forward and backward pass validation.

Exercises the pre-installed MIOpenDriver binary with representative forward,
backward-data, and backward-weights conv shapes and asserts GPU verification passes.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Conv command groups
# ---------------------------------------------------------------------------

_FORWARD_CONV_1 = [
    "conv -n 16 -c 16 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
    "conv -n 16 -c 64 -H 34 -W 34 -k 64 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
    "conv -n 32 -c 32 -H 17 -W 17 -k 32 -y 1 -x 7 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
    "conv -n 64 -c 256 -H 34 -W 34 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
    "conv -n 128 -c 128 -H 35 -W 35 -k 128 -y 3 -x 3 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F 1 -t 1",
    "conv -n 128 -c 48 -H 7 -W 7 -k 128 -y 5 -x 5 -p 2 -q 2 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
]

_FORWARD_CONV_2 = [
    "conv -n 16 -c 16 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
    "conv -n 128 -c 256 -H 28 -W 28 -k 128 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
    "conv -n 64 -c 1536 -H 8 -W 8 -k 256 -y 1 -x 1 -p 0 -q 3 -u 1 -v 1 -l 1 -j 1 -F 1 -t 1",
]

_BACKWARD_DATA_CONV_1 = [
    "conv -n 64 -c 64 -H 28 -W 28 -k 16 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2 -t 1",
    "conv -n 64 -c 64 -H 56 -W 56 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2 -t 1",
    "conv -n 16 -c 128 -H 36 -W 36 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2 -t 1",
    "conv -n 32 -c 128 -H 34 -W 34 -k 64 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2 -t 1",
    "conv -n 128 -c 128 -H 35 -W 35 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 2 -t 1",
]

_BACKWARD_WRW_CONV_2 = [
    "conv -n 64 -c 64 -H 28 -W 28 -k 32 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 -t 1",
    "conv -n 32 -c 128 -H 34 -W 34 -k 64 -y 3 -x 3 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 -t 1",
    "conv -n 128 -c 128 -H 35 -W 35 -k 128 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F 4 -t 1",
    "conv -n 128 -c 256 -H 56 -W 56 -k 64 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 4 -t 1",
]

_PASS_SENTINEL = "Verifies OK on GPU reference"


def _run_conv_group(
    target_executor,
    ld: str,
    driver: str,
    cmds: list[str],
    group_name: str,
    extra_flags: str = "",
) -> None:
    """Run every conv command in *cmds* and assert each verifies on GPU."""
    for args in cmds:
        cmd = f"env LD_LIBRARY_PATH={ld} {driver} {args}"
        if extra_flags:
            cmd = f"{cmd} {extra_flags}"
        result = target_executor.run(cmd)
        assert result.ok, (
            f"{group_name} failed (exit={result.exit_code}) for: {args}\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
        assert _PASS_SENTINEL in result.stdout, (
            f"{group_name} did not verify on GPU for: {args}\n" f"stdout: {result.stdout[:2000]}"
        )


# ---------------------------------------------------------------------------
# Test functions — one per conv group
# ---------------------------------------------------------------------------


@pytest.mark.runtime.fast
def test_miopen_forward_conv_1(target_executor, ld_path: dict, rock_dir: str, gpu_arch: str | None) -> None:
    """Validate MIOpenDriver forward conv group 1 (6 shapes, -F 1)."""
    driver = f"{rock_dir}/bin/MIOpenDriver"
    ld = ld_path["LD_LIBRARY_PATH"]
    _run_conv_group(target_executor, ld, driver, _FORWARD_CONV_1, "Forward_Conv_1")


@pytest.mark.runtime.fast
def test_miopen_forward_conv_2(target_executor, ld_path: dict, rock_dir: str, gpu_arch: str | None) -> None:
    """Validate MIOpenDriver forward conv group 2 (3 shapes, -F 1)."""
    driver = f"{rock_dir}/bin/MIOpenDriver"
    ld = ld_path["LD_LIBRARY_PATH"]
    _run_conv_group(target_executor, ld, driver, _FORWARD_CONV_2, "Forward_Conv_2")


@pytest.mark.runtime.fast
def test_miopen_backward_data_conv(target_executor, ld_path: dict, rock_dir: str, gpu_arch: str | None) -> None:
    """Validate MIOpenDriver backward-data conv group (5 shapes, -F 2)."""
    driver = f"{rock_dir}/bin/MIOpenDriver"
    ld = ld_path["LD_LIBRARY_PATH"]
    # gfx906 requires verification disabled for backward passes
    extra = "-V 0" if gpu_arch and "gfx906" in gpu_arch else ""
    _run_conv_group(target_executor, ld, driver, _BACKWARD_DATA_CONV_1, "Backward_Conv_1", extra_flags=extra)


@pytest.mark.runtime.fast
def test_miopen_backward_wrw_conv(target_executor, ld_path: dict, rock_dir: str, gpu_arch: str | None) -> None:
    """Validate MIOpenDriver backward weight-gradient conv group (4 shapes, -F 4)."""
    driver = f"{rock_dir}/bin/MIOpenDriver"
    ld = ld_path["LD_LIBRARY_PATH"]
    # gfx906 requires verification disabled for backward passes
    extra = "-V 0" if gpu_arch and "gfx906" in gpu_arch else ""
    _run_conv_group(target_executor, ld, driver, _BACKWARD_WRW_CONV_2, "Backward_Conv_2", extra_flags=extra)
