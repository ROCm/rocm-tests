# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_shape_boundary.py -- hipBLASLt integer overflow shape boundary validation.

Validates that hipBLASLt correctly handles nn.Linear calls at token-count
boundaries where 16-bit integer overflow may occur (32767/32768/65535/65536).
Tests both single-batch and multi-batch cases. One test (batch=16, tokens=32768)
is expected to raise a warning; all others should complete without errors.

Source script:
    tests/e2e/hipblaslt/src/hipblaslt_shape_boundary.py

Runtime: < 2 minutes (8 parametrized forward passes, no training).

This test exercises the PyTorch → hipBLASLt integration path for edge-case
input shapes (integer overflow boundary condition).

Markers auto-injected by CATEGORY_PROFILES (tests/e2e/hipblaslt):
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

runtime.fast is declared explicitly.
"""

import pathlib

import pytest

_SRC = pathlib.Path(__file__).parent / "src" / "hipblaslt_shape_boundary.py"


@pytest.mark.runtime.fast
def test_hipblaslt_shape_boundary(
    target_executor,
    ld_path: dict,
):
    """Run hipBLASLt nn.Linear shape boundary test (8 cases) via subprocess.

    Args:
        target_executor: Location-transparent GPU executor.
        ld_path:         LD_LIBRARY_PATH dict for ROCm libs.
    """
    try:
        import torch as _torch  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            f"PyTorch (ROCm build) not installed — {exc}.\n"
            "Install ROCm-enabled PyTorch wheels:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/rocm6.x\n"
            "See https://pytorch.org/get-started/locally/ for the correct ROCm version."
        )
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} python3 {_SRC}")
    assert result.ok, (
        f"hipblaslt_shape_boundary failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "Result: PASSED" in result.stdout, f"Expected 'Result: PASSED' in stdout:\n{result.stdout[:2000]}"
