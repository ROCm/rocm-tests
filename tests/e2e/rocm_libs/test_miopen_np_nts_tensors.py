# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_miopen_np_nts_tensors.py -- MIOpen non-packed / non-trivially-strided tensor convolution.

Exercises MIOpen's 3-D convolution correctness when the input tensor uses
non-trivial (non-packed) strides.  The binary runs three convolution variants:

1. Non-packed buffer with non-packed descriptor (the target NP-NTS path).
2. Non-packed buffer with packed descriptor (intentional mismatch, for reference).
3. Packed buffer with packed descriptor (the ground-truth reference).

The binary exits 0 and prints "W00t!" when the NP-NTS convolution matches the
packed reference; it exits non-zero and prints "FAIL" otherwise.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/rocm_libs/:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.medium
"""

import pytest


@pytest.mark.runtime.medium
def test_miopen_np_nts_tensors(
    target_executor,
    ld_path: dict,
    miopen_np_nts_tensors_binary: str,
):
    """Validate MIOpen correctness for non-packed, non-trivially-strided tensor convolution."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {miopen_np_nts_tensors_binary}",
        timeout=600.0,
    )
    assert result.ok, (
        f"miopen_np_nts_tensors failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:800]}"
    )
    assert "W00t!" in result.stdout, (
        f"Expected 'W00t!' sentinel in stdout but it was not found.\n" f"stdout: {result.stdout[:3000]}"
    )
    assert (
        "this is BAD" not in result.stdout
    ), f"Test indicated failure with 'this is BAD' in output:\n{result.stdout[:3000]}"
    assert (
        "This is a BAD thing" not in result.stdout
    ), f"Test indicated failure with 'This is a BAD thing' in output:\n{result.stdout[:3000]}"
