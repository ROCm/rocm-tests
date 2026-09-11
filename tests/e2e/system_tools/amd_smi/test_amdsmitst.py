# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmitst.py — AMD SMI gtest binary validation.

Validates:
    1. The amdsmitst gtest binary runs successfully on AMD GPU hardware.
    2. All test cases pass (gtest exits 0 and prints "[  PASSED  ]" to stdout).
    3. Blacklisted test cases are properly excluded via gtest filtering.

The amdsmitst binary is part of the amd-smi-lib package and is pre-installed
into the ROCm share directory by TheRock. This ported test simply runs the
gtest suite and validates the exit code and output.

"""

import pytest
import logging

logger = logging.getLogger(__name__)

@pytest.mark.runtime.fast
def test_amdsmitst(
    target_executor,
    ld_path: dict,
    amdsmitst_binary: str,
):
    """Run the amdsmitst gtest binary with ASIC blacklist exclusions.

    The amdsmitst binary is a gtest suite that validates amd-smi library
    functionality. It is pre-installed by the amd-smi-lib package into the
    ROCm tree. This test sources amdsmitst.exclude to apply ASIC-specific
    test exclusions via gtest filtering, then asserts:
        1. Exit code is 0 (all gtest cases pass)
        2. "PASSED" appears in stdout (gtest standard output marker)

    Args:
        target_executor: NodeExecutorGroup — location-transparent GPU executor.
        ld_path: Dict with "LD_LIBRARY_PATH" key for TheRock-linked libraries.
        amdsmitst_binary: Absolute path to the amdsmitst gtest binary from conftest.

    Raises:
        AssertionError: When gtest exits non-zero or does not print PASSED.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    binary_dir = f"$(dirname {amdsmitst_binary})"

    # Source amdsmitst.exclude to load BLACKLIST_ALL_ASICS for this ASIC.
    # Apply gtest filter to exclude blacklisted tests.
    # Use bash explicitly since amdsmitst.exclude contains bash-only syntax (declare, arrays).
    cmd = (
        f"bash -c 'cd {binary_dir} && "
        f". ./amdsmitst.exclude && "
        f"env LD_LIBRARY_PATH={ld} "
        f"{amdsmitst_binary} "
        f'--gtest_filter=\"-$(echo ${{BLACKLIST_ALL_ASICS}})\"'
        f"'"
    )

    result = target_executor.run(cmd)
    logger.info(f"Test result : {result.stdout}")

    # Validate exit code is 0 (gtest's success marker).
    assert result.ok, (
        f"amdsmitst gtest binary failed (exit_code={result.exit_code}):\n"
        f"stdout (first 2000 chars):\n{result.stdout[:2000]}\n"
        f"stderr (first 500 chars):\n{result.stderr[:500]}"
    )

    # Validate that gtest printed its success marker to stdout.
    assert "PASSED" in result.stdout, (
        f"amdsmitst did not report test PASSED status:\n" f"stdout (first 2000 chars):\n{result.stdout[:2000]}"
    )
