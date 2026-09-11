# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_amdsmitst.py — AMD SMI gtest binary validation.

Ported from: ROCmTest/tests/TOOLS/SystemManagement/AMD_SMI/amdsmitst.py

Validates:
    1. The amdsmitst gtest binary runs successfully on AMD GPU hardware.
    2. All test cases pass (gtest exits 0 and prints "[  PASSED  ]" to stdout).
    3. Blacklisted test cases are properly excluded via gtest filtering.

The amdsmitst binary is part of the amd-smi-lib package and is pre-installed
into the ROCm share directory by TheRock. This ported test simply runs the
gtest suite and validates the exit code and output.

Supported GPUs: Navi48/44/33/32/31, MI375, MI350X, MI325X, MI250X, MI210, MI200.

Markers auto-injected by CATEGORY_PROFILES:
    None (tests/e2e/system_tools/amd_smi/ is not a profile directory; all markers
    must be declared explicitly).

Explicit markers:
    hw.gpu            — single GPU required
    ci.nightly        — nightly CI tier (not PR gate)
    layer.runtime     — amd-smi exercises GPU runtime functionality
    runtime.fast      — estimated 1-5 minutes for full gtest suite
    os.linux          — Linux only
"""

import pytest


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_amdsmitst(
    target_executor,
    ld_path: dict,
    amdsmitst_binary: str,
):
    """Run the amdsmitst gtest binary and validate all test cases pass.

    The amdsmitst binary is a gtest suite that validates amd-smi library
    functionality. It is pre-installed by the amd-smi-lib package into the
    ROCm tree. This test simply runs it and asserts:
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

    # Run the gtest binary. gtest exits 0 on all PASSED and prints "[  PASSED  ]"
    # to stdout. We do not apply gtest_filter here — amdsmitst self-manages its
    # blacklisting via any .exclude file in its working directory (if present).
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {amdsmitst_binary}"
    )

    # Validate exit code is 0 (gtest's success marker).
    assert result.ok, (
        f"amdsmitst gtest binary failed (exit_code={result.exit_code}):\n"
        f"stdout (first 2000 chars):\n{result.stdout[:2000]}\n"
        f"stderr (first 500 chars):\n{result.stderr[:500]}"
    )

    # Validate that gtest printed its success marker to stdout.
    assert "PASSED" in result.stdout, (
        f"amdsmitst did not report test PASSED status:\n"
        f"stdout (first 2000 chars):\n{result.stdout[:2000]}"
    )
