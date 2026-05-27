# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hip_invalid_codeobject_load.py -- HIP code object loading error path regression.

Validates the HIP runtime handles invalid code objects gracefully without crashes
or memory corruption. Three sub-tests:

* LoadNonexistent   — hipModuleLoad on a nonexistent file returns
                      hipErrorFileNotFound, not a segfault.
* RepeatedLoads     — 1000 consecutive failed hipModuleLoad calls followed by
                      hipDeviceReset complete without memory corruption or crash.
* ArchSpecificLoad  — A real .hsaco from hipBLASLt loads successfully on the
                      current GPU architecture (skipped if hipBLASLt absent).

Regression for SWDEV-508590 / TMS 1002344.

Binary compiled via CMake from:
    tests/e2e/hip_runtime/src/hip_invalid_codeobject_load_test.cpp

Markers auto-injected by CATEGORY_PROFILES (tests/e2e/hip_runtime):
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

This test exercises HIP driver error paths (no GPU compute), so e2e.stack from
the profile is technically loose — the test is a driver-level regression test.
runtime.fast is declared explicitly.
"""

import pytest


@pytest.mark.runtime.fast
@pytest.mark.parametrize("subtest", ["LoadNonexistent", "RepeatedLoads", "ArchSpecificLoad"])
def test_hip_invalid_codeobject_load(
    target_executor,
    ld_path: dict,
    hip_invalid_codeobject_load_binary: str,
    rock_dir: str,
    subtest: str,
):
    """Validate HIP graceful error handling for invalid code object operations.

    Args:
        target_executor:                   Location-transparent GPU executor.
        ld_path:                           LD_LIBRARY_PATH dict for ROCm libs.
        hip_invalid_codeobject_load_binary: Path to compiled binary.
        rock_dir:                          ROCm install root (for ROCM_PATH env).
        subtest:                           Sub-test name to run.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir}" f" {hip_invalid_codeobject_load_binary} -t {subtest}"
    )
    assert result.ok, (
        f"hip_invalid_codeobject_load subtest={subtest} failed"
        f" (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Verify the subtest actually ran and reported success via stdout.
    # ArchSpecificLoad legitimately prints [SKIP] and returns early when hipBLASLt is absent.
    if subtest == "ArchSpecificLoad":
        if "[SKIP]" not in result.stdout:
            assert "[PASS]" in result.stdout, (
                f"Expected '[PASS]' or '[SKIP]' in stdout for {subtest}:\n" f"stdout: {result.stdout[:2000]}"
            )
    else:
        assert "[PASS]" in result.stdout, (
            f"Expected '[PASS]' in stdout for {subtest}:\n" f"stdout: {result.stdout[:2000]}"
        )
    # The CHECK/CHECK_EQ macros in the binary write failure details to stderr.
    assert "FAIL:" not in result.stderr, f"Failure detail found in stderr for {subtest}:\n{result.stderr[:1000]}"
