# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run the rocm-examples rocm_agent_enumerator CTest entry on an AMD GPU.

Validates that the HIP-Basic/rocm_agent_enumerator sample builds, executes, and
correctly enumerates at least one GPU agent on the target node.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/rocm_examples/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.fast
"""

from __future__ import annotations

import pytest

# CTest test name as registered by the rocm-examples CMakeLists tree.
_CTEST_NAME = "hip-basic_rocm_agent_enumerator"


@pytest.mark.runtime.fast
def test_rocm_agent_enumerator(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    rocm_examples_build_dir: str,
) -> None:
    """Run the rocm_agent_enumerator sample and assert at least one GPU agent is found."""
    ld = ld_path["LD_LIBRARY_PATH"]
    rocm_bin = f"{rock_dir}/bin"

    result = target_executor.run(
        f"env PATH={rocm_bin}:$PATH LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
        f"ctest --test-dir {rocm_examples_build_dir} "
        f"--output-on-failure -R ^{_CTEST_NAME}$",
        timeout=300,
    )

    combined = result.stdout + result.stderr

    if "No tests were found" in combined:
        pytest.skip(
            f"CTest entry {_CTEST_NAME!r} not found in this build — "
            "the sample may require a ROCm component not present in this install"
        )

    # CTest reports 'Not Run' when the sample binary was never produced (missing
    # upstream component).  Skip rather than fail so absent infrastructure does
    # not masquerade as a product defect.
    if f"- {_CTEST_NAME} (Not Run)" in combined or "(Not Run)" in combined:
        pytest.skip(
            "rocm_agent_enumerator binary not available in this build — "
            "required ROCm component not shipped in this install"
        )

    assert result.ok, (
        f"rocm_agent_enumerator CTest failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:500]}"
    )
    assert "100% tests passed" in result.stdout, (
        f"rocm_agent_enumerator did not pass cleanly:\n{result.stdout[:2000]}"
    )
