# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run HIP apps with hipcc.

This keeps the legacy ``hipcc_build_apps`` intent: each sample must compile,
run, and print its expected success marker.  The old Windows Event Viewer scan is
orchestration-specific and is not part of the Linux E2E port.
"""

import re

import pytest

# One entry per legacy app: build fixture name plus expected stdout marker.
_APPS = [
    ("vectoradd_hip_binary", r"PASSED!"),
    ("openmp_helloworld_binary", r"PASSED!"),
    ("matrixmultiplication_binary", r"^\s*Output\s*$"),
]


@pytest.mark.runtime.fast
@pytest.mark.parametrize(("binary_fixture", "marker"), _APPS)
def test_hip_app_build_using_hipcc(
    request: pytest.FixtureRequest,
    target_executor,
    ld_path: dict,
    binary_fixture: str,
    marker: str,
):
    """Resolve a sample build fixture, run the binary, and verify its marker."""
    # Resolving the fixture triggers the hipcc build; a build failure surfaces
    # as a fixture ERROR (mirrors the legacy per-app build pass/fail step).
    binary = request.getfixturevalue(binary_fixture)

    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary}")
    assert result.ok, (
        f"{binary_fixture} run failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert re.search(
        marker, result.stdout, re.MULTILINE
    ), f"{binary_fixture} ran but expected marker {marker!r} not found:\n{result.stdout[:2000]}"
