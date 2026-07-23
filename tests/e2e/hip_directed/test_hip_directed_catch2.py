# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run directed HIP catch2 unit tests from the public ROCm/rocm-systems hip-tests suite.

Builds the upstream ``projects/hip-tests/catch`` suite for ``HIP_PLATFORM=amd`` and
runs a curated set of directed ``Unit_hip*`` tests via ``ctest -R``. Each directed
test is its own parametrized row so a failure isolates to one API area.
"""

import pytest

# (id, ctest -R name-regex) for the directed unit tests.
_DIRECTED = [
    ("hipSetValidDevices", r"^Unit_hipSetValidDevices"),
    (
        "hipGetDriverEntryPoint",
        r"^Unit_hipGetDriverEntryPoint_(Positive|Negative|spt_Positive|spt_Negative)",
    ),
    ("hipStreamGetId", r"^Unit_hipStreamGetId"),
    ("hipStreamSetAttribute", r"^Unit_hipStreamSetAttribute"),
    ("hipStreamGetAttribute", r"^Unit_hipStreamGetAttribute"),
    ("hipMemPrefetchAsync_v2", r"^Unit_hipMemPrefetchAsync_v2"),
    ("hipMemAdvise_v2", r"^Unit_hipMemAdvise_v2_"),
    ("hipModuleGetFunctionCount", r"^Unit_hipModuleGetFunctionCount"),
    # hipModuleLoadFatBinary omitted: its positive test loads build-generated
    # .code fatbins produced by add_custom_target(... ALL ...) steps that are not
    # dependencies of the scoped ModuleTest executable build.
    ("hipMemcpy3DBatchAsync", r"^Unit_hipMemcpy3DBatchAsync"),
    ("hipMemcpy3DPeer", r"^Unit_hipMemcpy3DPeer"),
    ("hipMemcpyBatchAsync", r"^Unit_hipMemcpyBatchAsync"),
    ("hipMemsetD2D", r"^Unit_hipMemsetD2D"),
]


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("name", "pattern"), _DIRECTED, ids=[d[0] for d in _DIRECTED])
def test_hip_directed_catch2(
    target_executor, ld_path: dict, rock_dir: str, hip_catch_build_dir: str, name: str, pattern: str
):
    """Run one directed HIP catch2 unit test and assert it is green."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
        f"ctest --test-dir {hip_catch_build_dir} --output-on-failure -R {pattern!r}",
        timeout=3600,
    )
    if "No tests were found" in (result.stdout + result.stderr):
        pytest.skip(f"directed test {name!r}: no matching catch2 tests in this build")
    passed = result.ok and "100% tests passed" in result.stdout
    assert passed, f"hip directed {name!r} not green (exit={result.exit_code}):\n{result.stdout[-1500:]}"
