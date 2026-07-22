# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run the public ROCm/rocm-examples CTest suite, split by category.

The full CTest suite is bucketed into a few stable categories (by test-name
prefix) and each runs as its own parametrized test, so a failure isolates to an
area (e.g. env-dependent tools) instead of reddening the whole suite. No
exclusions -- environment-dependent samples are left to fail visibly.
"""

import pytest

# (category id, ctest -R name-regex). Buckets map to the rocm-examples areas by
# test-name prefix. "tools" (rocgdb/rocprof/rocprofv3) needs debugger/profiler
# environments and typically fails on a plain runner.
_CATEGORIES = [
    ("applications", r"^applications([_-]|$)"),
    ("hip_basic", r"^hip([_-]|$)"),
    (
        "libraries",
        r"^(rocblas|rocsparse|rocsolver|rocfft|rocrand|rocprim|rocthrust|rocwmma|rocalution|"
        r"rocjpeg|rocdecode|rocprofiler|hipblas|hipblaslt|hipfft|hipsolver|hipsparse|"
        r"hipsparselt|hiprand|hipcub|hipdnn|hiptensor|rccl|composable)([_-]|$)",
    ),
    ("tutorials", r"^(reduction|programming)([_-]|$)"),
    ("tools", r"^(rocgdb|rocprofv3|rocprof)-"),
]


def _ctest_summary(out: str) -> str:
    """Trim CTest output to the summary + failed-test lines (avoids dumping the whole log)."""
    keep = [
        line
        for line in out.splitlines()
        if (
            "tests passed" in line
            or "tests failed" in line
            or "***Failed" in line
            or "(Failed)" in line
            or line.strip().startswith(("The following tests FAILED", "Errors while running"))
        )
    ]
    return "\n".join(keep[-40:]) or out[-1500:]


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("category", "pattern"), _CATEGORIES, ids=[c[0] for c in _CATEGORIES])
def test_rocm_examples(
    target_executor, ld_path: dict, rock_dir: str, rocm_examples_build_dir: str, category: str, pattern: str
):
    """Run one rocm-examples CTest category and assert it is fully green."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
        f"ctest --test-dir {rocm_examples_build_dir} --output-on-failure -R {pattern!r}",
        timeout=7200,
    )
    if "No tests were found" in (result.stdout + result.stderr):
        pytest.skip(f"rocm-examples category {category!r}: no matching tests in this build")
    passed = result.ok and "100% tests passed" in result.stdout
    assert passed, (
        f"rocm-examples category {category!r} CTest not fully green (exit={result.exit_code}):\n"
        f"{_ctest_summary(result.stdout)}"
    )
