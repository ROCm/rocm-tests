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


# Binaries the "tools" bucket (rocgdb/rocprof/rocprofv3 samples) invokes. They
# ship only when the ROCm install carries the debugger + profiler artifact,
# which the nightly install (COMMON_FLAGS in e2e-tests.yml) does not pull today.
_TOOLS_BINARIES = ("rocgdb", "rocprof", "rocprofv3")


@pytest.mark.runtime.medium
@pytest.mark.parametrize(("category", "pattern"), _CATEGORIES, ids=[c[0] for c in _CATEGORIES])
def test_rocm_examples(
    target_executor, ld_path: dict, rock_dir: str, rocm_examples_build_dir: str, category: str, pattern: str
):
    """Run one rocm-examples CTest category and assert it is fully green."""
    ld = ld_path["LD_LIBRARY_PATH"]
    rocm_bin = f"{rock_dir}/bin"

    # Gate the env-dependent "tools" bucket: run it only when the debugger/
    # profiler binaries are actually present in the ROCm artifact, otherwise skip
    # with a message naming the artifact to add -- so it is a documented gate,
    # not a permanent red. Enable it by installing the rocgdb + rocprofiler
    # component (extend COMMON_FLAGS in .github/workflows/e2e-tests.yml).
    if category == "tools":
        probe = target_executor.run(
            f"export PATH={rocm_bin}:$PATH; "
            + "; ".join(f"command -v {b} >/dev/null 2>&1 || exit 1" for b in _TOOLS_BINARIES),
            timeout=60,
        )
        if not probe.ok:
            pytest.skip(
                "rocm-examples 'tools' bucket needs "
                + "/".join(_TOOLS_BINARIES)
                + " from the ROCm debugger+profiler artifact (not in this install); add the "
                "rocgdb/rocprofiler component to COMMON_FLAGS in .github/workflows/e2e-tests.yml "
                "to enable it."
            )

    result = target_executor.run(
        f"env PATH={rocm_bin}:$PATH LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
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
