# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build and run the public ROCm/rocm-examples CTest suite, split by category.

The full CTest suite is bucketed into a few stable categories (by test-name
prefix) and each runs as its own parametrized test, so a failure isolates to an
area (e.g. env-dependent tools) instead of reddening the whole suite.

A sample whose binary was never built -- because it needs a ROCm component the
install does not ship (e.g. Applications/monte_carlo_pi needs hipCUB, absent
from the nightly split-artifact install) -- shows up in CTest as "Not Run". Such
a category is reported as SKIPPED ("binary not available") rather than FAILED, so
missing infrastructure is distinguished from a sample that ran and failed.
"""

import re

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
        r"rocjpeg|rocdecode|hipblas|hipblaslt|hipfft|hipsolver|hipsparse|"
        r"hipsparselt|hiprand|hipcub|hipdnn|hiptensor|rccl|composable)([_-]|$)",
    ),
    ("tutorials", r"^(reduction|programming)([_-]|$)"),
    # rocprofiler-sdk API samples need the profiler runtime data (metrics
    # config.yaml + aqlprofile) from the --rocprofiler-sdk artifact; gated below.
    ("profiler_sdk", r"^rocprofiler([_-]|$)"),
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


# CTest lists each failure as "<num> - <name> (<status>)". A status of "Not Run"
# means CTest could not find the sample's executable -- i.e. the binary was not
# built because a required ROCm component is absent -- as opposed to a sample that
# ran and failed ("Failed", "Timeout", "Subprocess aborted", "SEGFAULT", ...).
_FAILED_LINE = re.compile(r"^\s*\d+\s*-\s*(?P<name>\S+)\s*\((?P<status>[^)]+)\)\s*$")
_NOT_AVAILABLE_STATUS = "not run"


def _partition_ctest_failures(out: str) -> tuple[list[str], list[str]]:
    """Split CTest failure lines into (not-available, real-failure) name lists."""
    not_available: list[str] = []
    real_failures: list[str] = []
    for line in out.splitlines():
        m = _FAILED_LINE.match(line)
        if not m:
            continue
        entry = f"{m.group('name')} ({m.group('status').strip()})"
        if m.group("status").strip().lower() == _NOT_AVAILABLE_STATUS:
            not_available.append(entry)
        else:
            real_failures.append(entry)
    return not_available, real_failures


# Binaries the "tools" bucket (rocgdb/rocprof/rocprofv3 samples) invokes. They
# ship only when the ROCm install carries the debugger + profiler artifact,
# which the nightly install (COMMON_FLAGS in e2e-tests.yml) does not pull today.
_TOOLS_BINARIES = ("rocgdb", "rocprof", "rocprofv3")

# The rocprofiler-sdk API samples abort at runtime unless the profiler runtime
# data is installed -- specifically the metrics config that ships with the
# --rocprofiler-sdk artifact (rocprofiler-sdk + aqlprofile + rocprofiler-sdk_run).
# Probe for this file to gate the "profiler_sdk" bucket.
_PROFILER_SDK_RUNTIME_FILE = "share/rocprofiler-sdk/config.yaml"


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

    # Gate the profiler_sdk bucket: the rocprofiler-sdk samples build fine but
    # abort at runtime ("Metric file 'config.yaml' not found") unless the
    # profiler runtime data is installed. Skip (documented gate) when it is
    # absent; run and enforce when present.
    if category == "profiler_sdk":
        probe = target_executor.run(f"test -f {rock_dir}/{_PROFILER_SDK_RUNTIME_FILE}", timeout=60)
        if not probe.ok:
            pytest.skip(
                f"rocm-examples 'profiler_sdk' bucket needs the rocprofiler-sdk runtime data "
                f"({_PROFILER_SDK_RUNTIME_FILE}) from the --rocprofiler-sdk artifact (not in this "
                "install); add --rocprofiler-sdk to COMMON_FLAGS in .github/workflows/e2e-tests.yml "
                "to enable it."
            )

    result = target_executor.run(
        f"env PATH={rocm_bin}:$PATH LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} "
        f"ctest --test-dir {rocm_examples_build_dir} --output-on-failure -R {pattern!r}",
        timeout=7200,
    )
    combined = result.stdout + result.stderr
    if "No tests were found" in combined:
        pytest.skip(f"rocm-examples category {category!r}: no matching tests in this build")

    if result.ok and "100% tests passed" in result.stdout:
        return

    # Not fully green. Separate samples whose binary was never built (component
    # not shipped -> "Not Run") from samples that ran and failed. Real failures
    # still fail the test; a category red *only* because of missing binaries is a
    # skip ("binary not available"), so absent infrastructure never masquerades as
    # a product defect and never blocks the suite.
    not_available, real_failures = _partition_ctest_failures(result.stdout)
    if not real_failures and not_available:
        pytest.skip(
            f"rocm-examples category {category!r}: "
            f"{len(not_available)} sample binary(ies) not available in this install "
            f"(required ROCm component not shipped): {', '.join(not_available)}"
        )
    fully_green = result.ok and "100% tests passed" in result.stdout
    assert fully_green, (
        f"rocm-examples category {category!r} CTest not fully green (exit={result.exit_code}):\n"
        + (f"binaries not available (skipped-if-alone): {', '.join(not_available)}\n" if not_available else "")
        + _ctest_summary(result.stdout)
    )
