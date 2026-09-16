# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""test_hip_mixbench.py -- HIP compute and memory bandwidth benchmark (mixbench).

Validates that the AMD GPU can execute a mixed compute/memory-bandwidth
workload and produces benchmark results across single-precision, double-precision,
half-precision, and integer arithmetic modes.

Supported GPU architectures: gfx906, gfx908, gfx90a, gfx942, gfx950
Supported OS: Ubuntu 24.04, CentOS 9.6, RHEL 10.1, SLES 15.7
Components under test: compiler, firmware, hip, kernel, rocr

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hip_runtime/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.medium
"""

from __future__ import annotations

import os

import pytest

_MIXBENCH_URL = "https://github.com/ekondis/mixbench.git"
_MIXBENCH_REF = os.environ.get("ROCM_TEST_MIXBENCH_REF", "master")


@pytest.fixture(scope="session")
def _mixbench_build_dir(external_build, cmake_build_dir, compiler_build_dir: str, gpu_arch: str | None):
    """Clone and build mixbench-hip; return the build directory path.

    Clones https://github.com/ekondis/mixbench.git once per session into
    the managed build workspace, then configures and builds the mixbench-hip
    CMake sub-project.  The resulting binary is ``mixbench-hip`` inside the
    build directory.
    """
    dest = os.path.join(compiler_build_dir, "hip_runtime", "mixbench")
    repo_dir = external_build.clone_repo(
        _MIXBENCH_URL,
        dest,
        ref=_MIXBENCH_REF,
    )
    external_build.assert_license_present(repo_dir)

    hip_src = os.path.join(str(repo_dir), "mixbench-hip")
    return cmake_build_dir(
        src=hip_src,
        subdir="hip_runtime/mixbench",
        gpu_arch=gpu_arch,
        compiler_mode="optional_cxx_hip",
        label="hip_runtime/mixbench",
        artifact="mixbench-hip",
    )


@pytest.fixture(scope="session")
def mixbench_hip_binary(_mixbench_build_dir: str, built_binary):
    """Path to the compiled ``mixbench-hip`` binary."""
    return built_binary(os.path.join(_mixbench_build_dir, "mixbench-hip"), "mixbench-hip")


@pytest.mark.runtime.medium
def test_hip_mixbench(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    mixbench_hip_binary: str,
) -> None:
    """Run mixbench-hip on an AMD GPU and verify benchmark results are produced.

    mixbench exercises a configurable mix of compute and memory operations across
    single-precision, double-precision, half-precision, and integer modes.  The
    binary exits 0 when all benchmark passes complete and prints a results table
    containing throughput (GFLOPS/GIOPS) and bandwidth (GB/sec) figures.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} ROCM_PATH={rock_dir} {mixbench_hip_binary}")
    assert result.ok, (
        f"mixbench-hip failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # mixbench prints device info and a CSV results table on success.
    assert result.stdout, "mixbench-hip produced no output — binary may have crashed silently"
    assert any(kw in result.stdout for kw in ("GFLOPS", "GB/sec", "Device", "device", "Flops", "bandwidth")), (
        f"mixbench-hip output does not contain expected benchmark data:\n" f"{result.stdout[:2000]}"
    )
