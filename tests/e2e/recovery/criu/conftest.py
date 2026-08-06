# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU checkpoint/restore suite.

Provides the CRIU runtime prefix and the matrix transpose HIP build, grouped into the
COMMON / matrix transpose sections below.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
import os

import pytest

from framework.builder.binary_builder import find_rocm_clangpp
from framework.executors.cpu_executor import CpuExecutor
from tests.common import criu as criu_common

logger = logging.getLogger(__name__)


# ###########################################################################
# #### COMMON #### -- CRIU runtime prefix (host/SSH)
# ###########################################################################


@pytest.fixture(scope="session")
def criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Make CRIU + amdgpu_plugin ready on the test node (host/SSH); auto-install if not.

    Session-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return criu_common.ensure_criu_runtime(external_build, cmake_executor, framework_config)


# ###########################################################################
# #### MATRIX_TRANSPOSE #### -- hip-tests 2_Cookbook/0_MatrixTranspose
# Upstream ROCm/hip-tests; cloned/patched/built at runtime, not vendored. See NOTICES.md.
# ###########################################################################

_HIP_TESTS_URL = os.environ.get("ROCM_TEST_HIP_TESTS_URL", "https://github.com/ROCm/hip-tests.git")
_HIP_TESTS_REF = os.environ.get("ROCM_TEST_HIP_TESTS_REF", "3543bc3b9140e0a506ed3dec643b4def672bd171")

_MATRIX_TRANSPOSE_SUBDIR = "recovery/hip_tests"
_MT_SAMPLE_SUBPATH = "samples/2_Cookbook/0_MatrixTranspose"
_MT_PATCH_SCRIPT = os.path.join(os.path.dirname(criu_common.__file__), "patch_matrix_transpose.py")


@dataclass(frozen=True)
class MatrixTransposeBuild:
    """Compiled MatrixTranspose: ``binary`` path and its build ``workdir`` (CRIU dumps into the CWD)."""

    binary: str
    workdir: str


def _mt_resolve_dest(cmake_executor, compiler_build_dir: str) -> str:
    """Return the absolute checkout path on the build node."""
    base = os.path.join(compiler_build_dir, _MATRIX_TRANSPOSE_SUBDIR)
    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        return str(cmake_executor.workspace_path_for(base))
    return os.path.abspath(base)


@pytest.fixture(scope="session")
def matrix_transpose_build(
    cmake_executor, rock_dir: str, compiler_build_dir: str, framework_config
) -> MatrixTransposeBuild:
    """Clone ROCm/hip-tests, patch the MatrixTranspose sample, and build it once per session.

    Clones at the pinned commit (``ROCM_TEST_HIP_TESTS_REF`` to override), transfers and runs the
    vendored ``patch_matrix_transpose.py`` (100-iteration loop + CMake device-lib flag), then
    ``cmake .. && make``. Skips when the ROCm toolchain is absent; fails if the build fails.
    """
    build_exec = _build_executor(cmake_executor, rock_dir)
    hipcc = f"{rock_dir}/bin/hipcc"

    tool_check = build_exec.run(
        f"(command -v {hipcc} >/dev/null 2>&1 || command -v hipcc >/dev/null 2>&1) && "
        "command -v cmake >/dev/null 2>&1 && command -v git >/dev/null 2>&1 && "
        "command -v python3 >/dev/null 2>&1 && echo TOOLS_OK"
    )
    if "TOOLS_OK" not in (tool_check.stdout or ""):
        pytest.skip(
            f"hipcc / cmake / git / python3 not found under {rock_dir}/bin or on PATH -- "
            "cannot build MatrixTranspose."
        )

    dest = _mt_resolve_dest(cmake_executor, compiler_build_dir)
    sample = f"{dest}/{_MT_SAMPLE_SUBPATH}"
    build_dir = f"{sample}/build"
    timeout = float(framework_config.therock.build_timeout_secs)

    # Clone hip-tests at the pinned commit; a blobless partial clone keeps it quick while still
    # allowing checkout of an arbitrary commit.
    clone = build_exec.run(
        "\n".join(
            (
                "set -e",
                f"rm -rf {dest}",
                f"mkdir -p {os.path.dirname(dest)}",
                f"git clone --filter=blob:none {_HIP_TESTS_URL} {dest}",
                f"cd {dest}",
                f"git checkout {_HIP_TESTS_REF}",
                "echo CLONE_OK",
            )
        ),
        timeout=timeout,
    )
    if "CLONE_OK" not in (clone.stdout or ""):
        pytest.fail(
            f"hip-tests clone/checkout failed (exit={clone.exit_code}):\n"
            f"stdout: {clone.stdout[-2000:]}\nstderr: {clone.stderr[-2000:]}"
        )

    # Transfer the vendored patch script into the cloned sample (base64 -> host/SSH transparent).
    with open(_MT_PATCH_SCRIPT, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode()
    transfer = build_exec.run(f"echo {payload} | base64 -d > {sample}/patch_matrix_transpose.py")
    if not transfer.ok:
        pytest.fail(
            f"Failed to transfer patch_matrix_transpose.py (exit={transfer.exit_code}):\n"
            f"stderr: {transfer.stderr[-1500:]}"
        )

    # Patch both sample files (cpp loop + CMake device-lib flag), then build. ROCM_PATH / HIP_PATH
    # are exported so hipcc/cmake resolve their internal toolchain under --rock-dir.
    script = "\n".join(
        (
            "set -e",
            f"export ROCM_PATH={rock_dir}",
            f"export HIP_PATH={rock_dir}",
            f"export PATH={rock_dir}/bin:$PATH",
            f"cd {sample}",
            "python3 patch_matrix_transpose.py .",
            f"rm -rf {build_dir}",
            f"mkdir -p {build_dir}",
            f"cd {build_dir}",
            "cmake .. && make",
            "test -x ./MatrixTranspose && echo BUILD_OK",
        )
    )
    logger.info("Building MatrixTranspose (hip-tests) in %s", build_dir)
    result = build_exec.run(script, timeout=timeout)
    if "BUILD_OK" not in (result.stdout or ""):
        pytest.fail(
            f"MatrixTranspose build failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        )
    return MatrixTransposeBuild(binary=f"{build_dir}/MatrixTranspose", workdir=build_dir)


# #### END MATRIX_TRANSPOSE ####
