# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Fixtures for tests/e2e/system_tools/amd_smi/.

Resolves the amd-smi binary, verifies metric/node subcommands are available,
and builds the CoralGemm workload binary for the power-under-load test.

pytest only auto-loads ``conftest.py`` for its own directory tree, so the RVS
build and config-lookup fixtures from the sibling ``tests/e2e/system_tools/rvs``
suite are re-exported here to make them requestable from this suite.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)


def _locate_amdsmitst_binary(executor, rock_dir: str) -> str:
    """Locate the amdsmitst gtest binary from rock_dir.

    The amdsmitst binary is pre-installed by the AMD SMI library (amd-smi-lib)
    into the ROCm share directory. We check two canonical locations in order:
        1. <rock_dir>/share/amd_smi/tests/amdsmitst
        2. <rock_dir>/share/amd_smi/amdsmitst

    Args:
        executor: NodeExecutorGroup or compatible executor to run commands on the test node.
        rock_dir: Resolved path to the TheRock/ROCm installation.

    Returns:
        Absolute path to the amdsmitst binary.

    Raises:
        pytest.fail.Exception: When amdsmitst is not found at either location.
    """
    candidates = [
        f"{rock_dir}/share/amd_smi/tests/amdsmitst",
        f"{rock_dir}/share/amd_smi/amdsmitst",
    ]

    for candidate in candidates:
        probe = executor.run(f"test -x {candidate} && echo FOUND")
        if (probe.stdout or "").strip() == "FOUND":
            logger.info("amdsmitst_binary: located at %s", candidate)
            return candidate

    pytest.fail(
        f"amdsmitst gtest binary not found in rock_dir={rock_dir} — "
        f"checked: {', '.join(candidates)}. "
        f"Ensure amd-smi-lib is installed with TheRock."
    )


@pytest.fixture
def amdsmitst_binary(target_executor, rock_dir: str) -> str:
    """Resolve the amdsmitst gtest binary from the ROCm install.

    The amdsmitst binary is a gtest suite that validates the amd-smi library's
    core functionality across all supported AMD GPU types. It exits 0 when all
    test cases pass and prints gtest PASSED/FAILED markers to stdout.

    Args:
        target_executor: NodeExecutorGroup for running commands on the test node.
        rock_dir: Resolved ROCm installation path from builder_plugin.

    Returns:
        Absolute path to the amdsmitst binary on the test node.
    """
    return _locate_amdsmitst_binary(target_executor, rock_dir)



from dataclasses import dataclass
import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess

import pytest

from tests.e2e.system_tools.rvs.conftest import (  # noqa: F401
    rvs_binary as _rvs_binary,
    rvs_env,
    rvs_find_conf,
    rvs_source as _rvs_source,
)

rvs_binary = _rvs_binary
rvs_source = _rvs_source

logger = logging.getLogger("rocm.test")

_CORAL_GEMM_URL = "https://github.com/AMD-HPC/CoralGemm"
_CORAL_GEMM_REF = os.environ.get("ROCM_TEST_CORAL_GEMM_REF") or None  # None = repo default branch


@dataclass(frozen=True)
class UbbEnv:
    """Resolved preflight state: amd-smi binary path."""

    amd_smi: str


def _resolve_amd_smi(rock_dir: str) -> str | None:
    """Prefer ``<rock_dir>/bin/amd-smi``; fall back to amd-smi on PATH. Returns None if absent."""
    if rock_dir:
        candidate = f"{rock_dir}/bin/amd-smi"
        if os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("amd-smi")


@pytest.fixture(scope="module")
def ubb_env(rock_dir: str) -> UbbEnv:
    """Verify amd-smi binary, metric --power, and node -p subcommands are available.

    Module-scoped so the preflight runs once per test module rather than before
    each individual test. Uses subprocess.run directly since module-scoped fixtures
    cannot consume function-scoped fixtures like target_executor.
    """
    logger.info("ubb_env: resolving amd-smi binary (rock_dir=%s)", rock_dir or "not set")
    amd_smi = _resolve_amd_smi(rock_dir)
    if not amd_smi:
        pytest.fail("amd-smi not found under --rock-dir or on PATH — it is required for this test suite")
    logger.info("ubb_env: amd-smi found at %s", amd_smi)

    logger.info("ubb_env: checking 'metric --power' subcommand availability")
    probe = subprocess.run([amd_smi, "metric", "--power", "--help"], capture_output=True, text=True)
    assert "power" in probe.stdout.lower(), (
        f"amd-smi 'metric --power' is not available on this node — it is mandatory.\n" f"stdout: {probe.stdout[:300]}"
    )
    logger.info("ubb_env: 'metric --power' available")

    logger.info("ubb_env: checking 'node -p' subcommand availability")
    probe_node = subprocess.run([amd_smi, "node", "-p", "--help"], capture_output=True, text=True)
    assert "power" in probe_node.stdout.lower(), (
        f"amd-smi 'node -p' is not available on this node — it is mandatory.\n" f"stdout: {probe_node.stdout[:300]}"
    )
    logger.info("ubb_env: 'node -p' available")

    logger.info("ubb_env: checking POWER_MANAGEMENT state for OAM_ID-0 GPU")
    list_result = subprocess.run([amd_smi, "list", "-e"], capture_output=True, text=True)
    oam0_gpu: str | None = None
    if list_result.returncode == 0:
        current: str | None = None
        for line in list_result.stdout.splitlines():
            m = re.search(r"GPU:\s*(\d+)", line)
            if m:
                current = m.group(1)
            if current and re.search(r"OAM_ID:\s*0\b", line):
                oam0_gpu = current
                break
    if oam0_gpu is not None:
        pwr = subprocess.run([amd_smi, "metric", "--power", "-g", oam0_gpu], capture_output=True, text=True)
        if "POWER_MANAGEMENT: DISABLED" in pwr.stdout:
            pytest.skip(
                f"GPU {oam0_gpu} (OAM_ID 0): POWER_MANAGEMENT is DISABLED — "
                "UBB_POWER fields return N/A; skipping power metric tests on this node"
            )
    logger.info("ubb_env: POWER_MANAGEMENT enabled — preflight complete")

    return UbbEnv(amd_smi=amd_smi)


@pytest.fixture(scope="session")
def coral_gemm_binary(external_build, cmake_build_dir, compiler_build_dir: str, rock_dir: str) -> str:
    """Clone and build CoralGemm; return absolute path to the gemm binary.

    Set ROCM_TEST_CORAL_GEMM_BIN to use a pre-built binary instead.
    CoralGemm is MIT-licensed (https://github.com/AMD-HPC/CoralGemm).
    """
    env_override = os.environ.get("ROCM_TEST_CORAL_GEMM_BIN", "").strip()
    if env_override:
        logger.info("coral_gemm_binary: using pre-built binary: %s", env_override)
        if not pathlib.Path(env_override).is_file():
            pytest.skip(f"ROCM_TEST_CORAL_GEMM_BIN={env_override} does not exist")
        return env_override

    logger.info("coral_gemm_binary: cloning CoralGemm from %s ref=%s", _CORAL_GEMM_URL, _CORAL_GEMM_REF or "default")
    dest = pathlib.Path(compiler_build_dir) / "system_tools" / "amd_smi" / "CoralGemm"
    repo_path = external_build.clone_repo(_CORAL_GEMM_URL, dest, ref=_CORAL_GEMM_REF)
    external_build.assert_license_present(repo_path)

    rocm_path = rock_dir
    logger.info("coral_gemm_binary: running cmake build in %s", repo_path)
    build_dir = cmake_build_dir(
        src=str(repo_path),
        subdir="system_tools/amd_smi/CoralGemm",
        gpu_arch=None,
        artifact="gemm",
        compiler_mode="cxx_hip",
        extra_cmake_args=[
            f"-DCMAKE_MODULE_PATH={rocm_path}/hip/cmake",
        ],
    )

    binary = pathlib.Path(build_dir) / "gemm"
    if not binary.is_file():
        pytest.fail(f"CoralGemm build succeeded but gemm binary not found at {binary} — check build logs")

    logger.info("coral_gemm_binary: binary ready at %s", binary)
    return str(binary)


# Ceiling for resolving the binary. Both probes are sub-second, so anything
# approaching this means the connection to the node is wedged.
_PROBE_TIMEOUT = 60.0


@pytest.fixture
def amd_smi_binary(target_executor, rock_dir: str) -> str:
    """Return a shell-quoted ``amd-smi`` belonging to the ROCm under test.

    Prefers ``<rock_dir>/bin/amd-smi`` so a host carrying several ROCm installs
    reports telemetry for the one ``--rock-dir`` selected rather than whichever
    copy happens to be first on ``PATH``. Both probes run through
    ``target_executor``, so a ``--remote-node`` run resolves the binary on the
    node that will be sampled instead of on the machine running pytest.
    """
    if rock_dir:
        candidate = shlex.quote(f"{rock_dir.rstrip('/')}/bin/amd-smi")
        if target_executor.run(f"test -x {candidate}", timeout=_PROBE_TIMEOUT).ok:
            return candidate

    which = target_executor.run("command -v amd-smi", timeout=_PROBE_TIMEOUT)
    resolved = (which.stdout or "").strip().splitlines()
    if which.ok and resolved:
        return shlex.quote(resolved[-1].strip())

    pytest.fail(
        f"amd-smi found neither at {rock_dir}/bin/amd-smi nor on PATH. It ships with "
        "ROCm, so a node without it is misconfigured rather than out of scope."
    )
