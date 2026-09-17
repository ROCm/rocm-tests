# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Preflight fixtures for tests/e2e/system_tools/amd_smi/.

Resolves the amd-smi binary, verifies metric/node subcommands are available,
and builds the CoralGemm workload binary for the power-under-load test.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import pathlib

import pytest

logger = logging.getLogger("rocm.test")

_CORAL_GEMM_URL = "https://github.com/AMD-HPC/CoralGemm"
_CORAL_GEMM_REF = os.environ.get("ROCM_TEST_CORAL_GEMM_REF") or None  # None = repo default branch


@dataclass(frozen=True)
class UbbEnv:
    """Resolved preflight state: amd-smi binary path."""

    amd_smi: str


def _resolve_amd_smi(executor, rock_dir: str) -> str | None:
    """Prefer ``<rock_dir>/bin/amd-smi``; fall back to amd-smi on PATH. Returns None if absent."""
    if rock_dir:
        probe = executor.run(f"test -x {rock_dir}/bin/amd-smi && echo OK")
        if (probe.stdout or "").strip() == "OK":
            return f"{rock_dir}/bin/amd-smi"
    which = executor.run("command -v amd-smi")
    if which.ok and (which.stdout or "").strip():
        return str(which.stdout).strip().splitlines()[-1].strip()
    return None


@pytest.fixture
def ubb_env(target_executor, rock_dir: str) -> UbbEnv:
    """Verify amd-smi binary, metric --power, and node -p subcommands are available."""
    logger.info("ubb_env: resolving amd-smi binary (rock_dir=%s)", rock_dir or "not set")
    amd_smi = _resolve_amd_smi(target_executor, rock_dir)
    if not amd_smi:
        pytest.fail("amd-smi not found under --rock-dir or on PATH — it is required for this test suite")
    logger.info("ubb_env: amd-smi found at %s", amd_smi)

    logger.info("ubb_env: checking 'metric --power' subcommand availability")
    probe = target_executor.run(f"{amd_smi} metric --power --help")
    assert "power" in (probe.stdout or "").lower(), (
        f"amd-smi 'metric --power' is not available on this node — it is mandatory for amd-smi power metric tests.\n"
        f"stdout: {(probe.stdout or '')[:500]}"
    )
    logger.info("ubb_env: 'metric --power' available")

    logger.info("ubb_env: checking 'node -p' subcommand availability")
    probe_node = target_executor.run(f"{amd_smi} node -p --help")
    assert "power" in (probe_node.stdout or "").lower(), (
        f"amd-smi 'node -p' is not available on this node — it is mandatory for amd-smi power metric tests.\n"
        f"stdout: {(probe_node.stdout or '')[:500]}"
    )
    logger.info("ubb_env: 'node -p' available — preflight complete")

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

    rocm_path = rock_dir or "/opt/rocm"
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
