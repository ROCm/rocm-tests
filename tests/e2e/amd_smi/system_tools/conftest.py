# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Preflight fixtures for tests/e2e/amd_smi/system_tools/.

Resolves the amd-smi binary, verifies metric/node subcommands are available,
and builds the CoralGemm workload binary for the UBB power-under-load test.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import pathlib
import subprocess

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
        if "OK" in (probe.stdout or ""):
            return f"{rock_dir}/bin/amd-smi"
    which = executor.run("command -v amd-smi")
    if which.ok and (which.stdout or "").strip():
        return str(which.stdout).strip().splitlines()[-1].strip()
    return None


@pytest.fixture
def ubb_env(target_executor, rock_dir: str) -> UbbEnv:
    """Verify amd-smi binary, metric --power, and node -p subcommands are available.

    Skips cleanly on any missing prerequisite.
    """
    logger.info("ubb_env: resolving amd-smi binary (rock_dir=%s)", rock_dir or "not set")
    amd_smi = _resolve_amd_smi(target_executor, rock_dir)
    if not amd_smi:
        pytest.skip("amd-smi not found under --rock-dir or on PATH")
    logger.info("ubb_env: amd-smi found at %s", amd_smi)

    logger.info("ubb_env: checking 'metric --power' subcommand availability")
    _chk = "2>&1 | grep -qi 'power' && echo SUPPORTED || echo UNSUPPORTED"
    probe = target_executor.run(f"{amd_smi} metric --power --help {_chk}")
    if "SUPPORTED" not in (probe.stdout or ""):
        pytest.skip("amd-smi 'metric --power' not available on this node")
    logger.info("ubb_env: 'metric --power' available")

    logger.info("ubb_env: checking 'node -p' subcommand availability")
    probe_node = target_executor.run(f"{amd_smi} node -p --help {_chk}")
    if "SUPPORTED" not in (probe_node.stdout or ""):
        pytest.skip("amd-smi 'node -p' not available on this node")
    logger.info("ubb_env: 'node -p' available — preflight complete")

    return UbbEnv(amd_smi=amd_smi)


@pytest.fixture(scope="session")
def coral_gemm_binary(external_build, compiler_build_dir: str, rock_dir: str) -> str:
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
    dest = pathlib.Path(compiler_build_dir) / "amd_smi" / "CoralGemm"
    repo_path = external_build.clone_repo(_CORAL_GEMM_URL, dest, ref=_CORAL_GEMM_REF)
    external_build.assert_license_present(repo_path)

    build_dir = pathlib.Path(str(repo_path)) / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    rocm_path = rock_dir or "/opt/rocm"

    logger.info("coral_gemm_binary: running cmake in %s", build_dir)
    build_env = {
        **os.environ,
        "HIP_PLATFORM": "amd",
        "ROCM_PATH": rocm_path,
        "PATH": f"{rocm_path}/bin:{os.environ.get('PATH', '')}",
        "LD_LIBRARY_PATH": f"{rocm_path}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    cmake_proc = subprocess.run(
        [
            "cmake",
            f"-DCMAKE_MODULE_PATH={rocm_path}/hip/cmake",
            f"-DCMAKE_PREFIX_PATH={rocm_path}/lib/cmake",
            "..",
        ],
        cwd=str(build_dir),
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if cmake_proc.returncode != 0:
        pytest.skip(f"CoralGemm cmake failed:\n{cmake_proc.stderr[:800]}")

    logger.info("coral_gemm_binary: running make in %s", build_dir)
    external_build.make_build(repo_dir=build_dir, env=build_env)

    binary = build_dir / "gemm"
    if not binary.is_file():
        pytest.skip("CoralGemm build did not produce the gemm binary — check build logs")

    logger.info("coral_gemm_binary: binary ready at %s", binary)
    return str(binary)
