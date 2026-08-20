# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Environment fixtures for the amd-smi event-monitoring tests.

Resolves the ``amd-smi`` binary, verifies the ``event`` subcommand, GPU presence,
and passwordless sudo (GPU reset needs root), and gives each test a node scratch dir.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re

import pytest

logger = logging.getLogger("rocm.test")


@dataclass(frozen=True)
class RandomEventsEnv:
    """Resolved amd-smi test environment: binary path, ROCm path, node scratch dir."""

    amd_smi: str
    rocm_path: str
    scratch_dir: str


def _resolve_amd_smi(executor, rock_dir: str) -> str | None:
    """Return an amd-smi path: prefer ``<rock_dir>/bin/amd-smi``, else the one on PATH."""
    if rock_dir:
        probe = executor.run(f"test -x {rock_dir}/bin/amd-smi && echo OK")
        if "OK" in (probe.stdout or ""):
            return f"{rock_dir}/bin/amd-smi"
    which = executor.run("command -v amd-smi")
    if which.ok and (which.stdout or "").strip():
        return which.stdout.strip().splitlines()[-1].strip()
    return None


@pytest.fixture
def random_events_env(target_executor, rock_dir: str, run_ctx, request):
    """Pre-flight the node and yield a RandomEventsEnv; skip when prerequisites are absent."""

    logger.info("preflight: resolving amd-smi binary (rock_dir=%s)", rock_dir or "not set")
    amd_smi = _resolve_amd_smi(target_executor, rock_dir)
    if not amd_smi:
        pytest.skip("amd-smi not found under --rock-dir or on PATH")
    logger.info("preflight: amd-smi found at %s", amd_smi)

    # Check that the 'event' subcommand exists without printing its full help text.
    logger.info("preflight: checking amd-smi 'event' subcommand availability")
    evt = target_executor.run(f"{amd_smi} event --help 2>&1 | grep -qi 'event' && echo SUPPORTED || echo UNSUPPORTED")
    if "SUPPORTED" not in (evt.stdout or ""):
        pytest.skip("amd-smi 'event' subcommand not available on this node")
    logger.info("preflight: amd-smi 'event' subcommand available")

    logger.info("preflight: verifying passwordless sudo (required for GPU reset)")
    if not target_executor.run("sudo -n true").ok:
        pytest.skip("passwordless sudo not available -- required for amd-smi GPU reset")
    logger.info("preflight: passwordless sudo confirmed")

    logger.info("preflight: detecting AMD GPUs via amd-smi list")
    listing = target_executor.run(f"{amd_smi} list 2>/dev/null | grep -c '^GPU' || echo 0")
    match = re.search(r"\d+", listing.stdout or "")
    gpu_count = int(match.group()) if match else 0
    if gpu_count < 1:
        pytest.skip("no AMD GPUs detected by amd-smi")
    logger.info("preflight: %d AMD GPU(s) detected", gpu_count)

    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    scratch = f"/tmp/rocm_test_amdsmi_{run_ctx.run_id}_{tag}"  # nosec B108 - node-local scratch
    logger.info("preflight: creating node scratch dir %s", scratch)
    target_executor.run(f"rm -rf {scratch} && mkdir -p {scratch}")
    rocm_path = rock_dir or "/opt/rocm"
    try:
        yield RandomEventsEnv(amd_smi=amd_smi, rocm_path=rocm_path, scratch_dir=scratch)
    finally:
        logger.info("teardown: removing scratch dir %s", scratch)
        target_executor.run(f"rm -rf {scratch}")
