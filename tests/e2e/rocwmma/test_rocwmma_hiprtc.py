# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocwmma_hiprtc.py — rocWMMA hipRTC GEMM sample validation.

Validates:
    1. The rocWMMA ``hipRTC_gemm`` sample compiles its GEMM kernel at runtime
       through hipRTC and runs it to completion, reporting "Finished!".
"""

from __future__ import annotations

import logging
import pathlib
import shlex

import pytest

logger = logging.getLogger(__name__)

_PASS_MARKER = "Finished!"


@pytest.mark.runtime.fast
def test_rocwmma_hiprtc(
    target_executor,
    hiprtc_gemm_binary: str,
    rock_dir: str,
    ld_path: dict,
):
    """Run the hipRTC GEMM sample and require its completion marker."""
    binary = pathlib.Path(hiprtc_gemm_binary)
    cmd = (
        f"cd {shlex.quote(str(binary.parent))} && "
        f"env LD_LIBRARY_PATH={shlex.quote(ld_path['LD_LIBRARY_PATH'])} "
        f"ROCM_PATH={shlex.quote(rock_dir)} ./{binary.name}"
    )

    logger.info("Running rocWMMA hipRTC sample: %s", cmd)
    result = target_executor.run(cmd, timeout=600.0)

    assert _PASS_MARKER in result.stdout, (
        f"{binary.name} did not report '{_PASS_MARKER}' (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
    )
