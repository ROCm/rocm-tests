# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""hipInfo.exe validation tests for Windows ROCm installations.

Validates:
    1. hipInfo.exe executes successfully on Windows with TheRock ROCm installation.
    2. All required GPU information fields are present in hipInfo output:
       - memInfo.total (total GPU memory)
       - memInfo.free (free GPU memory)
       - gcnArchName (GPU architecture name)
       - Name (GPU device name)
       - computeMode (GPU compute mode)
    3. hipInfo.exe produces non-empty output.

Markers:
    hw.gpu, ci.pr, layer.runtime, os.windows, runtime.fast

Platform:
    Windows-only test. Automatically skipped on Linux via os.windows marker
    and os_plugin which checks the platform at collection time.
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = [
    "memInfo.total:",
    "memInfo.free:",
    "gcnArchName:",
    "Name:",
    "computeMode:",
]


@pytest.mark.hw.gpu
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.os.windows
@pytest.mark.runtime.fast
def test_hipinfo(target_executor, rock_dir: str) -> None:
    """Validate hipInfo.exe execution and output on Windows.

    Verifies that hipInfo.exe:
    1. Executes successfully from the TheRock/ROCm installation
    2. Produces non-empty output (at least 100 characters)
    3. Contains all required GPU information fields

    Args:
        target_executor: Executor for running commands on the GPU node.
        rock_dir: Path to TheRock/ROCm installation (e.g., C:\\Program Files\\AMD\\ROCm).
    """
    if not rock_dir:
        pytest.fail(
            "rock_dir (--rock-dir) is required for this test. "
            "Set it to the TheRock/ROCm installation path (e.g., C:\\Program Files\\AMD\\ROCm)."
        )

    hipinfo_path = f"{rock_dir}/bin/hipInfo.exe"
    logger.info("Running hipInfo.exe from %s", hipinfo_path)

    result = target_executor.run(hipinfo_path, timeout=30.0)

    assert result.ok, (
        f"hipInfo.exe failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\n"
        f"stderr: {result.stderr[:500]}"
    )

    assert result.stdout and len(result.stdout) > 100, (
        f"hipInfo.exe produced insufficient output (length={len(result.stdout or '')}). "
        f"Output:\n{result.stdout or '(empty)'}"
    )

    missing_keys = [key for key in _REQUIRED_KEYS if key not in result.stdout]

    if missing_keys:
        pytest.fail(
            f"hipInfo.exe output missing required fields: {missing_keys}\n"
            f"Output excerpt (first 1500 chars):\n{result.stdout[:1500]}"
        )

    logger.info("hipInfo output validation passed (size=%d bytes)", len(result.stdout))
