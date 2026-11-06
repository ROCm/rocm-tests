# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hipinfo.py -- hipInfo.exe validation for Windows ROCm installations.

Validates:
    1. ``hipInfo.exe`` executes successfully on Windows against a TheRock/ROCm
       install (``<rock_dir>/bin/hipInfo.exe``).
    2. Its output contains every field ``WindowsGpuDetector``'s ``hipInfo``
       fallback tier relies on: ``memInfo.total``, ``memInfo.free``,
       ``gcnArchName``, ``Name``, ``computeMode``.
    3. The output is non-trivial (not an empty or truncated run).

This complements (does not replace) the hardware-independent unit coverage
for ``hipInfo`` parsing in ``tests/dry_run/test_windows_gpu_detector.py``,
which mocks the subprocess output. This test is the one place that actually
invokes the real ``hipInfo.exe`` binary on real Windows GPU hardware.

Explicit markers (override the ``tests/e2e/system_tools`` directory default of
``ci.nightly``/``os.linux`` — this test must run in the PR gate, and only on
Windows):
    hw.gpu, ci.pr, layer.runtime, os.windows, runtime.fast
"""

from __future__ import annotations

import logging

import pytest

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("memInfo.total:", "memInfo.free:", "gcnArchName:", "Name:", "computeMode:")


# Dotted marker attributes are installed dynamically by the root conftest.
@pytest.mark.hw.gpu  # type: ignore[attr-defined]
@pytest.mark.ci.pr  # type: ignore[attr-defined]
@pytest.mark.layer.runtime  # type: ignore[attr-defined]
@pytest.mark.os.windows  # type: ignore[attr-defined]
@pytest.mark.runtime.fast  # type: ignore[attr-defined]
def test_hipinfo(target_executor, rock_dir: str) -> None:
    """Validate hipInfo.exe execution and output on Windows.

    Args:
        target_executor: Executor for running commands on the acquired GPU node.
        rock_dir: Path to the TheRock/ROCm installation
                  (e.g. ``C:\\Program Files\\AMD\\ROCm``).
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
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert result.stdout, "hipInfo.exe produced no output"
    assert (
        len(result.stdout) > 100
    ), f"hipInfo.exe produced insufficient output (length={len(result.stdout)}).\nOutput:\n{result.stdout}"

    missing_keys = [key for key in _REQUIRED_KEYS if key not in result.stdout]
    if missing_keys:
        pytest.fail(
            f"hipInfo.exe output missing required fields: {missing_keys}\n"
            f"Output excerpt (first 1500 chars):\n{result.stdout[:1500]}"
        )

    logger.info("hipInfo output validation passed (size=%d bytes)", len(result.stdout))
