# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_hipblaslt_linear_shape_boundary.py -- hipBLASLt integer overflow shape boundary validation.

Validates that hipBLASLt correctly handles nn.Linear calls at token-count
boundaries where 16-bit integer overflow may occur (32767/32768/65535/65536).
Tests both single-batch and multi-batch cases. One test (batch=16, tokens=32768)
is expected to raise a warning; all others should complete without errors.

Source script:
    tests/e2e/hipblaslt/src/hipblaslt_linear_shape_boundary.py

Runtime: < 2 minutes (8 parametrized forward passes, no training).

This test exercises the PyTorch → hipBLASLt integration path for edge-case
input shapes (integer overflow boundary condition). PyTorch is provisioned on
the execution node via ``require_torch``; the workload script runs under the
provisioned interpreter (``torch_python``) so no coordinator-side torch import
is ever needed.
"""

import pathlib
import shlex

import pytest

from tests.common.ml_provisioning.workload import workload_failure_detail

_SRC = pathlib.Path(__file__).parent / "src" / "hipblaslt_linear_shape_boundary.py"
_SRC_DIR = str(_SRC.parent)


@pytest.mark.runtime.fast
def test_hipblaslt_linear_shape_boundary(
    require_torch,
    torch_python: str,
    target_executor,
    ld_path: dict,
):
    """Run hipBLASLt nn.Linear shape boundary test (8 cases) via subprocess.

    PyTorch is provisioned on the execution node (``require_torch`` skips when it
    is unavailable, or fails on a genuine ROCm runtime incompatibility); the
    workload runs under the provisioned interpreter (``torch_python``) rather than
    a coordinator-side import.

    When running against a remote node the ``src/`` directory is SFTP-staged first
    so the script is available at the remote path used in the run command.
    """
    # Stage src/ to the remote node when the underlying executor supports it
    # (SshExecutor.upload_tree), then resolve the remote path for the script.
    # For local/DryRun executors the local path is used as-is.
    executor = next(iter(target_executor))
    if hasattr(executor, "upload_tree"):
        remote_src_dir = executor.upload_tree(_SRC_DIR)
        script_path = remote_src_dir + "/" + _SRC.name
    else:
        script_path = str(_SRC)

    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    python = shlex.quote(torch_python)
    src = shlex.quote(script_path)
    result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {python} {src}", timeout=300)
    detail = workload_failure_detail(result, "hipblaslt_linear_shape_boundary")
    assert result.ok, (
        f"hipblaslt_linear_shape_boundary failed (exit={result.exit_code}):{detail}\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "Result: PASSED" in result.stdout, f"Expected 'Result: PASSED' in stdout:{detail}\n{result.stdout[:2000]}"
