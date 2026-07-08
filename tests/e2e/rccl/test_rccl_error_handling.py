# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rccl_error_handling.py -- RCCL signal-handler error path validation.

Validate RCCL's *built-in* SIGSEGV signal handler, which is gated by the
RCCL_ENABLE_SIGNALHANDLER env var. The stub (src/rccl_error_handling/main.cpp)
calls ncclCommInitAll then dereferences a null pointer, raising SIGSEGV.
RCCL — not the stub — owns the handler:

    RCCL_ENABLE_SIGNALHANDLER=1 -> RCCL intercepts the fault and logs
                                   "Inside handler function signal"
    RCCL_ENABLE_SIGNALHANDLER=0 -> handler bypassed; the string is absent

Single-GPU: ncclCommInitAll(&comm, 1, NULL) uses one device, so hw.gpu is
declared per function and overrides the hw.multi_gpu profile default.
"""

import pytest


@pytest.mark.hw.gpu
@pytest.mark.runtime.fast
def test_rccl_error_handling_with_signal_handler(
    target_executor,
    ld_path: dict,
    require_rccl,
    rccl_error_handling_binary: str,
):
    """RCCL_ENABLE_SIGNALHANDLER=1: RCCL intercepts the SIGSEGV and logs its marker."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env RCCL_ENABLE_SIGNALHANDLER=1 LD_LIBRARY_PATH={ld} {rccl_error_handling_binary}",
        timeout=120,
    )
    combined = result.stdout + result.stderr
    assert "Inside handler function signal" in combined, (
        "RCCL signal handler did not intercept the fault — "
        "'Inside handler function signal' not found.\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )


@pytest.mark.hw.gpu
@pytest.mark.runtime.fast
def test_rccl_error_handling_no_signal_handler(
    target_executor,
    ld_path: dict,
    require_rccl,
    rccl_error_handling_binary: str,
):
    """RCCL_ENABLE_SIGNALHANDLER=0: handler bypassed; process crashes (SIGSEGV), marker absent.

    With the handler disabled, the null dereference faults straight to the OS — the run
    must terminate non-zero AND RCCL's handler marker must not appear.

    Note: SSH exec_command (Paramiko) does not emit a shell "Segmentation fault" line even
    when the process is killed by SIGSEGV; only the non-zero exit code is reliable.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env RCCL_ENABLE_SIGNALHANDLER=0 LD_LIBRARY_PATH={ld} {rccl_error_handling_binary}",
        timeout=120,
    )
    combined = result.stdout + result.stderr
    assert not result.ok, (
        "expected a crash (non-zero exit) from the deliberate null dereference "
        f"with the handler disabled, got exit=0:\nstdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "Inside handler function signal" not in combined, (
        "RCCL signal handler output appeared even though RCCL_ENABLE_SIGNALHANDLER=0.\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
