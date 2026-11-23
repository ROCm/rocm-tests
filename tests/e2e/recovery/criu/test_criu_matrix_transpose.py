# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the hip-tests MatrixTranspose HIP workload.

Launch the patched ``MatrixTranspose``, ``criu dump`` then ``criu restore`` it, and verify the
restored PID is alive and still the workload. Linux only; skips on GFX targets outside SUPPORTED_ARCHS.
"""

from __future__ import annotations

import logging

import pytest

from framework.reporting.allure_reporter import step
from tests.common.criu import steps as criu
from tests.common.criu.steps import assert_criu_dump_ok, assert_criu_restore_ok, launch_workload
from tests.common.system_tools.runner import skip_unsupported_arch

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of MatrixTranspose.
SUPPORTED_ARCHS = frozenset({"gfx1250", "gfx950", "gfx942", "gfx90a", "gfx908"})

# Token used to confirm the restored PID is still the MatrixTranspose workload.
_WORKLOAD_NAME = "MatrixTranspose"
# Let the workload spin up its HIP kernels before checkpointing.
_LAUNCH_SLEEP_SECS = 5
_LAUNCH_TIMEOUT = 60.0


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    """Skip when ``--gpu-arch`` names a GFX target outside SUPPORTED_ARCHS."""
    skip_unsupported_arch(gpu_arch, SUPPORTED_ARCHS, label="CRIU MatrixTranspose")


def _launch(executor, build, ld: str) -> str:
    """Launch ``MatrixTranspose`` in the background from the build dir; return its PID."""
    with step("Launch MatrixTranspose"):
        pid = launch_workload(
            executor,
            binary=build.binary,
            workdir=build.workdir,
            ld_library_path=ld,
            launch_sleep_secs=_LAUNCH_SLEEP_SECS,
            timeout=_LAUNCH_TIMEOUT,
            out_file="matrixtranspose.out",
        )
        logger.info("MatrixTranspose running with PID %s", pid)
        return pid


def _checkpoint(executor, criu_cmd: str, build, pid: str, full_log: bool = False) -> None:
    """Checkpoint the running process with ``criu dump`` and assert it stopped."""
    with step("Checkpoint with criu dump"):
        dump = criu.criu_dump(executor, criu_cmd, build.workdir, pid)
        criu.attach_criu_log(executor, build.workdir, "dump.log", full=full_log)
        assert_criu_dump_ok(dump, workdir=build.workdir, label="MatrixTranspose")


def _restore(executor, criu_cmd: str, build, full_log: bool = False) -> None:
    """Restore the checkpoint with ``criu restore`` and assert CRIU reported success."""
    with step("Restore with criu restore"):
        restore = criu.criu_restore(executor, criu_cmd, build.workdir)
        criu.attach_criu_log(executor, build.workdir, "restore.log", full=full_log)
        assert_criu_restore_ok(restore, workdir=build.workdir, label="MatrixTranspose")


# Serialized via this xdist_group: CRIU checkpoint/restore is a node-level (KFD) operation, so no
# two CRIU tests run concurrently on the same node.
@pytest.mark.runtime.medium
@pytest.mark.xdist_group("criu_serial")
def test_criu_checkpoint_restore_matrix_transpose(
    target_executor, ld_path, matrix_transpose_build, criu_runtime, gpu_arch, request
):
    """Launch MatrixTranspose, checkpoint it, restore it, and confirm it resumed under the original PID."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"

    pid = _launch(executor, matrix_transpose_build, ld)
    try:
        _checkpoint(executor, criu_runtime, matrix_transpose_build, pid, full_log)
        _restore(executor, criu_runtime, matrix_transpose_build, full_log)

        with step("Verify MatrixTranspose resumed under criu"):
            # criu restores into the original PID; confirm it is alive and still the workload.
            check = executor.run(
                f"if ps -p {pid} >/dev/null 2>&1 && "
                f'tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null | grep -q {_WORKLOAD_NAME}; '
                "then echo RESUMED_OK; else echo RESUMED_NO; fi"
            )
            assert "RESUMED_OK" in check.stdout, f"MatrixTranspose did not resume under criu (PID {pid})."
    finally:
        criu.kill_pid(executor, pid)
