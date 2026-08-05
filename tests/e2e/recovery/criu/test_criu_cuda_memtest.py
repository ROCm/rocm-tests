# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the cuda_memtest HIP workload.

Two tests sharing one checkpoint: checkpoint the running workload (``criu dump``), then restore it
and verify it resumed and is still running (``criu restore``). Build/install come from the
``cuda_memtest_build`` / ``criu_runtime`` fixtures; dump/restore use ``tests.common.criu.steps``.
Both tests share the ``criu_cuda_memtest_serial`` xdist_group so they run serially on one worker
under parallel CI and the restore reuses the checkpoint. Linux only; skips on GFX targets outside
SUPPORTED_ARCHS when ``--gpu-arch`` is given.
"""

from __future__ import annotations

import logging

import pytest

from framework.reporting.allure_reporter import report_metric, step
from tests.common.criu import steps as criu

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of cuda_memtest.
SUPPORTED_ARCHS = frozenset(
    {"gfx90a", "gfx908", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"}
)

# Run Test 0 continuously (no --num_passes, so cuda_memtest loops until killed) -- the
# workload must stay alive for criu dump; criu dump / kill_pid terminate it afterward.
_WORKLOAD_ARGS = "--disable_all --enable_test 0"
_LAUNCH_TIMEOUT = 60.0

# GPU footprint for the workload, in cuda_memtest blocks (~1 MB each). Small on purpose: the
# upstream robustness test caps nothing, but CRIU's amdgpu_plugin drains each GPU buffer object via
# sDMA within a fence timeout, so a VRAM-filling allocation makes the dump time out ("failed to
# query fence status - Timer expired"). ~2 GB exercises real device memory while keeping the
# checkpoint fast.
_MAX_NUM_BLOCKS = 2000


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    if gpu_arch and gpu_arch not in SUPPORTED_ARCHS:
        pytest.skip(f"GPU arch {gpu_arch} not supported for CRIU cuda_memtest: {sorted(SUPPORTED_ARCHS)}")


def _launch(executor, build, ld: str) -> str:
    """Launch cuda_memtest with a small fixed GPU footprint (see ``_MAX_NUM_BLOCKS``); return its PID."""
    with step("Launch cuda_memtest"):
        report_metric("CUDA_MEMTEST_MAX_NUM_BLOCKS", float(_MAX_NUM_BLOCKS))
        # Capture the PID via $! -- `grep cuda_memtest` would also match the pytest argv.
        cmd = (
            f"cd {build.workdir} && rm -f dump.log restore.log cuda_memtest.out *.img 2>/dev/null; "
            f"env LD_LIBRARY_PATH={ld} nohup {build.binary} {_WORKLOAD_ARGS} --max_num_blocks {_MAX_NUM_BLOCKS} "
            "> cuda_memtest.out 2>&1 & pid=$!; disown 2>/dev/null || true; sleep 5; "
            'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 cuda_memtest.out; fi'
        )
        out = executor.run(cmd, timeout=_LAUNCH_TIMEOUT).stdout or ""
        pid = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")), "")
        assert pid, f"cuda_memtest did not start:\n{out[-1500:]}"
        logger.info("cuda_memtest running with PID %s (max_num_blocks=%d)", pid, _MAX_NUM_BLOCKS)
        return pid


def _checkpoint(executor, criu_cmd: str, build, pid: str, full_log: bool = False) -> None:
    """Checkpoint the running workload and assert the dump succeeded and the PID is gone."""
    with step("Checkpoint with criu dump"):
        dump = criu.criu_dump(executor, criu_cmd, build.workdir, pid)
        log = criu.attach_criu_log(executor, build.workdir, "dump.log", full=full_log)
        assert "OK" in dump.stdout, f"criu dump did not report OK:\n{dump.stdout[-1500:]}\n{log[-1500:]}"
        assert "PID_GONE" in dump.stdout, "cuda_memtest still exists after criu dump."


def _restore(executor, criu_cmd: str, build, full_log: bool = False) -> None:
    """Restore the checkpoint and assert CRIU reported success."""
    with step("Restore with criu restore"):
        restore = criu.criu_restore(executor, criu_cmd, build.workdir)
        log = criu.attach_criu_log(executor, build.workdir, "restore.log", full=full_log)
        assert "RESTORE_OK" in restore.stdout, f"criu restore did not finish successfully:\n{log[-1500:]}"


# Launch + criu dump happen once per worker and the on-disk image is reused by the restore test.
# Both tests share the ``criu_cuda_memtest_serial`` xdist_group, so they co-locate on one worker and
# this cache is reused (no re-dump). Correctness never depends on it, though -- _ensure_checkpoint
# re-launches and re-dumps when the cache is empty, so either test runs standalone.
_CHECKPOINT: dict = {}


def _ensure_checkpoint(executor, criu_cmd: str, build, ld: str, full_log: bool) -> str:
    """Launch cuda_memtest and ``criu dump`` it once per run; return the checkpointed PID.

    Cached after the first call so the restore test reuses the same image instead of re-dumping.
    """
    if not _CHECKPOINT.get("done"):
        pid = _launch(executor, build, ld)
        try:
            _checkpoint(executor, criu_cmd, build, pid, full_log)
        except BaseException:
            criu.kill_pid(executor, pid)  # dump failed -> workload still alive
            raise
        _CHECKPOINT["pid"] = pid
        _CHECKPOINT["done"] = True
    return _CHECKPOINT["pid"]


@pytest.mark.runtime.medium
@pytest.mark.xdist_group("criu_cuda_memtest_serial")
def test_criu_check_point_cuda_memtest(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch, request):
    """Checkpoint a running cuda_memtest process with ``criu dump``."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    _ensure_checkpoint(executor, criu_runtime, cuda_memtest_build, ld, full_log)


@pytest.mark.runtime.medium
@pytest.mark.xdist_group("criu_cuda_memtest_serial")
def test_criu_restore_cuda_memtest(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch, request):
    """Restore the checkpointed cuda_memtest process and verify it resumed and is still running."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    pid = _ensure_checkpoint(executor, criu_runtime, cuda_memtest_build, ld, full_log)
    try:
        _restore(executor, criu_runtime, cuda_memtest_build, full_log)
        with step("Verify cuda_memtest resumed under criu"):
            # criu restores into the original PID; confirm it is alive and still the workload.
            check = executor.run(
                f"if ps -p {pid} >/dev/null 2>&1 && "
                f'tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null | grep -q cuda_memtest; '
                "then echo RESUMED_OK; else echo RESUMED_NO; fi"
            )
            assert "RESUMED_OK" in check.stdout, f"cuda_memtest did not resume under criu (PID {pid})."
    finally:
        criu.kill_pid(executor, pid)
