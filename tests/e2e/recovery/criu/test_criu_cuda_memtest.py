# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the cuda_memtest HIP workload.

Three cumulative tests: checkpoint the running workload (``criu dump``), restore it
(``criu restore``), and the full cycle verifying it resumed under CRIU. Build/install come
from the ``cuda_memtest_build`` / ``criu_runtime`` fixtures; dump/restore use ``_criu_steps``.
Linux only; skips on GFX targets outside SUPPORTED_ARCHS when ``--gpu-arch`` is given.
"""

from __future__ import annotations

import logging
import re

import pytest

from framework.reporting.allure_reporter import report_metric, step
from tests.e2e.recovery.criu import _criu_steps as criu

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of cuda_memtest.
SUPPORTED_ARCHS = frozenset(
    {"gfx90a", "gfx908", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1102", "gfx1200", "gfx1201"}
)

_WORKLOAD_ARGS = "--disable_all --enable_test 0 --num_passes 1"
_LAUNCH_TIMEOUT = 60.0


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    if gpu_arch and gpu_arch not in SUPPORTED_ARCHS:
        pytest.skip(f"GPU arch {gpu_arch} not supported for CRIU cuda_memtest: {sorted(SUPPORTED_ARCHS)}")


def _total_vram_mb(executor, ld: str) -> int | None:
    """Return total VRAM (MB) for the acquired GPU via rocm-smi/amd-smi, or None."""
    cmd = (
        f"env LD_LIBRARY_PATH={ld} sh -c '"
        "{ command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showmeminfo vram 2>/dev/null; } ; "
        "{ command -v amd-smi >/dev/null 2>&1 && amd-smi metric -g 0 --mem-usage 2>/dev/null; }'"
    )
    out = executor.run(cmd, timeout=120).stdout or ""
    totals = [int(m) for line in out.splitlines() if "total" in line.lower() for m in re.findall(r"\d+", line)]
    if not totals:
        return None
    raw = max(totals)
    if raw >= (1 << 30):  # bytes
        return raw // (1024 * 1024)
    return raw if raw >= 1024 else None  # already MB


def _launch(executor, build, ld: str) -> str:
    """Size ``--max_num_blocks`` from VRAM (round(GB)*1000-2000), launch cuda_memtest, return its PID."""
    with step("Size and launch cuda_memtest"):
        vram_mb = _total_vram_mb(executor, ld)
        if not vram_mb:
            pytest.skip("Could not determine total GPU VRAM to size cuda_memtest.")
        blocks = round(vram_mb / 1024) * 1000
        blocks = blocks - 2000 if blocks > 2000 else blocks
        report_metric("GPU_VRAM_MB", float(vram_mb), "MB")
        report_metric("CUDA_MEMTEST_MAX_NUM_BLOCKS", float(blocks))
        # Capture the PID via $! -- `grep cuda_memtest` would also match the pytest argv.
        cmd = (
            f"cd {build.workdir} && rm -f dump.log restore.log cuda_memtest.out *.img 2>/dev/null; "
            f"env LD_LIBRARY_PATH={ld} nohup {build.binary} {_WORKLOAD_ARGS} --max_num_blocks {blocks} "
            "> cuda_memtest.out 2>&1 & pid=$!; disown 2>/dev/null || true; sleep 5; "
            'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 cuda_memtest.out; fi'
        )
        out = executor.run(cmd, timeout=_LAUNCH_TIMEOUT).stdout or ""
        pid = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")), "")
        assert pid, f"cuda_memtest did not start:\n{out[-1500:]}"
        logger.info("cuda_memtest running with PID %s (max_num_blocks=%d)", pid, blocks)
        return pid


def _checkpoint(executor, criu_cmd: str, build, pid: str) -> None:
    """Checkpoint the running workload and assert the dump succeeded and the PID is gone."""
    with step("Checkpoint with criu dump"):
        dump = criu.criu_dump(executor, criu_cmd, build.workdir, pid)
        log = criu.attach_criu_log(executor, build.workdir, "dump.log")
        assert "OK" in dump.stdout, f"criu dump did not report OK:\n{dump.stdout[-1500:]}\n{log[-1500:]}"
        assert "PID_GONE" in dump.stdout, "cuda_memtest still exists after criu dump."


def _restore(executor, criu_cmd: str, build) -> None:
    """Restore the checkpoint and assert CRIU reported success."""
    with step("Restore with criu restore"):
        restore = criu.criu_restore(executor, criu_cmd, build.workdir)
        log = criu.attach_criu_log(executor, build.workdir, "restore.log")
        assert "RESTORE_OK" in restore.stdout, f"criu restore did not finish successfully:\n{log[-1500:]}"


@pytest.mark.runtime.medium
def test_criu_check_point_cuda_memtest(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch):
    """Checkpoint a running cuda_memtest process with ``criu dump``."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    pid = None
    try:
        pid = _launch(executor, cuda_memtest_build, ld)
        _checkpoint(executor, criu_runtime, cuda_memtest_build, pid)
    finally:
        criu.kill_pid(executor, pid)


@pytest.mark.runtime.medium
def test_criu_restore_cuda_memtest(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch):
    """Restore a checkpointed cuda_memtest process with ``criu restore``."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    pid = None
    try:
        pid = _launch(executor, cuda_memtest_build, ld)
        _checkpoint(executor, criu_runtime, cuda_memtest_build, pid)
        _restore(executor, criu_runtime, cuda_memtest_build)
    finally:
        criu.kill_pid(executor, pid)


@pytest.mark.runtime.medium
def test_cuda_memtest_under_criu(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch):
    """Checkpoint, restore, then verify cuda_memtest resumed and is still running under CRIU."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    pid = None
    try:
        pid = _launch(executor, cuda_memtest_build, ld)
        _checkpoint(executor, criu_runtime, cuda_memtest_build, pid)
        _restore(executor, criu_runtime, cuda_memtest_build)
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
