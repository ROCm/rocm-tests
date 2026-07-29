# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the LLNL RAJAPerf HIP workload.

Checkpoint a running ``raja-perf.exe`` (``criu dump``), then restore it and verify it resumed
(``criu restore``). Build/runtime come from the ``rajaperf_build`` / ``criu_runtime`` fixtures;
dump/restore helpers from ``_criu_steps``. Linux only; skips on GFX targets outside SUPPORTED_ARCHS.
"""

from __future__ import annotations

import logging

import pytest
from tests.e2e.recovery.criu.test_criu_cuda_memtest import _checkpoint, _restore

from framework.reporting.allure_reporter import step
from tests.e2e.recovery.criu import _criu_steps as criu

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of RAJAPerf: gfx1250 (MI450), gfx950 (MI350X),
# gfx942 (MI300A), gfx90a (MI250X/MI210/MI200), gfx908 (MI100).
SUPPORTED_ARCHS = frozenset({"gfx1250", "gfx950", "gfx942", "gfx90a", "gfx908"})

# RAJAPerf variants to exercise under CRIU.
_WORKLOAD_ARGS = "-v RAJA_HIP Base_HIP"
# Token used to confirm the restored PID is still the RAJAPerf workload.
_WORKLOAD_NAME = "raja-perf"
# Let the workload spin up its RAJA/HIP kernels before checkpointing.
_LAUNCH_SLEEP_SECS = 8
_LAUNCH_TIMEOUT = 60.0


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    if gpu_arch and gpu_arch not in SUPPORTED_ARCHS:
        pytest.skip(f"GPU arch {gpu_arch} not supported for CRIU RAJAPerf: {sorted(SUPPORTED_ARCHS)}")


def _launch(executor, build, ld: str, rock_dir: str) -> str:
    """Launch ``raja-perf.exe`` in the background from the build dir and return its PID.

    The PID is captured via ``$!`` (the exact launched process). ``rock_dir`` is accepted for
    signature parity with the shared launch helper and is unused here.
    """
    del rock_dir  # unused: RAJAPerf launch needs no VRAM sizing
    with step("Launch RAJAPerf"):
        # Launch from build.workdir so ./bin/raja-perf.exe resolves and CRIU dumps into this CWD;
        # clear any stale dump/restore logs and image files first.
        cmd = (
            f"cd {build.workdir} && rm -f dump.log restore.log rajaperf.out *.img 2>/dev/null; "
            f"env LD_LIBRARY_PATH={ld} nohup ./bin/raja-perf.exe {_WORKLOAD_ARGS} "
            "> rajaperf.out 2>&1 & pid=$!; disown 2>/dev/null || true; "
            f"sleep {_LAUNCH_SLEEP_SECS}; "
            'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 rajaperf.out; fi'
        )
        out = executor.run(cmd, timeout=_LAUNCH_TIMEOUT).stdout or ""
        pid = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")), "")
        assert pid, f"RAJAPerf did not start:\n{out[-1500:]}"
        logger.info("RAJAPerf running with PID %s", pid)
        return pid


# One checkpoint shared by the two tests in a single run: launch + criu dump happen once
# (materialized on first use so either test still runs standalone) and the on-disk image is reused
# by the restore test. NOTE: not xdist-safe -- with -n the tests may split across workers, each
# re-dumping; run these single-process.
_CHECKPOINT: dict = {}


def _ensure_checkpoint(executor, criu_cmd: str, build, ld: str, rock_dir: str, full_log: bool) -> str:
    """Launch RAJAPerf and ``criu dump`` it once per run; return the checkpointed PID.

    Cached after the first call so the restore test reuses the same image instead of re-dumping.
    """
    if not _CHECKPOINT.get("done"):
        pid = _launch(executor, build, ld, rock_dir)
        try:
            _checkpoint(executor, criu_cmd, build, pid, full_log)
        except BaseException:
            criu.kill_pid(executor, pid)  # dump failed -> workload still alive
            raise
        _CHECKPOINT["pid"] = pid
        _CHECKPOINT["done"] = True
    return _CHECKPOINT["pid"]


@pytest.mark.runtime.medium
def test_criu_check_point_rajaperf(target_executor, ld_path, rajaperf_build, criu_runtime, gpu_arch, rock_dir, request):
    """Checkpoint a running RAJAPerf process with ``criu dump``."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    _ensure_checkpoint(executor, criu_runtime, rajaperf_build, ld, rock_dir, full_log)


@pytest.mark.runtime.medium
def test_criu_restore_rajaperf(target_executor, ld_path, rajaperf_build, criu_runtime, gpu_arch, rock_dir, request):
    """Restore the checkpointed RAJAPerf process and verify it resumed and is still running."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    pid = _ensure_checkpoint(executor, criu_runtime, rajaperf_build, ld, rock_dir, full_log)
    try:
        _restore(executor, criu_runtime, rajaperf_build, full_log)
        with step("Verify RAJAPerf resumed under criu"):
            # criu restores into the original PID; confirm it is alive and still the workload.
            check = executor.run(
                f"if ps -p {pid} >/dev/null 2>&1 && "
                f'tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null | grep -q {_WORKLOAD_NAME}; '
                "then echo RESUMED_OK; else echo RESUMED_NO; fi"
            )
            assert "RESUMED_OK" in check.stdout, f"RAJAPerf did not resume under criu (PID {pid})."
    finally:
        criu.kill_pid(executor, pid)
