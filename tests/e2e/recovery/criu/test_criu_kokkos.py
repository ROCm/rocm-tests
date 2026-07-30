# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the Kokkos performance-benchmark HIP workload.

Two tests sharing one checkpoint: checkpoint the running Kokkos benchmark (``criu dump``), then
restore it and verify it resumed and is still running (``criu restore``). The benchmark binary is
provided by the session ``kokkos_build`` fixture (clone + CMake/HIP build, device arch
auto-detected on the build node) and CRIU by the ``criu_runtime`` fixture; checkpoint/restore go
through the shared ``_criu_steps`` helpers. Linux only.
"""

from __future__ import annotations

import logging

import pytest

from framework.reporting.allure_reporter import step
from tests.e2e.recovery.criu import _criu_steps as criu

logger = logging.getLogger(__name__)

# Kokkos google-benchmark flag; tabular counters keep the benchmark running long
# enough to be checkpointed while emitting parseable output.
_WORKLOAD_ARGS = "--benchmark_counters_tabular=true"
_LAUNCH_TIMEOUT = 60.0
# Substring of the benchmark binary name, matched against /proc/<pid>/cmdline on resume.
_WORKLOAD_MATCH = "Kokkos_PerformanceTest_Benchmark"


def _launch(executor, build, ld: str) -> str:
    """Launch the Kokkos benchmark from the build dir and return its PID.

    The PID is captured via ``$!`` -- matching by process name would also match the pytest argv
    running this suite. ``HSA_XNACK=1`` is exported for gfx942 APU builds, which the APU build
    requires at run time.
    """
    with step("Launch Kokkos performance benchmark"):
        xnack = "HSA_XNACK=1 " if build.is_apu else ""
        cmd = (
            f"cd {build.workdir} && rm -f dump.log restore.log kokkos.out *.img 2>/dev/null; "
            f"env {xnack}LD_LIBRARY_PATH={ld} nohup {build.binary} {_WORKLOAD_ARGS} "
            "> kokkos.out 2>&1 & pid=$!; disown 2>/dev/null || true; sleep 5; "
            'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 kokkos.out; fi'
        )
        out = executor.run(cmd, timeout=_LAUNCH_TIMEOUT).stdout or ""
        pid = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")), "")
        assert pid, f"Kokkos benchmark did not start:\n{out[-1500:]}"
        logger.info("Kokkos benchmark running with PID %s", pid)
        return pid


def _checkpoint(executor, criu_cmd: str, build, pid: str, full_log: bool = False) -> None:
    """Checkpoint the running workload and assert the dump succeeded and the PID is gone."""
    with step("Checkpoint with criu dump"):
        dump = criu.criu_dump(executor, criu_cmd, build.workdir, pid)
        log = criu.attach_criu_log(executor, build.workdir, "dump.log", full=full_log)
        assert "OK" in dump.stdout, f"criu dump did not report OK:\n{dump.stdout[-1500:]}\n{log[-1500:]}"
        assert "PID_GONE" in dump.stdout, "Kokkos benchmark still exists after criu dump."


def _restore(executor, criu_cmd: str, build, full_log: bool = False) -> None:
    """Restore the checkpoint and assert CRIU reported success."""
    with step("Restore with criu restore"):
        restore = criu.criu_restore(executor, criu_cmd, build.workdir)
        log = criu.attach_criu_log(executor, build.workdir, "restore.log", full=full_log)
        assert "RESTORE_OK" in restore.stdout, f"criu restore did not finish successfully:\n{log[-1500:]}"


# One checkpoint shared by the two tests in a single run: launch + criu dump happen once
# (materialized on first use so either test still runs standalone) and the on-disk image is reused
# by the restore test. NOTE: not xdist-safe -- with -n the tests may split across workers, each
# re-dumping; run these single-process.
_CHECKPOINT: dict = {}


def _ensure_checkpoint(executor, criu_cmd: str, build, ld: str, full_log: bool) -> str:
    """Launch the Kokkos benchmark and ``criu dump`` it once per run; return the checkpointed PID.

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
def test_criu_check_point_kokkos(target_executor, ld_path, kokkos_build, criu_runtime, request):
    """Checkpoint a running Kokkos benchmark process with ``criu dump``."""
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    _ensure_checkpoint(executor, criu_runtime, kokkos_build, ld, full_log)


@pytest.mark.runtime.medium
def test_criu_restore_kokkos(target_executor, ld_path, kokkos_build, criu_runtime, request):
    """Restore the checkpointed Kokkos benchmark process and verify it resumed and is still running."""
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    pid = _ensure_checkpoint(executor, criu_runtime, kokkos_build, ld, full_log)
    try:
        _restore(executor, criu_runtime, kokkos_build, full_log)
        with step("Verify Kokkos benchmark resumed under criu"):
            # criu restores into the original PID; confirm it is alive and still the workload.
            check = executor.run(
                f"if ps -p {pid} >/dev/null 2>&1 && "
                f'tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null | grep -q {_WORKLOAD_MATCH}; '
                "then echo RESUMED_OK; else echo RESUMED_NO; fi"
            )
            assert "RESUMED_OK" in check.stdout, f"Kokkos benchmark did not resume under criu (PID {pid})."
    finally:
        criu.kill_pid(executor, pid)
