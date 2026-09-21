# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint/restore of the Kokkos performance-benchmark HIP workload.

Launch the Kokkos benchmark, ``criu dump`` then ``criu restore`` it, and verify the restored PID
is alive and still the workload. Linux only; skips on GFX targets outside SUPPORTED_ARCHS.
"""

from __future__ import annotations

import logging

import pytest

from framework.reporting.allure_reporter import step
from tests.common.criu import steps as criu

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of the Kokkos benchmark.
SUPPORTED_ARCHS = frozenset({"gfx1250", "gfx950", "gfx942", "gfx90a", "gfx908"})

# Kokkos google-benchmark flag; tabular counters keep the benchmark running long enough
# to be checkpointed while emitting parseable output.
_WORKLOAD_ARGS = "--benchmark_counters_tabular=true"
# Token used to confirm the restored PID is still the Kokkos benchmark workload.
_WORKLOAD_NAME = "Kokkos_PerformanceTest_Benchmark"
# Let the benchmark spin up its HIP kernels before checkpointing.
_LAUNCH_SLEEP_SECS = 5
_LAUNCH_TIMEOUT = 60.0


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    """Skip when ``--gpu-arch`` names a GFX target outside SUPPORTED_ARCHS."""
    if gpu_arch and gpu_arch not in SUPPORTED_ARCHS:
        pytest.skip(f"GPU arch {gpu_arch} not supported for CRIU Kokkos: {sorted(SUPPORTED_ARCHS)}")


def _launch(executor, build, ld: str) -> str:
    """Launch the Kokkos benchmark from the build dir; return its PID.

    The PID is captured via ``$!`` -- matching by process name would also match the pytest argv.
    ``HSA_XNACK=1`` is exported for gfx942 APU builds, which the APU build requires at run time.
    """
    with step("Launch Kokkos performance benchmark"):
        xnack = "HSA_XNACK=1 " if build.is_apu else ""
        cmd = (
            f"cd {build.workdir} && rm -f dump.log restore.log kokkos.out *.img 2>/dev/null; "
            f"env {xnack}LD_LIBRARY_PATH={ld} nohup {build.binary} {_WORKLOAD_ARGS} "
            f"> kokkos.out 2>&1 & pid=$!; disown 2>/dev/null || true; sleep {_LAUNCH_SLEEP_SECS}; "
            'if ps -p "$pid" >/dev/null 2>&1; then echo PID=$pid; else echo PID=; tail -n 40 kokkos.out; fi'
        )
        out = executor.run(cmd, timeout=_LAUNCH_TIMEOUT).stdout or ""
        pid = next((ln.split("=", 1)[1].strip() for ln in out.splitlines() if ln.strip().startswith("PID=")), "")
        assert pid, f"Kokkos benchmark did not start:\n{out[-1500:]}"
        logger.info("Kokkos benchmark running with PID %s", pid)
        return pid


def _checkpoint(executor, criu_cmd: str, build, pid: str, full_log: bool = False) -> None:
    """Checkpoint the running process with ``criu dump`` and assert it stopped."""
    with step("Checkpoint with criu dump"):
        dump = criu.criu_dump(executor, criu_cmd, build.workdir, pid)
        log = criu.attach_criu_log(executor, build.workdir, "dump.log", full=full_log)
        assert "OK" in dump.stdout, f"criu dump did not report OK:\n{dump.stdout[-1500:]}\n{log[-1500:]}"
        assert "PID_GONE" in dump.stdout, "Kokkos benchmark still exists after criu dump."


def _restore(executor, criu_cmd: str, build, full_log: bool = False) -> None:
    """Restore the checkpoint with ``criu restore`` and assert CRIU reported success."""
    with step("Restore with criu restore"):
        restore = criu.criu_restore(executor, criu_cmd, build.workdir)
        log = criu.attach_criu_log(executor, build.workdir, "restore.log", full=full_log)
        assert "RESTORE_OK" in restore.stdout, f"criu restore did not finish successfully:\n{log[-1500:]}"


# Serialized via this xdist_group: CRIU checkpoint/restore is a node-level (KFD) operation, so no
# two CRIU tests run concurrently on the same node.
@pytest.mark.runtime.medium
@pytest.mark.xdist_group("criu_serial")
def test_criu_checkpoint_restore_kokkos(target_executor, ld_path, kokkos_build, criu_runtime, gpu_arch, request):
    """Launch the Kokkos benchmark, checkpoint it, restore it, and confirm it resumed under the original PID."""
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"

    pid = _launch(executor, kokkos_build, ld)
    try:
        _checkpoint(executor, criu_runtime, kokkos_build, pid, full_log)
        _restore(executor, criu_runtime, kokkos_build, full_log)

        with step("Verify Kokkos benchmark resumed under criu"):
            # criu restores into the original PID; confirm it is alive and still the workload.
            check = executor.run(
                f"if ps -p {pid} >/dev/null 2>&1 && "
                f'tr "\\0" " " < /proc/{pid}/cmdline 2>/dev/null | grep -q {_WORKLOAD_NAME}; '
                "then echo RESUMED_OK; else echo RESUMED_NO; fi"
            )
            assert "RESUMED_OK" in check.stdout, f"Kokkos benchmark did not resume under criu (PID {pid})."
    finally:
        criu.kill_pid(executor, pid)
