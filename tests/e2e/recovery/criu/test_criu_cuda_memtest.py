# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_criu_cuda_memtest.py -- CRIU checkpoint/restore stress test for a HIP workload.

Ported from: an external AMD test framework
    - criu_cuda_memtest_stressTest.py (class criuStressCudaMemTest)
    - CRIU/utils.py (get_test_pid / check_criu_installed / verify_criu_dump_and_restore /
      update_and_upload_to_artfctry)
which relied on ``execution_APIs.test``, ``platforms.BareMetal``, ``get_rocm_utils`` and an
Artifactory uploader -- none of which exist in rocm-tests. The whole workflow is
re-expressed with rocm-tests fixtures: ``target_executor`` (GPU node command
execution, ROCR_VISIBLE_DEVICES injected automatically), ``cuda_memtest_build`` and
``criu_runtime`` (this suite's conftest), ``ld_path`` (TheRock runtime libs), and the
Allure ``step`` / ``attach_text`` / ``report_metric`` helpers (which replace the
Artifactory dump.log/restore.log upload).

Validates:
    1. checkpoint_restore -- a running cuda_memtest HIP process can be checkpointed
       with ``criu dump`` (process gone afterward) and brought back with
       ``criu restore`` (process running again).
    2. restore_tampered_image_fails -- corrupting ``inventory.img`` makes restore
       fail with "Magic doesn't match for inventory.img" (negative test).
    3. restore_image_twice -- the same checkpoint images can be restored more than
       once (restore from a backed-up copy of the images).

Setup, sizing, and privilege:
    The cuda_memtest binary is built once per session by ``cuda_memtest_build`` (clone
    + hipify + patch + hipcc). ``--max_num_blocks`` is sized from the target GPU's
    total VRAM at runtime (round(GB) * 1000 - 2000, i.e. ~2 GB headroom: 64 GB ->
    62000, 192 GB -> 190000), matching the manual steps. CRIU requires root; rocm-tests
    executors expose no privilege API, so ``criu`` is invoked through a ``sudo -n``
    prefix resolved by ``criu_runtime``, which also AUTO-INSTALLS CRIU + amdgpu_plugin
    on the test node (via scripts/install_criu.py) when missing -- mirroring the
    source's check_criu_installed. The suite skips cleanly when passwordless sudo is
    unavailable (it can then neither install nor run CRIU).

Markers:
    hw.gpu / layer.runtime / ci.weekly / e2e.stack / os.linux are injected by the
    CATEGORY_PROFILES entry for tests/e2e/recovery/criu in taxonomy.py. runtime.* is
    declared explicitly on each test (soak for the happy-path checkpoint/restore, medium
    for the two shorter negative / restore-twice cases).

Supported architectures:
    gfx90a (MI250X/MI210/MI200), gfx908 (MI100), gfx942 (MI300X/MI300A/MI325X/MI308X),
    gfx950 (MI350X), and Navi gfx1100/gfx1101/gfx1102/gfx1200/gfx1201. The gate is
    evaluated against ``--gpu-arch`` when supplied; without it the arch cannot be
    determined reliably (lspci detection reports "unknown"), so the test proceeds.

Supported OS:
    Linux only (Ubuntu 20.04/22.04/24.04, CentOS 9.6, RHEL 8.10/9.7/10.1, SLES 15.7).
"""

from __future__ import annotations

import logging
import re

import pytest

from framework.reporting.allure_reporter import attach_text, report_metric, step

logger = logging.getLogger(__name__)

# GFX targets validated for CRIU checkpoint/restore of the cuda_memtest HIP workload.
SUPPORTED_ARCHS = frozenset(
    {
        "gfx90a",
        "gfx908",
        "gfx942",
        "gfx950",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1200",
        "gfx1201",
    }
)

# cuda_memtest CLI (from the manual steps): one pass of test 0, VRAM-sized block count.
_WORKLOAD_ARGS = "--disable_all --enable_test 0 --num_passes 1"

# Per-command timeouts (seconds).
_LAUNCH_TIMEOUT = 180.0
_CRIU_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Small parsing / sizing helpers
# ---------------------------------------------------------------------------


def _skip_if_unsupported_arch(gpu_arch: str | None) -> None:
    """Skip when ``--gpu-arch`` names a GFX target outside SUPPORTED_ARCHS.

    When ``--gpu-arch`` is not supplied the arch is unknown (lspci detection yields
    "unknown"), so the gate cannot be evaluated and the test proceeds.
    """
    if gpu_arch and gpu_arch not in SUPPORTED_ARCHS:
        pytest.skip(f"GPU arch {gpu_arch} is not in the supported set for CRIU cuda_memtest: {sorted(SUPPORTED_ARCHS)}")


def _parse_sentinel(text: str, key: str) -> str:
    """Return the value of a ``KEY=value`` sentinel line in *text*, or ``""``."""
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1].strip()
    return ""


def _probe_total_vram_mb(executor, ld: str) -> int | None:
    """Return total VRAM (MB) for the acquired GPU, or ``None`` if undeterminable.

    Queries ``rocm-smi`` then ``amd-smi`` on the target node (through the executor,
    with ROCR_VISIBLE_DEVICES pinned to the acquired GPU so index 0 is that GPU).
    The largest integer on any "total" line is taken and normalized: values that
    look like bytes (>= 1 GiB) are converted to MB; values already in the MB range
    are used as-is.
    """
    cmd = (
        f"env LD_LIBRARY_PATH={ld} sh -c '"
        "{ command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showmeminfo vram 2>/dev/null; } ; "
        "{ command -v amd-smi >/dev/null 2>&1 && amd-smi metric -g 0 --mem-usage 2>/dev/null; }'"
    )
    out = executor.run(cmd, timeout=120).stdout or ""
    candidates: list[int] = []
    for line in out.splitlines():
        if "total" in line.lower():
            candidates.extend(int(m) for m in re.findall(r"\d+", line))
    if not candidates:
        return None
    raw = max(candidates)
    if raw >= (1 << 30):  # looks like bytes
        return raw // (1024 * 1024)
    if raw >= 1024:  # already MB
        return raw
    return None


def _compute_max_num_blocks(vram_mb: int) -> int:
    """Return the cuda_memtest ``--max_num_blocks`` for *vram_mb* total VRAM.

    ``round(GB) * 1000 - 2000`` leaves ~2 GB headroom (64 GB -> 62000,
    192 GB -> 190000), matching the authoritative manual steps.
    """
    vram_gb = round(vram_mb / 1024)
    blocks = vram_gb * 1000
    return blocks - 2000 if blocks > 2000 else blocks


# ---------------------------------------------------------------------------
# CRIU workflow phases
# ---------------------------------------------------------------------------


def _launch_and_get_pid(executor, workdir: str, binary: str, ld: str, blocks: int) -> str:
    """Launch cuda_memtest in the background and return its PID.

    Clears any previous CRIU images/logs, launches the workload detached (so it
    survives past this command), then polls ``ps`` for up to 90 s for the process
    to appear -- a faithful re-expression of the source's ``get_test_pid(timeout=90)``.

    Raises:
        AssertionError: When the workload never appears (build or GPU failure).
    """
    cmd = (
        f"cd {workdir} && rm -f dump.log restore.log restore2.log cuda_memtest.out *.img 2>/dev/null; "
        f"rm -rf imgbackup 2>/dev/null; "
        f"env LD_LIBRARY_PATH={ld} nohup {binary} {_WORKLOAD_ARGS} --max_num_blocks {blocks} "
        f"> cuda_memtest.out 2>&1 & disown 2>/dev/null || true; "
        "for i in $(seq 1 90); do "
        "pid=$(ps -eo pid,args | grep '[c]uda_memtest --disable_all' | awk '{print $1}' | tail -n1); "
        '[ -n "$pid" ] && break; sleep 1; done; '
        "echo CUDA_MEMTEST_PID=$pid"
    )
    result = executor.run(cmd, timeout=_LAUNCH_TIMEOUT)
    pid = _parse_sentinel(result.stdout, "CUDA_MEMTEST_PID")
    assert pid, (
        "cuda_memtest fails to start: no PID found via ps within 90s "
        f"(exit={result.exit_code}).\nstdout: {result.stdout[-1500:]}\nstderr: {result.stderr[-1000:]}"
    )
    logger.info("cuda_memtest launched with PID %s (max_num_blocks=%d)", pid, blocks)
    return pid


def _criu_dump(executor, criu: str, workdir: str, pid: str):
    """Run ``criu dump`` on *pid* and report whether the process is gone afterward."""
    cmd = (
        f"cd {workdir} && {criu} dump -t {pid} -j -vvv -o dump.log --link-remap --file-lock; "
        "rc=$?; if [ $rc -eq 0 ]; then echo DUMP_OK; else echo DUMP_FAIL; tail -n 10 dump.log 2>/dev/null; fi; "
        f"if ps -p {pid} >/dev/null 2>&1; then echo PID_ALIVE; else echo PID_GONE; fi"
    )
    return executor.run(cmd, timeout=_CRIU_TIMEOUT)


def _criu_restore(executor, criu: str, workdir: str, log: str = "restore.log"):
    """Run a detached ``criu restore`` (``-d``) and echo RESTORE_OK / RESTORE_FAIL."""
    cmd = (
        f"cd {workdir} && {criu} restore -d -vvv -o {log} --shell-job --link-remap --file-lock; "
        f"rc=$?; if [ $rc -eq 0 ]; then echo RESTORE_OK; else echo RESTORE_FAIL; tail -n 10 {log} 2>/dev/null; fi"
    )
    return executor.run(cmd, timeout=_CRIU_TIMEOUT)


def _process_running(executor) -> bool:
    """Return True when at least one cuda_memtest process is currently running."""
    res = executor.run("ps -eo args | grep -q '[c]uda_memtest' && echo RUNNING || echo STOPPED")
    return "RUNNING" in (res.stdout or "")


def _attach_log(executor, workdir: str, filename: str, name: str) -> str:
    """Cat a CRIU log from the node and attach it to the Allure report; return its text."""
    text = executor.run(f"cat {workdir}/{filename} 2>/dev/null").stdout or ""
    attach_text(text, name=name)
    return text


def _kill_cuda_memtest(executor) -> None:
    """Best-effort cleanup: kill any lingering cuda_memtest (original or restored)."""
    executor.run("sudo -n pkill -9 -f cuda_memtest 2>/dev/null; true")


def _prepare_checkpoint(executor, criu: str, build, ld: str):
    """Size the workload, launch it, and checkpoint it; assert the dump succeeded.

    Shared by all three tests. Skips when total VRAM cannot be determined.

    Returns:
        The ``criu dump`` ExecutionResult (the CRIU images now exist in build.workdir).
    """
    with step("Size cuda_memtest from GPU VRAM"):
        vram_mb = _probe_total_vram_mb(executor, ld)
        if not vram_mb:
            pytest.skip("Could not determine total GPU VRAM (rocm-smi / amd-smi) to size cuda_memtest.")
        blocks = _compute_max_num_blocks(vram_mb)
        report_metric("GPU_VRAM_MB", float(vram_mb), "MB")
        report_metric("CUDA_MEMTEST_MAX_NUM_BLOCKS", float(blocks))
        logger.info("Sized cuda_memtest: vram=%d MB -> max_num_blocks=%d", vram_mb, blocks)

    with step("Launch cuda_memtest workload"):
        pid = _launch_and_get_pid(executor, build.workdir, build.binary, ld, blocks)

    with step("Checkpoint with criu dump"):
        dump = _criu_dump(executor, criu, build.workdir, pid)
        dump_log = _attach_log(executor, build.workdir, "dump.log", "dump.log")
        assert "DUMP_OK" in dump.stdout, (
            f"criu dump failed (exit={dump.exit_code}):\n"
            f"stdout: {dump.stdout[-1500:]}\ndump.log tail:\n{dump_log[-1500:]}"
        )
        assert "PID_GONE" in dump.stdout, "cuda_memtest is still alive after a successful criu dump (expected it gone)."
    return dump


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.runtime.soak
def test_criu_checkpoint_restore_cuda_memtest(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch):
    """Checkpoint a running cuda_memtest HIP process and restore it (happy path)."""
    _skip_if_unsupported_arch(gpu_arch)
    ld = ld_path["LD_LIBRARY_PATH"]
    build = cuda_memtest_build
    criu = criu_runtime
    try:
        _prepare_checkpoint(target_executor, criu, build, ld)
        with step("Restore with criu restore"):
            restore = _criu_restore(target_executor, criu, build.workdir)
            restore_log = _attach_log(target_executor, build.workdir, "restore.log", "restore.log")
            assert "RESTORE_OK" in restore.stdout, (
                f"criu restore failed (exit={restore.exit_code}):\n"
                f"stdout: {restore.stdout[-1500:]}\nrestore.log tail:\n{restore_log[-1500:]}"
            )
            assert _process_running(target_executor), "cuda_memtest is not running after a successful criu restore."
    finally:
        _kill_cuda_memtest(target_executor)


@pytest.mark.runtime.medium
def test_criu_restore_tampered_image_fails(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch):
    """Corrupting inventory.img makes criu restore fail with a magic-mismatch error."""
    _skip_if_unsupported_arch(gpu_arch)
    ld = ld_path["LD_LIBRARY_PATH"]
    build = cuda_memtest_build
    criu = criu_runtime
    try:
        _prepare_checkpoint(target_executor, criu, build, ld)
        with step("Corrupt inventory.img"):
            tamper = target_executor.run(
                f"cd {build.workdir} && sed -i '1i dumpdata 111000' inventory.img && echo TAMPERED"
            )
            assert "TAMPERED" in tamper.stdout, f"Failed to corrupt inventory.img: {tamper.stderr[-800:]}"
        with step("Restore must fail on the tampered image"):
            restore = _criu_restore(target_executor, criu, build.workdir)
            restore_log = _attach_log(target_executor, build.workdir, "restore.log", "restore.log")
            assert (
                "RESTORE_OK" not in restore.stdout
            ), "criu restore unexpectedly succeeded on a tampered inventory.img."
            assert "Magic doesn't match for inventory.img" in restore_log, (
                "Expected 'Magic doesn't match for inventory.img' in restore.log after tampering; "
                f"got:\n{restore_log[-1500:]}"
            )
    finally:
        _kill_cuda_memtest(target_executor)


@pytest.mark.runtime.medium
def test_criu_restore_image_twice(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch):
    """The same checkpoint images can be restored more than once (from a saved copy)."""
    _skip_if_unsupported_arch(gpu_arch)
    ld = ld_path["LD_LIBRARY_PATH"]
    build = cuda_memtest_build
    criu = criu_runtime
    try:
        _prepare_checkpoint(target_executor, criu, build, ld)

        with step("Back up the checkpoint images"):
            backup = target_executor.run(
                f"cd {build.workdir} && rm -rf imgbackup && mkdir imgbackup && cp -a *.img imgbackup/ && echo BACKUP_OK"
            )
            assert "BACKUP_OK" in backup.stdout, f"Failed to back up CRIU images: {backup.stderr[-800:]}"

        with step("First restore"):
            first = _criu_restore(target_executor, criu, build.workdir, log="restore.log")
            first_log = _attach_log(target_executor, build.workdir, "restore.log", "restore.log (1st)")
            assert "RESTORE_OK" in first.stdout, (
                f"first criu restore failed (exit={first.exit_code}):\n"
                f"stdout: {first.stdout[-1200:]}\nrestore.log tail:\n{first_log[-1200:]}"
            )
            assert _process_running(target_executor), "cuda_memtest not running after the first restore."
            _kill_cuda_memtest(target_executor)

        with step("Second restore from the backed-up images"):
            reset = target_executor.run(
                f"cd {build.workdir} && rm -f *.img && cp -a imgbackup/*.img . && echo RESET_OK"
            )
            assert "RESET_OK" in reset.stdout, f"Failed to restore image copy: {reset.stderr[-800:]}"
            second = _criu_restore(target_executor, criu, build.workdir, log="restore2.log")
            second_log = _attach_log(target_executor, build.workdir, "restore2.log", "restore.log (2nd)")
            assert "RESTORE_OK" in second.stdout, (
                f"second criu restore failed (exit={second.exit_code}):\n"
                f"stdout: {second.stdout[-1200:]}\nrestore2.log tail:\n{second_log[-1200:]}"
            )
            assert _process_running(target_executor), "cuda_memtest not running after the second restore."
    finally:
        _kill_cuda_memtest(target_executor)
