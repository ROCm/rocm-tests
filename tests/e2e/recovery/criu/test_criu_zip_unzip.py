# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CRIU checkpoint zip/unzip round-trip of the cuda_memtest HIP workload.

Launch cuda_memtest, ``criu dump`` it, ``tar`` the checkpoint image, untar it into a
fresh directory, ``criu restore`` from there, and verify the workload resumed. Linux.
"""

from __future__ import annotations

import logging
import re

import pytest

from framework.reporting.allure_reporter import report_metric, step
from tests.common.criu import steps as criu
from tests.e2e.recovery.criu.test_criu_cuda_memtest import (
    _checkpoint,
    _launch,
    _skip_if_unsupported_arch,
)

logger = logging.getLogger(__name__)

# tar of the checkpoint image can be sizeable (VRAM-scaled .img files) and is done under sudo.
_TAR_TIMEOUT = 600.0


def _zip_checkpoint(executor, workdir: str, tarball: str) -> None:
    """Archive the checkpoint image directory into *tarball* (root-owned .img files -> sudo)."""
    with step("Zip checkpoint image into a tar archive"):
        # tarball is created outside workdir so the archive never tries to include itself.
        result = executor.run(
            f"sudo -n rm -f {tarball} 2>/dev/null; "
            f"sudo -n tar -cvf {tarball} -C {workdir} . && echo ZIP_OK; "
            f"if [ -f {tarball} ]; then echo SIZE=$(stat -c %s {tarball} 2>/dev/null); fi",
            timeout=_TAR_TIMEOUT,
        )
        out = result.stdout or ""
        assert "ZIP_OK" in out, f"tar -cvf of the checkpoint image failed:\n{out[-1500:]}\n{result.stderr[-1500:]}"
        size = next((m.group(1) for line in out.splitlines() if (m := re.match(r"SIZE=(\d+)", line.strip()))), None)
        if size:
            report_metric("CRIU_TAR_BYTES", float(size), "bytes")
        logger.info("Checkpoint image archived to %s (%s bytes)", tarball, size or "unknown")


def _unzip_checkpoint(executor, tarball: str, dest: str) -> None:
    """Extract *tarball* into a fresh *dest* directory and assert the image files landed there."""
    with step("Unzip tar into a fresh destination directory"):
        result = executor.run(
            f"sudo -n rm -rf {dest} && sudo -n mkdir -p {dest} && "
            f"sudo -n tar -xvf {tarball} -C {dest} && echo UNZIP_OK; "
            f"echo IMG_COUNT=$(sudo -n sh -c 'ls {dest}/*.img 2>/dev/null | wc -l')",
            timeout=_TAR_TIMEOUT,
        )
        out = result.stdout or ""
        assert "UNZIP_OK" in out, f"tar -xvf into {dest} failed:\n{out[-1500:]}\n{result.stderr[-1500:]}"
        count = next(
            (int(m.group(1)) for line in out.splitlines() if (m := re.match(r"IMG_COUNT=(\d+)", line.strip()))), 0
        )
        assert count > 0, f"No CRIU .img files found in the untarred destination {dest}:\n{out[-1500:]}"
        report_metric("CRIU_RESTORE_IMG_COUNT", float(count))
        logger.info("Checkpoint image extracted to %s (%d .img files)", dest, count)


@pytest.mark.runtime.medium
@pytest.mark.xdist_group("criu_cuda_memtest_serial")
def test_criu_zip_unzip_cuda_memtest(target_executor, ld_path, cuda_memtest_build, criu_runtime, gpu_arch, request):
    """Zip a cuda_memtest checkpoint, unzip it elsewhere, and restore from the unzipped copy.

    Launch -> ``criu dump`` -> ``tar -cvf`` -> ``tar -xvf`` into a fresh dir -> ``criu restore``
    from that dir -> verify the process resumed. The launched PID is always killed in ``finally``.
    """
    _skip_if_unsupported_arch(gpu_arch)
    executor, ld = target_executor, ld_path["LD_LIBRARY_PATH"]
    full_log = request.config.getoption("capture") == "no"
    build = cuda_memtest_build

    # Sibling paths on the test node (POSIX; never os.path.join -- this host may be Windows).
    tarball = f"{build.workdir}_zipped.tar"
    dest = f"{build.workdir}_zipped"

    pid = _launch(executor, build, ld)
    try:
        # Checkpoint into the build workdir (writes dump.log + *.img there).
        _checkpoint(executor, criu_runtime, build, pid, full_log)

        _zip_checkpoint(executor, build.workdir, tarball)
        _unzip_checkpoint(executor, tarball, dest)

        with step("Restore from the unzipped checkpoint directory"):
            # criu_restore cd's into the workdir it is given and polls <workdir>/restore.log;
            # pass the untarred destination so the restore reads the archived-then-extracted image.
            restore = criu.criu_restore(executor, criu_runtime, dest)
            log = criu.attach_criu_log(executor, dest, "restore.log", full=full_log)
            assert "RESTORE_OK" in restore.stdout, f"criu restore from unzipped dir did not finish:\n{log[-1500:]}"

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
        executor.run(f"sudo -n rm -rf {dest} {tarball} 2>/dev/null; true")
