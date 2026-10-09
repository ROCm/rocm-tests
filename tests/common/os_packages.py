# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""os_packages.py -- Generic OS package pre-installation helper for rocm-tests.

Usage in any test area's conftest.py (install once per session, xdist-safe)::

    import os, pathlib
    from filelock import FileLock
    from tests.common.os_packages import install_packages

    _LOCK_DIR = pathlib.Path("output") / ".pkg-locks"
    _RUN_ID = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getpid())

    @pytest.fixture(autouse=True)   # function-scoped; target_executor cannot be session-scoped
    def my_packages(target_executor):
        _LOCK_DIR.mkdir(parents=True, exist_ok=True)
        sentinel = _LOCK_DIR / f"my_area_{_RUN_ID}.done"
        with FileLock(str(_LOCK_DIR / f"my_area_{_RUN_ID}.lock")):
            if sentinel.exists():
                return
            install_packages(target_executor, ["rocblas", "hipblas"])
            sentinel.touch()

A plain module-level ``_installed`` flag is NOT xdist-safe: each worker is a
separate process with its own flag, so all workers would install concurrently.

Supported OS: Ubuntu 24.04, RHEL 10.1, SLES 15.7.
On RHEL and SLES the -devel variant of each package name is installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rocm.test")

# Max seconds to wait for a package-manager install before treating it as hung.
_INSTALL_TIMEOUT_SECS = 600


def detect_os(executor) -> str:
    """Return 'ubuntu', 'rhel', or 'sles' from /etc/os-release on the target node."""
    result = executor.run("grep -oP '(?<=^ID=)[\"\\w-]+' /etc/os-release | tr -d '\"'")
    os_id = (result.stdout or "").strip().lower()
    if "ubuntu" in os_id:
        return "ubuntu"
    if "rhel" in os_id:
        return "rhel"
    if "sles" in os_id:
        return "sles"
    raise RuntimeError(
        f"Unsupported OS '{os_id}'. Supported: ubuntu 24.04, rhel 10.1, sles 15.7. "
        f"(exit={result.exit_code}, stderr={(result.stderr or '')[:200]})"
    )


def _require_passwordless_sudo(executor) -> None:
    """Fail early if passwordless sudo is unavailable — package installs need it."""
    probe = executor.run("sudo -n true")
    if not probe.ok:
        raise RuntimeError(
            "Passwordless sudo is required to install OS packages but 'sudo -n true' failed "
            f"(exit={probe.exit_code}, stderr={(probe.stderr or '')[:200]}). "
            "Configure NOPASSWD sudo for the test user on the target node."
        )


def install_packages(executor, packages: list[str]) -> None:
    """Best-effort install of *packages* using the OS-appropriate package manager.

    *executor* is a NodeExecutorGroup (as returned by target_executor). On RHEL
    and SLES each name is suffixed with -devel. Installs run with ``-y``.

    This step is best-effort: ROCm is frequently installed from a TheRock/tarball
    under --rock-dir rather than from the OS package repo, so the devel packages
    may legitimately be absent from apt/dnf/zypper (e.g. apt "Unable to locate
    package"). A non-zero exit is therefore logged with full detail but not raised
    here — the authoritative gate is the downstream cmake-config-dir existence
    check, which fails clearly if a required package's cmake dir is missing.
    """
    os_family = detect_os(executor)
    _require_passwordless_sudo(executor)

    pkg_names = [f"{p}-devel" for p in packages] if os_family in ("rhel", "sles") else list(packages)

    pkg_str = " ".join(pkg_names)

    if os_family == "ubuntu":
        cmd = f"sudo apt-get install -y --no-install-recommends {pkg_str}"
    elif os_family == "rhel":
        cmd = f"sudo dnf install -y --nogpgcheck {pkg_str}"
    else:  # sles
        cmd = f"sudo zypper install -y --no-gpg-checks {pkg_str}"

    logger.info("installing packages [%s]: %s", os_family, pkg_str)
    result = executor.run(cmd, timeout=_INSTALL_TIMEOUT_SECS)
    if not result.ok:
        # Not fatal: packages may be supplied via --rock-dir instead of the OS repo.
        # The cmake-dir existence check downstream is the real pass/fail signal.
        logger.warning(
            "package install [%s] returned non-zero (exit=%s) for [%s] — "
            "continuing; cmake-dir checks will catch any genuinely missing package.\n%s",
            os_family,
            result.exit_code,
            pkg_str,
            (result.stderr or "")[:500],
        )
