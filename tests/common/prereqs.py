# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""prereqs.py -- Generic OS package pre-installation helper for rocm-tests.

Usage in any test area's conftest.py::

    from tests.common.prereqs import install_packages

    _installed = False

    @pytest.fixture(autouse=True)   # function-scoped; target_executor cannot be session-scoped
    def my_packages(target_executor):
        global _installed
        if not _installed:
            install_packages(target_executor, ["rocblas", "hipblas"])
            _installed = True

Supported OS: Ubuntu 24.04, RHEL 10.1, SLES 15.7.
On RHEL and SLES the -devel variant of each package name is installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("rocm.test")


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
        f"Unsupported OS '{os_id}'. Supported: ubuntu 24.04, rhel 10.1, sles 15.7."
    )


def install_packages(executor, packages: list[str]) -> None:
    """Install *packages* on the target node using the OS-appropriate package manager.

    *executor* is a NodeExecutorGroup (as returned by target_executor). On RHEL
    and SLES each name is suffixed with -devel. Non-zero exit is logged as a
    warning because packages may already be installed (idempotent).
    """
    os_family = detect_os(executor)

    if os_family in ("rhel", "sles"):
        pkg_names = [f"{p}-devel" for p in packages]
    else:
        pkg_names = list(packages)

    pkg_str = " ".join(pkg_names)

    if os_family == "ubuntu":
        cmd = f"sudo apt-get install -y --no-install-recommends {pkg_str}"
    elif os_family == "rhel":
        cmd = f"sudo dnf install -y --nogpgcheck {pkg_str}"
    else:  # sles
        cmd = f"sudo zypper install -y {pkg_str}"

    logger.info("installing packages [%s]: %s", os_family, pkg_str)
    result = executor.run(cmd)
    if not result.ok:
        logger.warning(
            "package install exited %d — packages may already be present or unavailable:\n%s",
            result.exit_code,
            (result.stderr or "")[:500],
        )
