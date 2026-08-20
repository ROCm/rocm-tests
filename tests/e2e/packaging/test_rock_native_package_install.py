# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""
test_rock_native_package_install.py -- install ROCm from the public deb/rpm repos.

Realizes TMS rock_apt_install_deb / rock_yum_install_rpm: configure the public
prerelease ROCm package repository, install the ``amdrocm-<gfx>`` metapackage, and
verify it via ``dpkg -s`` / ``rpm -q``.

These mutate system packages, so they are **destructive** and **opt-in**: each skips
unless ``ROCM_TEST_ALLOW_PKG_INSTALL=1`` AND the process is root AND the host is the
matching distro family. Intended for disposable CI containers (the e2e job already
runs in an ephemeral container), never a shared host. Marked ci.weekly.

Override the package via ``ROCM_TEST_ROCM_PACKAGE`` (default derived from
``ROCM_TEST_ARTIFACT_GROUP``, e.g. ``amdrocm-gfx94x``).
"""

import os
import shlex
import shutil

import pytest

_ALLOW = os.environ.get("ROCM_TEST_ALLOW_PKG_INSTALL") == "1"
_ARTIFACT_GROUP = os.environ.get("ROCM_TEST_ARTIFACT_GROUP", "gfx94X-dcgpu")
_PACKAGE = os.environ.get("ROCM_TEST_ROCM_PACKAGE", "amdrocm-" + _ARTIFACT_GROUP.lower().split("-")[0])
_PKG_BASE = "https://rocm.prereleases.amd.com/packages"
_GPG_URL = f"{_PKG_BASE}/gpg/rocm.gpg"


def _os_profile() -> str | None:
    """Map /etc/os-release to a ROCm repo profile (e.g. ubuntu2404, debian12, rhel8)."""
    data: dict[str, str] = {}
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            for line in fh:
                if "=" in line:
                    k, v = line.rstrip().split("=", 1)
                    data[k] = v.strip('"')
    except OSError:
        return None
    os_id, ver = data.get("ID", ""), data.get("VERSION_ID", "")
    if not os_id or not ver:
        return None
    return f"{os_id}{ver.replace('.', '')}"


def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


@pytest.mark.runtime.medium
@pytest.mark.ci.weekly
def test_rock_apt_install_deb(cpu_executor):
    """Configure the public ROCm apt repo, install amdrocm-<gfx>, verify with dpkg."""
    if not _ALLOW:
        pytest.skip("destructive package install; set ROCM_TEST_ALLOW_PKG_INSTALL=1 to enable")
    if not _is_root():
        pytest.skip("apt package install requires root")
    if not shutil.which("apt-get"):
        pytest.skip("not a Debian-based system (no apt-get)")
    profile = _os_profile()
    if not profile:
        pytest.skip("could not determine os-release profile")
    script = (
        "set -e; "
        "mkdir -p --mode=0755 /etc/apt/keyrings; "
        f"wget -qO - {shlex.quote(_GPG_URL)} | gpg --dearmor | tee /etc/apt/keyrings/amdrocm.gpg >/dev/null; "
        "printf 'deb [arch=amd64 signed-by=/etc/apt/keyrings/amdrocm.gpg] "
        f"{_PKG_BASE}/{profile} stable main\\n' > /etc/apt/sources.list.d/rocm.list; "
        "apt-get update -qq; "
        f"apt-get install -y --no-install-recommends {shlex.quote(_PACKAGE)}; "
        f"dpkg -s {shlex.quote(_PACKAGE)}"
    )
    result = cpu_executor.run(f"bash -c {shlex.quote(script)}", timeout=3600)
    assert result.ok, f"apt install of {_PACKAGE!r} failed (exit={result.exit_code}):\n{result.stdout[-2000:]}"
    assert (
        "install ok installed" in result.stdout.lower()
    ), f"dpkg did not confirm {_PACKAGE!r}:\n{result.stdout[-1500:]}"


@pytest.mark.runtime.medium
@pytest.mark.ci.weekly
def test_rock_yum_install_rpm(cpu_executor):
    """Configure the public ROCm rpm repo, install amdrocm-<gfx>, verify with rpm -q."""
    if not _ALLOW:
        pytest.skip("destructive package install; set ROCM_TEST_ALLOW_PKG_INSTALL=1 to enable")
    if not _is_root():
        pytest.skip("rpm package install requires root")
    pkg_mgr = shutil.which("dnf") or shutil.which("yum")
    if not pkg_mgr:
        pytest.skip("not an RPM-based system (no dnf/yum)")
    profile = _os_profile()
    if not profile:
        pytest.skip("could not determine os-release profile")
    repo = (
        "[rocm]\n"
        "name=ROCm Prerelease Repository\n"
        f"baseurl={_PKG_BASE}/{profile}/x86_64/\n"
        "enabled=1\ngpgcheck=1\n"
        f"gpgkey={_GPG_URL}\n"
    )
    script = (
        "set -e; "
        f"printf {shlex.quote(repo)} > /etc/yum.repos.d/rocm.repo; "
        f"{pkg_mgr} clean all; "
        f"{pkg_mgr} install -y {shlex.quote(_PACKAGE)}; "
        f"rpm -q {shlex.quote(_PACKAGE)}"
    )
    result = cpu_executor.run(f"bash -c {shlex.quote(script)}", timeout=3600)
    assert result.ok, f"{pkg_mgr} install of {_PACKAGE!r} failed (exit={result.exit_code}):\n{result.stdout[-2000:]}"
    assert _PACKAGE in result.stdout, f"rpm -q did not confirm {_PACKAGE!r}:\n{result.stdout[-1500:]}"
