# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_os_deps_plugin.py -- Real installation tests for os_deps_plugin.

Each test installs packages via OS_DEPS, verifies via specs() and InstallResult,
and a session teardown calls uninstall() for packages this session newly installed.

API coverage:
  install()   -- all spec forms, cross-distro, src=, sources=, command=
  specs()     -- structured {pkg: {installed, version}} after every install
  uninstall() -- plain name and version-pinned, session teardown
  upgrade()   -- to latest and to specific version

Platform coverage:
  Linux / Ubuntu  -- apt    (ci.pr and ci.nightly)
  Linux / RHEL    -- dnf    (ci.nightly, skips on non-dnf nodes)
  Linux / SLES    -- zypper (ci.nightly, skips on non-zypper nodes)
  Windows         -- winget / choco / scoop (ci.nightly, skips on Linux)
  Any platform    -- pip (skips if command not available)

Run with -s to see installation, specs, and teardown logs on the console.
"""

from __future__ import annotations

import pytest

from framework.plugins.os_deps_plugin import OS_DEPS

# ---------------------------------------------------------------------------
# Session teardown -- uninstall only packages this session newly installed
# ---------------------------------------------------------------------------

_session_installed: list[str] = []


@pytest.fixture(scope="session", autouse=True)
def _session_teardown(cmake_executor):
    """After all tests, uninstall only packages this session installed."""
    yield
    if not _session_installed:
        print("\n[Teardown] No packages were newly installed -- nothing to uninstall.")
        return
    print(f"\n[Teardown] Uninstalling packages installed this session: {_session_installed}")
    result = OS_DEPS().uninstall(*_session_installed, executor=cmake_executor)
    removed = [p for p, ok in result.items() if ok]
    skipped = [p for p, ok in result.items() if not ok]
    print(f"[Teardown] Removed: {removed}, Skipped (not installed): {skipped}")


# ---------------------------------------------------------------------------
# Linux / Ubuntu (apt) -- ci.pr
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_apt_install_common_build_tools(target_executor):
    """Install cmake, git, curl via apt; verify via specs() and InstallResult."""
    result = OS_DEPS().install("cmake", "git", "curl")
    for p in result.installed:
        if p not in _session_installed:
            _session_installed.append(p)

    info = OS_DEPS().specs("cmake", "git", "curl")
    print(f"\n[apt] common build tools specs: {info}")
    for pkg in ("cmake", "git", "curl"):
        assert info[pkg]["installed"], f"{pkg} not installed"
    assert result.version_summary(), "Expected version info after install"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_apt_install_version_pinned(target_executor):
    """Install a version-pinned package and verify via specs()."""
    result = OS_DEPS().install("cmake==3.22", on_missing="skip")
    if result.installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("cmake")
    print(f"\n[apt] version-pinned cmake specs: {info}")
    assert info["cmake"]["installed"], "cmake not installed"
    assert result.version_summary(), "Expected version info"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_apt_install_list_of_packages(target_executor):
    """Install multiple packages in one call; verify each via specs()."""
    result = OS_DEPS().install({"apt": ["wget", "unzip", "file"], "*": "wget"})
    if result.installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("wget", "unzip", "file")
    print(f"\n[apt] list install specs: {info}")
    for pkg in ("wget", "unzip", "file"):
        assert info[pkg]["installed"], f"{pkg} not installed"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_apt_already_installed_skips_install(target_executor):
    """Pre-installed packages show in already_present; specs() confirms."""
    result = OS_DEPS().install("git")
    info = OS_DEPS().specs("git")
    print(f"\n[apt] already-installed git specs: {info}")
    assert info["git"]["installed"], "git not installed"
    assert "git" in result.all_packages


# ---------------------------------------------------------------------------
# Linux / RHEL (dnf) -- skips on non-dnf nodes
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_dnf_install_development_tools(target_executor):
    """Install dnf-specific packages; skips on apt/zypper nodes."""
    result = OS_DEPS().install(
        {"dnf": "numactl-devel", "yum": "numactl-devel", "*": "libnuma-dev"},
        on_missing="skip",
    )
    if result.installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("numactl-devel")
    print(f"\n[dnf] numactl-devel specs: {info}")
    assert result.ok
    assert result.all_packages, "No packages accounted for"
    r = target_executor.run("find /usr/include -name 'numa.h' 2>/dev/null | head -1")
    assert r.ok and r.stdout.strip(), "numa.h not found after install"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_dnf_install_openssl_devel(target_executor):
    """Install openssl-devel with cross-distro dict spec; verify via specs()."""
    result = OS_DEPS().install(
        {
            "apt": "libssl-dev",
            "dnf": "openssl-devel",
            "yum": "openssl-devel",
            "zypper": "libopenssl-devel",
            "*": "libssl-dev",
        },
        on_missing="skip",
    )
    if result.installed:
        _session_installed.extend(result.installed)

    print(f"\n[dnf] openssl-devel installed: {result.installed}, present: {result.already_present}")
    assert result.ok and result.all_packages
    r = target_executor.run("find /usr/include -name 'openssl' -type d 2>/dev/null | head -1")
    assert r.ok and r.stdout.strip(), "openssl header dir not found"


# ---------------------------------------------------------------------------
# Linux / SLES (zypper) -- skips on non-zypper nodes
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_zypper_install_libdrm(target_executor):
    """Install libdrm-devel via zypper; skips on apt/dnf nodes."""
    result = OS_DEPS().install(
        {"apt": "libdrm-dev", "dnf": "libdrm-devel", "zypper": "libdrm-devel", "*": "libdrm-dev"},
        on_missing="skip",
    )
    if result.installed:
        _session_installed.extend(result.installed)

    print(f"\n[zypper] libdrm specs: installed={result.installed}, present={result.already_present}")
    assert result.ok and result.all_packages
    r = target_executor.run("find /usr/include -name 'drm.h' -path '*/drm/*' 2>/dev/null | head -1")
    assert r.ok and r.stdout.strip(), "drm.h not found after install"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_zypper_install_libnuma(target_executor):
    """Install libnuma-devel via zypper with cross-distro dict spec."""
    result = OS_DEPS().install(
        {
            "apt": "libnuma-dev",
            "dnf": "numactl-devel",
            "yum": "numactl-devel",
            "zypper": "libnuma-devel",
            "*": "libnuma-dev",
        },
        on_missing="skip",
    )
    if result.installed:
        _session_installed.extend(result.installed)

    print(f"\n[zypper] libnuma specs: installed={result.installed}, present={result.already_present}")
    assert result.ok and result.all_packages
    r = target_executor.run("find /usr/include -name 'numa.h' 2>/dev/null | head -1")
    assert r.ok and r.stdout.strip(), "numa.h not found after install"


# ---------------------------------------------------------------------------
# Error handling (xfail)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Installing a non-existent package must raise RuntimeError",
)
@pytest.mark.hw.cpu_only
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_invalid_package_raises_error(target_executor):
    """Non-existent package must raise RuntimeError."""
    OS_DEPS().install("rocmtest-this-package-does-not-exist-12345", on_missing="install")


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Installing a non-existent pip package must raise RuntimeError",
)
@pytest.mark.hw.cpu_only
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_invalid_pip_package_raises_error(target_executor):
    """Non-existent pip package must raise RuntimeError."""
    OS_DEPS().install("rocmtest-nonexistent-pip-package-12345", command="pip", on_missing="install")


# ---------------------------------------------------------------------------
# Python packages via pip
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_pip_install_packaging(target_executor):
    """Install 'packaging' via pip; verify import and specs()."""
    result = OS_DEPS().install("packaging", command="pip", extra_args=["--quiet"])

    assert result.ok and result.command == "pip"
    assert "packaging" in result.installed
    r = target_executor.run("python3 -c 'import packaging; print(packaging.__version__)'")
    assert r.ok and r.stdout.strip(), "packaging import failed"
    print(f"\n[pip] packaging version: {r.stdout.strip()}")


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_pip_install_with_extra_args(target_executor):
    """Install setuptools via pip with --upgrade extra_arg."""
    result = OS_DEPS().install("setuptools", command="pip", extra_args=["--quiet", "--upgrade"])

    assert result.ok and result.command == "pip"
    r = target_executor.run("python3 -c 'import setuptools; print(setuptools.__version__)'")
    assert r.ok and r.stdout.strip()
    print(f"\n[pip] setuptools version: {r.stdout.strip()}")


# ---------------------------------------------------------------------------
# Windows -- winget / choco / scoop (skips on Linux)
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_install_package(target_executor):
    """Install git via winget on Windows; falls back to apt/dnf on Linux."""
    result = OS_DEPS().install({"winget": "Git.Git", "*": "git"}, on_missing="skip")
    assert result.ok and result.all_packages
    assert result.version_summary()


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_choco_install_package():
    """Install cmake via choco on Windows; falls back on Linux."""
    result = OS_DEPS().install({"choco": "cmake", "*": "cmake"}, on_missing="skip")
    assert result.ok and result.all_packages
    assert result.version_summary()


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_scoop_install_package():
    """Install curl via scoop on Windows; falls back on Linux."""
    result = OS_DEPS().install({"scoop": "curl", "*": "curl"}, on_missing="skip")
    assert result.ok and result.all_packages
    assert result.version_summary()


# ---------------------------------------------------------------------------
# specs() API -- structured package info
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_specs_installed_package(target_executor):
    """specs() returns installed=True and version for a known-present package."""
    info = OS_DEPS().specs("git")
    print(f"\n[specs] git: {info}")
    assert info["git"]["installed"], "git should be installed"
    assert info["git"]["version"], "git version should be non-empty"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_specs_multiple_packages(target_executor):
    """specs() returns correct info for multiple packages in one call."""
    result = OS_DEPS().install({"apt": "pv", "dnf": "pv", "*": "pv"})
    if result.installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("git", "pv", "rocmtest-nonexistent-99999")
    print(f"\n[specs] multi-package: {info}")
    assert info["git"]["installed"]
    assert info["pv"]["installed"]
    assert not info["rocmtest-nonexistent-99999"]["installed"]
    assert info["rocmtest-nonexistent-99999"]["version"] is None


# ---------------------------------------------------------------------------
# uninstall() API -- plain name and version-pinned
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_uninstall_version_specific(target_executor):
    """Install pv, then uninstall with wrong/correct version; verify via specs()."""
    result = OS_DEPS().install({"apt": "pv", "dnf": "pv", "*": "pv"})
    if result.installed and "pv" not in _session_installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("pv")
    version = info["pv"]["version"]
    print(f"\n[uninstall] pv specs before: {info}")
    assert info["pv"]["installed"], "pv must be installed"

    # Wrong version -- should skip
    wrong = OS_DEPS().uninstall("pv==0.0.0")
    print(f"[uninstall] wrong version result: {wrong}")
    assert not wrong.get("pv==0.0.0"), "should NOT remove pv with wrong version"

    # Correct version -- should remove
    correct = OS_DEPS().uninstall(f"pv=={version}")
    print(f"[uninstall] correct version result: {correct}")
    assert correct.get(f"pv=={version}"), f"should remove pv=={version}"

    if "pv" in _session_installed:
        _session_installed.remove("pv")

    info_after = OS_DEPS().specs("pv")
    print(f"[uninstall] pv specs after: {info_after}")
    assert not info_after["pv"]["installed"], "pv should be uninstalled"


# ---------------------------------------------------------------------------
# upgrade() API -- to latest and to specific version
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_upgrade_to_latest(target_executor):
    """Upgrade tree to latest; verify still installed via specs()."""
    result = OS_DEPS().install({"apt": "tree", "dnf": "tree", "*": "tree"})
    if result.installed and "tree" not in _session_installed:
        _session_installed.extend(result.installed)

    before = OS_DEPS().specs("tree")
    print(f"\n[upgrade] tree before: {before}")

    upgraded = OS_DEPS().upgrade({"apt": "tree", "dnf": "tree", "*": "tree"})
    print(f"[upgrade] upgraded versions: {upgraded.versions}")

    after = OS_DEPS().specs("tree")
    print(f"[upgrade] tree after: {after}")
    assert after["tree"]["installed"], "tree should still be installed after upgrade"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_upgrade_to_specific_version(target_executor):
    """Upgrade jq to its current version (idempotent); verify via specs()."""
    result = OS_DEPS().install({"apt": "jq", "dnf": "jq", "*": "jq"})
    if result.installed and "jq" not in _session_installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("jq")
    current_ver = info["jq"]["version"]
    print(f"\n[upgrade specific] jq current: {info}")

    if current_ver:
        upgraded = OS_DEPS().upgrade(f"jq=={current_ver}")
        print(f"[upgrade specific] upgraded to {current_ver}: {upgraded.versions}")
        assert upgraded.versions.get("jq") or upgraded.installed


# ---------------------------------------------------------------------------
# src= URL patterns
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_src_deb_url_install(target_executor):
    """Install figlet from a direct .deb URL; verify via specs()."""
    _figlet_deb_url = (
        "http://archive.ubuntu.com/ubuntu/pool/universe/f/figlet/"
        "figlet_2.2.5-3_amd64.deb"
    )
    result = OS_DEPS().install({"apt": "figlet", "*": "figlet"}, src=_figlet_deb_url)
    if result.installed:
        _session_installed.extend(result.installed)

    info = OS_DEPS().specs("figlet")
    print(f"\n[src=.deb] figlet specs: {info}")
    assert info["figlet"]["installed"], "figlet not installed after .deb URL install"
    print(f"[src=.deb] version: {info['figlet']['version']}")


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_src_repo_url_install(target_executor):
    """Install from a repo source URL via sources= dict; validate install ran."""
    _rocm_apt_sources = {
        "apt": "deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ focal main",
        "dnf": "https://repo.radeon.com/rocm/rhel9/main/",
    }
    result = OS_DEPS().install(
        {"apt": "rocm-smi", "dnf": "rocm-smi", "*": "rocm-smi"},
        sources=_rocm_apt_sources,
    )
    if result.installed:
        _session_installed.extend(result.installed)

    print(f"\n[src=repo] rocm-smi: installed={result.installed}, present={result.already_present}")
    assert result.installed or result.already_present, "rocm-smi not installed via repo URL"
    info = OS_DEPS().specs("rocm-smi")
    print(f"[src=repo] specs (best-effort): {info}")
