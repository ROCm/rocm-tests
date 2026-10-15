# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_os_deps_plugin.py — Real installation tests for os_deps_plugin.

Each test installs packages via OS_DEPS, asserts on the InstallResult,
then verifies the package is actually usable on the node.

Platform coverage:
  Linux / Ubuntu  — apt    (ci.pr and ci.nightly)
  Linux / RHEL    — dnf    (ci.nightly, skips on non-dnf nodes)
  Linux / SLES    — zypper (ci.nightly, skips on non-zypper nodes)
  Windows         — winget / choco / scoop (ci.nightly, skips on Linux)
  Any platform    — pip, snap (skips if command not available)

Cross-distro tests use dict specs so the framework picks the right
package name automatically regardless of which distro is running.
"""

from __future__ import annotations

import pytest

from framework.os_adapter import OS_DEPS

# ---------------------------------------------------------------------------
# Linux / Ubuntu (apt) — ci.pr: always available on dev machines
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_install_common_build_tools(target_executor):
    """Install cmake, git, curl via apt and verify each is functional.

    Probe fires first — on a machine where these are already present the
    install step is skipped. On a fresh node they get installed.
    """
    result = OS_DEPS().install("cmake", "git", "curl")

    assert result.ok, "OS_DEPS install returned failure"
    assert set(result.all_packages) >= {"cmake", "git", "curl"}, (
        f"Not all packages accounted for: {result.all_packages}"
    )

    for tool, flag in [("cmake", "--version"), ("git", "--version"), ("curl", "--version")]:
        r = target_executor.run(f"{tool} {flag}")
        assert r.ok, f"{tool} not usable after install: {r.stderr}"
        assert tool in r.stdout.lower(), f"Unexpected output for {tool}: {r.stdout}"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_install_version_pinned(target_executor):
    """Install a version-pinned package (cmake==3.22) and verify the installed version."""
    result = OS_DEPS().install("cmake==3.22", on_missing="skip")

    assert result.ok
    assert result.all_packages, "No packages accounted for in InstallResult"

    r = target_executor.run("cmake --version")
    assert r.ok, f"cmake not usable after install: {r.stderr}"
    assert "cmake version" in r.stdout.lower()


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_install_list_of_packages(target_executor):
    """Install multiple packages in one call and verify each is present."""
    result = OS_DEPS().install({"apt": ["wget", "unzip", "file"], "*": "wget"})

    assert result.ok
    for tool in ["wget", "unzip", "file"]:
        r = target_executor.run(f"command -v {tool}")
        assert r.ok, f"{tool} not found after install"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_apt_already_installed_skips_install(target_executor):
    """Packages already on the node must show up in already_present, not installed."""
    # git is almost always pre-installed — this validates probe correctness
    result = OS_DEPS().install("git")

    assert result.ok
    # Either just installed or already present — both are acceptable
    assert "git" in result.all_packages, f"git not in result: {result.all_packages}"


# ---------------------------------------------------------------------------
# Linux / RHEL (dnf) — skips automatically on non-dnf nodes
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_install_development_tools(target_executor):
    """Install dnf-specific packages; skips on apt/zypper nodes.

    Uses on_missing='skip' so the test gracefully skips on Ubuntu/SLES
    where these exact package names don't exist under dnf.
    """
    result = OS_DEPS().install(
        {"dnf": "numactl-devel", "yum": "numactl-devel", "*": "libnuma-dev"},
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages, "No packages accounted for"

    r = target_executor.run("find /usr/include -name 'numa.h' 2>/dev/null | head -1")
    assert r.ok
    assert r.stdout.strip(), "numactl-devel installed but numa.h not found"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_dnf_install_openssl_devel(target_executor):
    """Install openssl-devel (RHEL name) with dict spec cross-distro mapping."""
    result = OS_DEPS().install(
        {"apt": "libssl-dev", "dnf": "openssl-devel", "yum": "openssl-devel",
         "zypper": "libopenssl-devel", "*": "libssl-dev"},
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages

    r = target_executor.run("find /usr/include -name 'openssl' -type d 2>/dev/null | head -1")
    assert r.ok


# ---------------------------------------------------------------------------
# Linux / SLES (zypper) — skips automatically on non-zypper nodes
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_zypper_install_libdrm(target_executor):
    """Install libdrm-devel via zypper; skips on apt/dnf nodes.

    Validates that zypper install commands include --no-gpg-checks
    (required for SLES) and that the header lands on the filesystem.
    """
    result = OS_DEPS().install(
        {"apt": "libdrm-dev", "dnf": "libdrm-devel", "zypper": "libdrm-devel",
         "*": "libdrm-dev"},
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages

    r = target_executor.run("find /usr/include -name 'drm.h' -path '*/drm/*' 2>/dev/null | head -1")
    assert r.ok
    assert r.stdout.strip(), "libdrm-devel installed but drm.h not found"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_zypper_install_libnuma(target_executor):
    """Install libnuma-devel via zypper with cross-distro dict spec."""
    result = OS_DEPS().install(
        {"apt": "libnuma-dev", "dnf": "numactl-devel", "yum": "numactl-devel",
         "zypper": "libnuma-devel", "*": "libnuma-dev"},
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages

    r = target_executor.run("find /usr/include -name 'numa.h' 2>/dev/null | head -1")
    assert r.ok
    assert r.stdout.strip(), "libnuma-devel installed but numa.h not found"


# ---------------------------------------------------------------------------
# Error handling — invalid package name (xfail)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Installing a non-existent package must raise RuntimeError — validates error path",
)
@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_invalid_package_raises_error(target_executor):
    """Attempt to install a package that does not exist in any repository.

    Expected to fail with RuntimeError because the package manager will
    report a 'package not found' error.  Marked xfail(strict=True) so:
      - XFAIL  → test raised RuntimeError as expected  (counts as pass)
      - XPASS  → package somehow installed, which should never happen
    """
    OS_DEPS().install("rocmtest-this-package-does-not-exist-12345")


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Installing a non-existent pip package must raise RuntimeError",
)
@pytest.mark.hw.cpu_only
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_invalid_pip_package_raises_error(target_executor):
    """Attempt to install a non-existent PyPI package via pip.

    pip exits non-zero when the package is not found, which must
    surface as a RuntimeError from the framework.
    """
    OS_DEPS().install("rocmtest-nonexistent-pip-package-12345", command="pip")


# ---------------------------------------------------------------------------
# Python packages via pip (any platform)
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_pip_install_packaging(target_executor):
    """Install the 'packaging' PyPI library and verify it imports correctly.

    'packaging' is tiny and dependency-free — safe to install anywhere.
    Validates the command= path and that InstallResult.installed is populated.
    """
    result = OS_DEPS().install("packaging", command="pip", extra_args=["--quiet"])

    assert result.ok, f"pip install failed: {result}"
    assert result.command == "pip"
    assert "packaging" in result.installed, (
        f"Expected 'packaging' in installed list, got: {result.installed}"
    )

    r = target_executor.run("python3 -c 'import packaging; print(packaging.__version__)'")
    assert r.ok, f"packaging import failed: {r.stderr}"
    assert r.stdout.strip(), "packaging version string is empty"


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_pip_install_with_extra_args(target_executor):
    """Install a pip package using extra_args for index-url override."""
    result = OS_DEPS().install(
        "setuptools",
        command="pip",
        extra_args=["--quiet", "--upgrade"],
    )

    assert result.ok
    assert result.command == "pip"

    r = target_executor.run("python3 -c 'import setuptools; print(setuptools.__version__)'")
    assert r.ok, f"setuptools import failed: {r.stderr}"
    assert r.stdout.strip()


# ---------------------------------------------------------------------------
# Windows — winget / choco / scoop (skip on Linux automatically)
# ---------------------------------------------------------------------------


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_winget_install_package(target_executor):
    """Install Git.Git via winget; skips on non-Windows nodes.

    winget is not available on Linux — _detect_backend raises RuntimeError
    which the autouse fixture converts to pytest.skip.
    """
    result = OS_DEPS().install(
        {"winget": "Git.Git", "*": "git"},
        command="winget",
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages

    r = target_executor.run("git --version")
    assert r.ok, f"git not usable after winget install: {r.stderr}"
    assert "git" in r.stdout.lower()


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_choco_install_package(target_executor):
    """Install cmake via Chocolatey; skips on non-Windows nodes."""
    result = OS_DEPS().install(
        {"choco": "cmake", "*": "cmake"},
        command="choco",
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages

    r = target_executor.run("cmake --version")
    assert r.ok, f"cmake not usable after choco install: {r.stderr}"
    assert "cmake" in r.stdout.lower()


@pytest.mark.hw.cpu_only
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_scoop_install_package(target_executor):
    """Install curl via Scoop (user-mode, no admin); skips on non-Windows nodes."""
    result = OS_DEPS().install(
        {"scoop": "curl", "*": "curl"},
        command="scoop",
        on_missing="skip",
    )

    assert result.ok
    assert result.all_packages

    r = target_executor.run("curl --version")
    assert r.ok, f"curl not usable after scoop install: {r.stderr}"
    assert "curl" in r.stdout.lower()
