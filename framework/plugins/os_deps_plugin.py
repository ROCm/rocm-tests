# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
os_deps_plugin.py — OS package auto-install for rocm-tests.

Self-contained: copy this single file to use in another repo.

Public API (import from framework.os_adapter for convenience):
    from framework.os_adapter import OS_DEPS

Usage:
    @OS_DEPS("cmake", "git")
    @OS_DEPS("cmake==3.26")
    @OS_DEPS({"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"})
    @OS_DEPS({"apt": ["libssl-dev", "libdrm-dev"], "dnf": ["openssl-devel", "libdrm-devel"]})
    @OS_DEPS("pkg", command="pip")
    @OS_DEPS("pkg", command="pip", extra_args=["--index-url", "https://..."])
    @OS_DEPS("pkg", src="https://my-repo/")
    @OS_DEPS("pkg", sources={"apt": "deb https://...", "dnf": "https://..."})

    class TestFoo:
        __OS_DEPS__ = OS_DEPS("cmake", "git")   # class-level

    __OS_DEPS__ = OS_DEPS("cmake")               # module-level

    OS_DEPS().install("cmake")                   # mid-test; same arg forms

pkg_spec forms:
    "cmake"              plain name, auto-detect PM
    "cmake==3.26"        version pin (translated per PM)
    {"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"}
    {"apt": ["libssl-dev", "libdrm-dev"], "dnf": ["openssl-devel"]}

command= values:
    None (default)  auto-detect OS PM
    "pip" / "snap" / "conda" / any string

extra_args=:
    Extra flags passed verbatim to the package manager.

on_missing (env: ROCM_TEST_PKG_ON_MISSING, default "install"):
    "install" / "skip" / "fail" / "probe"

Resolution priority: function.__OS_DEPS__ > class.__OS_DEPS__ > module.__OS_DEPS__

Platform support:
    Linux:   apt, dnf, yum, zypper, apk, pacman (auto-detected)
    Windows: winget, choco, scoop (auto-detected; bootstrapped if missing)
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from dataclasses import dataclass, field
import hashlib
import logging
import os
import shlex
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Executor context — set by autouse fixture before every test
# ---------------------------------------------------------------------------
_current_executor: AbstractExecutor | None = None

_ENV_POLICY = os.environ.get("ROCM_TEST_PKG_ON_MISSING", "install").lower()

_BACKEND_ATTR = "_os_deps_backend"
_SUDO_ATTR = "_os_deps_sudo_prefix"

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

PkgSpec = str | dict


@dataclass
class InstallResult:
    """Result of an OS_DEPS installation attempt.

    Attributes:
        installed:      Packages that were missing and successfully installed.
        already_present: Packages already on the node (probe passed).
        versions:       {pkg: version} for all accounted packages.
        dry_run:        True when running in DryRun mode.
        command:        Custom installer used ("pip", "snap", …) or None for OS PM.
    """

    installed: list[str] = field(default_factory=list)
    already_present: list[str] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    dry_run: bool = False
    command: str | None = None

    @property
    def ok(self) -> bool:
        """True when all requested packages are available (no RuntimeError raised)."""
        return True

    @property
    def all_packages(self) -> list[str]:
        """installed + already_present."""
        return self.installed + self.already_present

    def version_summary(self) -> str:
        """Human-readable string: 'cmake=3.28.3, git=2.43.0'."""
        if not self.versions:
            return ", ".join(self.all_packages)
        return ", ".join(f"{p}={self.versions[p]}" if p in self.versions else p for p in self.all_packages)


# ---------------------------------------------------------------------------
# Internal spec
# ---------------------------------------------------------------------------


@dataclass
class _InstallSpec:
    pkg_specs: tuple
    src: str | None = None
    sources: dict[str, str] | None = None
    command: str | None = None
    extra_args: list[str] = field(default_factory=list)
    on_missing: str = _ENV_POLICY


# ---------------------------------------------------------------------------
# OS_DEPS — single public API class
# ---------------------------------------------------------------------------


class OS_DEPS:  # noqa: N801
    """Declare and install OS package dependencies for tests.

    See module docstring for full usage and all argument forms.
    """

    def __init__(
        self,
        *pkg_specs: PkgSpec,
        src: str | None = None,
        sources: dict[str, str] | None = None,
        command: str | None = None,
        extra_args: list[str] | None = None,
        on_missing: str = _ENV_POLICY,
    ) -> None:
        self._spec = _InstallSpec(
            pkg_specs=pkg_specs,
            src=src,
            sources=sources,
            command=command,
            extra_args=extra_args or [],
            on_missing=on_missing,
        )

    def __call__(self, fn_or_cls):
        """Apply as decorator on a function, method, or class."""
        fn_or_cls.__OS_DEPS__ = self._spec
        return fn_or_cls

    def install(
        self,
        *pkg_specs: PkgSpec,
        src: str | None = None,
        sources: dict[str, str] | None = None,
        command: str | None = None,
        extra_args: list[str] | None = None,
        on_missing: str = _ENV_POLICY,
    ) -> InstallResult:
        """Install packages immediately using the current test's executor.

        Reads executor from the autouse fixture context — no argument needed.

        Returns:
            InstallResult — assert result.ok or check result.installed / result.already_present.
        """
        if _current_executor is None:
            raise RuntimeError(
                "OS_DEPS().install() must be called from within a test function. "
                "No executor context is available outside a test."
            )
        spec = _InstallSpec(
            pkg_specs=pkg_specs,
            src=src,
            sources=sources,
            command=command,
            extra_args=extra_args or [],
            on_missing=on_missing,
        )
        return _execute_install(_current_executor, spec)


# ---------------------------------------------------------------------------
# Backend descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Backend:
    key: str
    detect_cmd: str
    install_fn: Callable  # (pkgs, sudo, src, extra_args) -> str
    probe_cmds: dict[str, Callable[[str], str]]  # pm_key → probe command builder
    bootstrap_fn: Callable | None = None
    needs_sudo: bool = True
    platform: str = "linux"


# ---------------------------------------------------------------------------
# Version parsing helpers (shared by both PM classes)
# ---------------------------------------------------------------------------


def _parse_version(pkg: str) -> tuple[str, str | None]:
    """'cmake==3.26' → ('cmake', '3.26'); 'cmake' → ('cmake', None)."""
    if "==" in pkg:
        name, ver = pkg.split("==", 1)
        return name.strip(), ver.strip()
    return pkg.strip(), None


def _versioned(name: str, version: str | None, pm_key: str) -> str:
    """Translate a version pin to the correct syntax for a given PM."""
    if not version:
        return name
    if pm_key == "apt":
        return f"{name}={version}*"
    if pm_key in ("dnf", "yum"):
        return f"{name}-{version}"
    if pm_key in ("zypper", "apk", "pacman"):
        return f"{name}={version}"
    return f"{name}=={version}"  # pip, snap, conda, generic


# ---------------------------------------------------------------------------
# _LinuxPM — all Linux package manager logic
# ---------------------------------------------------------------------------


class _LinuxPM:
    """Linux PM logic: apt, dnf, yum, zypper, apk, pacman."""

    # Probe commands — each exits 0 if the package is installed, 1 if not.
    # Embedded inside bash -c '...' so must not contain single quotes.
    PROBE_CMDS: dict[str, Callable[[str], str]] = {
        "apt": lambda pkg: f"dpkg-query -W {shlex.quote(_parse_version(pkg)[0])} >/dev/null 2>&1",
        "dnf": lambda pkg: f"rpm -q {shlex.quote(_parse_version(pkg)[0])} >/dev/null 2>&1",
        "yum": lambda pkg: f"rpm -q {shlex.quote(_parse_version(pkg)[0])} >/dev/null 2>&1",
        "zypper": lambda pkg: f"rpm -q {shlex.quote(_parse_version(pkg)[0])} >/dev/null 2>&1",
        "apk": lambda pkg: f"apk info -e {shlex.quote(_parse_version(pkg)[0])} >/dev/null 2>&1",
        "pacman": lambda pkg: f"pacman -Q {shlex.quote(_parse_version(pkg)[0])} >/dev/null 2>&1",
    }

    # ── Install command builders ──────────────────────────────────────────────

    @staticmethod
    def apt_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            if src.endswith(".deb"):
                return (
                    f"bash -c '{sudo}wget -q -O /tmp/_os_deps.deb {shlex.quote(src)} && "
                    f"{sudo}dpkg -i /tmp/_os_deps.deb && {sudo}apt-get install -f -y 2>&1'"
                )
            repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
            repo_file = f"/etc/apt/sources.list.d/os_deps_{repo_hash}.list"
            return (
                f"bash -c 'grep -qF {shlex.quote(src)} {repo_file} 2>/dev/null || "
                f"echo {shlex.quote(src)} | {sudo}tee {repo_file} >/dev/null; "
                f"{sudo}apt-get update -q -y && "
                f"{sudo}apt-get install -y --no-install-recommends {pkg_str}{extra} 2>&1'"
            )
        return f"bash -c '{sudo}apt-get install -y --no-install-recommends {pkg_str}{extra} 2>&1'"

    @staticmethod
    def dnf_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            if src.endswith(".rpm"):
                return f"bash -c '{sudo}dnf install -y {shlex.quote(src)}{extra} 2>&1'"
            repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
            repo_file = f"/etc/yum.repos.d/os_deps_{repo_hash}.repo"
            content = f"[os_deps_{repo_hash}]\\nname=os_deps custom\\nbaseurl={src}\\nenabled=1\\ngpgcheck=0"
            return (
                f"bash -c 'echo -e {shlex.quote(content)} | {sudo}tee {repo_file} >/dev/null && "
                f"{sudo}dnf install -y {pkg_str}{extra} 2>&1'"
            )
        return f"bash -c '{sudo}dnf install -y {pkg_str}{extra} 2>&1'"

    @staticmethod
    def yum_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            if src.endswith(".rpm"):
                return f"bash -c '{sudo}yum install -y {shlex.quote(src)}{extra} 2>&1'"
            repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
            repo_file = f"/etc/yum.repos.d/os_deps_{repo_hash}.repo"
            content = f"[os_deps_{repo_hash}]\\nname=os_deps custom\\nbaseurl={src}\\nenabled=1\\ngpgcheck=0"
            return (
                f"bash -c 'echo -e {shlex.quote(content)} | {sudo}tee {repo_file} >/dev/null && "
                f"{sudo}yum install -y {pkg_str}{extra} 2>&1'"
            )
        return f"bash -c '{sudo}yum install -y {pkg_str}{extra} 2>&1'"

    @staticmethod
    def zypper_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
            alias = f"os_deps_{repo_hash}"
            return (
                f"bash -c '{sudo}zypper --non-interactive addrepo --no-gpgcheck "
                f"{shlex.quote(src)} {alias} 2>&1; "
                f"{sudo}zypper --non-interactive refresh 2>&1 && "
                f"{sudo}zypper --non-interactive install -y --no-recommends --no-gpg-checks "
                f"{pkg_str}{extra} 2>&1'"
            )
        return (
            f"bash -c '{sudo}zypper --non-interactive install -y "
            f"--no-recommends --no-gpg-checks {pkg_str}{extra} 2>&1'"
        )

    @staticmethod
    def apk_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            return f"bash -c '{sudo}apk add --no-cache --repository {shlex.quote(src)} {pkg_str}{extra} 2>&1'"
        return f"bash -c '{sudo}apk add --no-cache {pkg_str}{extra} 2>&1'"

    @staticmethod
    def pacman_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            logger.warning("os_deps: pacman does not support src= URL; ignoring %s", src)
        return f"bash -c '{sudo}pacman -S --noconfirm {pkg_str}{extra} 2>&1'"

    # ── Probe orchestration ───────────────────────────────────────────────────

    @classmethod
    def probe_missing(cls, executor, backend: _Backend, pkgs: list[str]) -> list[str]:
        """Batch all probes into one bash compound command — one round-trip."""
        parts = []
        for pkg in pkgs:
            probe = backend.probe_cmds[backend.key](pkg)
            escaped = pkg.replace("'", "\\'")
            parts.append(f"( {probe} && echo OK:{escaped} || echo MISS:{escaped} )")

        result = executor.run("bash -c '" + "; ".join(parts) + "'")

        missing, present = [], []
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("OK:"):
                present.append(line[3:])
            elif line.startswith("MISS:"):
                missing.append(line[5:])

        if not missing and not present:
            logger.debug("os_deps: probe returned no tokens — treating all as missing: %s", pkgs)
            return list(pkgs)

        if present:
            logger.debug("os_deps: probe PRESENT — %s", present)
        if missing:
            logger.debug("os_deps: probe MISSING — %s", missing)
        return missing

    # ── Version queries ───────────────────────────────────────────────────────

    @classmethod
    def query_versions(cls, executor, backend: _Backend, pkgs: list[str]) -> dict[str, str]:  # noqa: C901
        """Query installed version per package using dpkg-query / rpm -q / etc."""
        versions: dict[str, str] = {}
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)

        if backend.key == "apt":
            result = executor.run(f"dpkg-query -W -f='${{Package}}\\t${{Version}}\\n' {pkg_str} 2>/dev/null")
            for line in result.stdout.splitlines():
                parts = line.strip().split("\t", 1)
                if len(parts) == 2 and parts[1].strip():
                    versions[parts[0].strip()] = parts[1].strip()

        elif backend.key in ("dnf", "yum", "zypper"):
            result = executor.run(
                f"rpm -q --queryformat '%{{NAME}}\\t%{{VERSION}}-%{{RELEASE}}\\n' {pkg_str} 2>/dev/null"
            )
            for line in result.stdout.splitlines():
                parts = line.strip().split("\t", 1)
                if len(parts) == 2 and "not installed" not in parts[1]:
                    versions[parts[0].strip()] = parts[1].strip()

        elif backend.key == "apk":
            result = executor.run(f"apk info -v {pkg_str} 2>/dev/null")
            for line in result.stdout.splitlines():
                line = line.strip()
                for pkg in pkgs:
                    base, _ = _parse_version(pkg)
                    if line.startswith(base + "-"):
                        versions[base] = line[len(base) + 1 :]

        elif backend.key == "pacman":
            result = executor.run(f"pacman -Q {pkg_str} 2>/dev/null")
            for line in result.stdout.splitlines():
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    versions[parts[0]] = parts[1]

        return versions


# ---------------------------------------------------------------------------
# _WindowsPM — all Windows package manager logic
# ---------------------------------------------------------------------------


class _WindowsPM:
    """Windows PM logic: winget, choco, scoop."""

    # Probe commands — run via cmd.exe; exit 0 if installed, non-zero if not.
    PROBE_CMDS: dict[str, Callable[[str], str]] = {
        "winget": lambda pkg: f"winget list --id {shlex.quote(_parse_version(pkg)[0])} --exact >NUL 2>&1",
        "choco": lambda pkg: f"choco list --local-only --exact {shlex.quote(_parse_version(pkg)[0])} >NUL 2>&1",
        "scoop": lambda pkg: f"scoop list {shlex.quote(_parse_version(pkg)[0])} >NUL 2>&1",
    }

    # ── Install command builders ──────────────────────────────────────────────

    @staticmethod
    def winget_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        base = "winget install --silent --accept-package-agreements --accept-source-agreements"
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        cmds = []
        for pkg in pkgs:
            if src:
                cmds.append(f"{base} --source {shlex.quote(src)} --id {shlex.quote(pkg)}{extra}")
            else:
                cmds.append(f"{base} --id {shlex.quote(pkg)}{extra}")
        return " && ".join(cmds)

    @staticmethod
    def choco_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        pkg_str = " ".join(shlex.quote(p) for p in pkgs)
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            return f"choco install -y {pkg_str} --source {shlex.quote(src)}{extra}"
        return f"choco install -y {pkg_str}{extra}"

    @staticmethod
    def scoop_install(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
        extra = (" " + " ".join(extra_args)) if extra_args else ""
        if src:
            bucket = f"os_deps_{hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]}"
            add = f"scoop bucket add {shlex.quote(bucket)} {shlex.quote(src)}"
            install = " && ".join(f"scoop install {shlex.quote(p)}{extra}" for p in pkgs)
            return f"{add} && {install}"
        return " && ".join(f"scoop install {shlex.quote(p)}{extra}" for p in pkgs)

    # ── Bootstrappers ─────────────────────────────────────────────────────────

    @staticmethod
    def bootstrap_winget(executor) -> None:
        logger.info("os_deps: winget not found — bootstrapping App Installer")
        url = (
            "https://github.com/microsoft/winget-cli/releases/latest/download/"
            "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle"
        )
        result = executor.run(
            f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"Add-AppxPackage -Path '{url}'\""
        )
        if not result.ok:
            raise RuntimeError(
                f"os_deps: failed to bootstrap winget.\n"
                f"exit_code: {result.exit_code}\nstderr: {result.stderr[-500:]}"
            )

    @staticmethod
    def bootstrap_choco(executor) -> None:
        logger.info("os_deps: Chocolatey not found — bootstrapping")
        result = executor.run(
            "powershell -NoProfile -ExecutionPolicy Bypass -Command "
            '"Set-ExecutionPolicy Bypass -Scope Process -Force; '
            "[System.Net.ServicePointManager]::SecurityProtocol = "
            "[System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
            "iex ((New-Object System.Net.WebClient).DownloadString("
            "'https://community.chocolatey.org/install.ps1'))\""
        )
        if not result.ok:
            raise RuntimeError(
                f"os_deps: failed to bootstrap Chocolatey.\n"
                f"exit_code: {result.exit_code}\nstderr: {result.stderr[-500:]}"
            )

    @staticmethod
    def bootstrap_scoop(executor) -> None:
        logger.info("os_deps: Scoop not found — bootstrapping")
        result = executor.run('powershell -NoProfile -ExecutionPolicy RemoteSigned -Command "irm get.scoop.sh | iex"')
        if not result.ok:
            raise RuntimeError(
                f"os_deps: failed to bootstrap Scoop.\n"
                f"exit_code: {result.exit_code}\nstderr: {result.stderr[-500:]}"
            )

    # ── Probe orchestration ───────────────────────────────────────────────────

    @classmethod
    def probe_missing(cls, executor, backend: _Backend, pkgs: list[str]) -> list[str]:
        """Probe each package individually via cmd.exe (no bash on Windows)."""
        missing, present = [], []
        for pkg in pkgs:
            probe = backend.probe_cmds[backend.key](pkg)
            result = executor.run(probe)
            if result.ok:
                present.append(pkg)
            else:
                missing.append(pkg)
        if present:
            logger.debug("os_deps: probe PRESENT — %s", present)
        if missing:
            logger.debug("os_deps: probe MISSING — %s", missing)
        return missing

    # ── Version queries ───────────────────────────────────────────────────────

    @classmethod
    def query_versions(cls, executor, backend: _Backend, pkgs: list[str]) -> dict[str, str]:  # noqa: C901
        """Query installed version per package via winget/choco/scoop list."""
        versions: dict[str, str] = {}
        for pkg in pkgs:
            base, _ = _parse_version(pkg)
            if backend.key == "winget":
                result = executor.run(f"winget list --id {shlex.quote(base)} --exact 2>NUL")
                for line in result.stdout.splitlines():
                    if base.lower() in line.lower():
                        cols = line.split()
                        if len(cols) >= 3:
                            versions[base] = cols[2]  # Name  Id  Version  …
                        break
            elif backend.key == "choco":
                result = executor.run(f"choco list --local-only --exact {shlex.quote(base)} 2>NUL")
                for line in result.stdout.splitlines():
                    if base.lower() in line.lower():
                        parts = line.strip().split(None, 1)
                        if len(parts) == 2:
                            versions[base] = parts[1]
                        break
            elif backend.key == "scoop":
                result = executor.run(f"scoop list {shlex.quote(base)} 2>NUL")
                for line in result.stdout.splitlines():
                    if base.lower() in line.lower():
                        cols = line.split()
                        if len(cols) >= 2:
                            versions[base] = cols[1]
                        break
        return versions


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_LPC = _LinuxPM.PROBE_CMDS  # shorthand for the registry below
_WPC = _WindowsPM.PROBE_CMDS


def _linux(key: str, detect: str, install_fn) -> _Backend:
    return _Backend(
        key=key, detect_cmd=detect, install_fn=install_fn, probe_cmds=_LPC, needs_sudo=True, platform="linux"
    )


_LINUX_BACKENDS: list[_Backend] = [
    _linux("apt", "command -v apt-get", _LinuxPM.apt_install),
    _linux("dnf", "command -v dnf", _LinuxPM.dnf_install),
    _linux("yum", "command -v yum", _LinuxPM.yum_install),
    _linux("zypper", "command -v zypper", _LinuxPM.zypper_install),
    _linux("apk", "command -v apk", _LinuxPM.apk_install),
    _linux("pacman", "command -v pacman", _LinuxPM.pacman_install),
]

_WINDOWS_BACKENDS: list[_Backend] = [
    _Backend(
        key="winget",
        detect_cmd="winget --version",
        install_fn=_WindowsPM.winget_install,
        probe_cmds=_WPC,
        bootstrap_fn=_WindowsPM.bootstrap_winget,
        needs_sudo=False,
        platform="windows",
    ),
    _Backend(
        key="choco",
        detect_cmd="choco --version",
        install_fn=_WindowsPM.choco_install,
        probe_cmds=_WPC,
        bootstrap_fn=_WindowsPM.bootstrap_choco,
        needs_sudo=False,
        platform="windows",
    ),
    _Backend(
        key="scoop",
        detect_cmd="scoop --version",
        install_fn=_WindowsPM.scoop_install,
        probe_cmds=_WPC,
        bootstrap_fn=_WindowsPM.bootstrap_scoop,
        needs_sudo=False,
        platform="windows",
    ),
]


# ---------------------------------------------------------------------------
# Top-level dispatcher functions (thin wrappers over PM classes)
# ---------------------------------------------------------------------------


def _probe_missing(executor, backend: _Backend, pkgs: list[str]) -> list[str]:
    """Dispatch to _LinuxPM or _WindowsPM probe_missing."""
    if not pkgs:
        return []
    if backend.platform == "windows":
        return _WindowsPM.probe_missing(executor, backend, pkgs)
    return _LinuxPM.probe_missing(executor, backend, pkgs)


def _get_installed_versions(executor, backend: _Backend, pkgs: list[str]) -> dict[str, str]:
    """Dispatch to _LinuxPM or _WindowsPM query_versions."""
    if not pkgs:
        return {}
    if backend.platform == "windows":
        return _WindowsPM.query_versions(executor, backend, pkgs)
    return _LinuxPM.query_versions(executor, backend, pkgs)


def _resolve_src(backend: _Backend, src: str | None, sources: dict[str, str] | None) -> str | None:
    if sources and backend.key in sources:
        return sources[backend.key]
    return src


def _resolve_pkgs(pkg_specs: tuple, backend: _Backend) -> list[str]:
    """Resolve all pkg_specs to PM-specific package name strings."""
    result = []
    for spec in pkg_specs:
        if isinstance(spec, str):
            name, ver = _parse_version(spec)
            result.append(_versioned(name, ver, backend.key))
        elif isinstance(spec, dict):
            raw = spec.get(backend.key) or spec.get("*")
            if not raw:
                raise RuntimeError(
                    f"os_deps: no mapping for backend '{backend.key}' in spec {spec!r}. "
                    f"Add a '{backend.key}' key or a '*' fallback."
                )
            items = [raw] if isinstance(raw, str) else list(raw)
            for item in items:
                name, ver = _parse_version(item)
                result.append(_versioned(name, ver, backend.key))
        else:
            raise TypeError(f"pkg_spec must be str or dict, got {type(spec).__name__!r}")
    return result


# ---------------------------------------------------------------------------
# Backend detection (cached per executor instance)
# ---------------------------------------------------------------------------


def _detect_backend(executor) -> _Backend:
    cached = getattr(executor, _BACKEND_ATTR, None)
    if isinstance(cached, _Backend):
        logger.debug("os_deps: using cached backend '%s'", cached.key)
        return cached
    logger.debug("os_deps: probing for package manager on executor")
    for backend in _LINUX_BACKENDS:
        result = executor.run(f"bash -c '{backend.detect_cmd} >/dev/null 2>&1 && echo OK || echo NO'")
        if result.ok and "OK" in result.stdout:
            _cache_backend(executor, backend)
            logger.debug("os_deps: detected package manager: '%s'", backend.key)
            return backend
    for backend in _WINDOWS_BACKENDS:
        result = executor.run(f"{backend.detect_cmd} >NUL 2>&1 && echo OK || echo NO")
        if result.ok and "OK" in result.stdout:
            _cache_backend(executor, backend)
            logger.debug("os_deps: detected package manager: '%s'", backend.key)
            return backend
        if backend.bootstrap_fn is not None:
            logger.info("os_deps: '%s' not found — attempting bootstrap", backend.key)
            try:
                backend.bootstrap_fn(executor)
                result2 = executor.run(f"{backend.detect_cmd} >NUL 2>&1 && echo OK || echo NO")
                if result2.ok and "OK" in result2.stdout:
                    _cache_backend(executor, backend)
                    logger.info("os_deps: bootstrapped and using '%s'", backend.key)
                    return backend
            except RuntimeError as exc:
                logger.warning("os_deps: bootstrap of '%s' failed: %s", backend.key, exc)
    logger.error(
        "os_deps: no supported package manager found. "
        "Tried: apt-get, dnf, yum, zypper, apk, pacman (Linux); "
        "winget, choco, scoop (Windows)."
    )
    raise RuntimeError(
        "os_deps: no supported package manager found. "
        "Tried: apt-get, dnf, yum, zypper, apk, pacman (Linux); "
        "winget, choco, scoop (Windows, with auto-bootstrap)."
    )


def _cache_backend(executor, backend: _Backend) -> None:
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(executor, _BACKEND_ATTR, backend)


# ---------------------------------------------------------------------------
# Privilege escalation (cached per executor instance)
# ---------------------------------------------------------------------------


def _resolve_sudo_prefix(executor, backend: _Backend) -> str:
    """Return the correct sudo prefix — probes once and caches the result.

    Returns "sudo -n " on fleet nodes, "" on Docker root or Windows.
    """
    if not backend.needs_sudo:
        return ""
    cached = getattr(executor, _SUDO_ATTR, None)
    if isinstance(cached, str):
        return cached
    result = executor.run("bash -c 'command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null && echo OK || echo NO'")
    prefix = "sudo -n " if (result.ok and "OK" in result.stdout) else ""
    if prefix:
        logger.debug("os_deps: sudo available — will use 'sudo -n' prefix")
    else:
        logger.debug("os_deps: sudo not available — running without sudo (root or container)")
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(executor, _SUDO_ATTR, prefix)
    return prefix


# ---------------------------------------------------------------------------
# Install execution
# ---------------------------------------------------------------------------


def _run_install_cmd(executor, backend: _Backend, pkgs: list[str], src: str | None, extra_args: list[str]) -> None:
    """Run the OS PM install command with the correct privilege prefix."""
    sudo = _resolve_sudo_prefix(executor, backend)
    cmd = backend.install_fn(pkgs, sudo, src, extra_args)
    result = executor.run(cmd)
    if result.ok:
        return
    raise RuntimeError(
        f"os_deps: failed to install {pkgs} via '{backend.key}'.\n"
        f"exit_code: {result.exit_code}\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}\n"
        "Ensure passwordless sudo (sudo -n) is configured, or run as root."
    )


def _run_custom_cmd(executor, pkg_specs: tuple, command: str, extra_args: list[str]) -> list[str]:
    """Install via a custom command (pip/snap/conda/etc.) instead of an OS PM.

    Returns the list of package names passed to the command.
    """
    flat: list[str] = []
    for spec in pkg_specs:
        if isinstance(spec, str):
            flat.append(spec)
        elif isinstance(spec, dict):
            raw = spec.get("*") or next(iter(spec.values()), "")
            flat.extend(raw if isinstance(raw, list) else [str(raw)])
    pkg_str = " ".join(shlex.quote(p) for p in flat)
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    cmd = f"{command} install {pkg_str}{extra}"
    logger.debug("os_deps: [%s] installing %s", command, flat)
    result = executor.run(cmd)
    if not result.ok:
        logger.error(
            "os_deps: [%s] install failed for %s — exit=%d\nstderr: %s",
            command,
            flat,
            result.exit_code,
            result.stderr[-500:],
        )
        raise RuntimeError(
            f"os_deps: '{command} install' failed for {flat}.\n"
            f"exit_code: {result.exit_code}\nstderr: {result.stderr[-1000:]}"
        )
    if result.stderr.strip():
        logger.warning("os_deps: [%s] install warnings for %s:\n%s", command, flat, result.stderr.strip())
    logger.info("os_deps: installed %s via %s", flat, command)
    return flat


# ---------------------------------------------------------------------------
# Main install orchestration
# ---------------------------------------------------------------------------


def _execute_install(executor, spec: _InstallSpec) -> InstallResult:
    """Install packages described by spec through executor.

    Returns InstallResult with .installed, .already_present, .versions, .dry_run, .command.
    """
    if not spec.pkg_specs:
        logger.debug("os_deps: no packages specified — nothing to do")
        return InstallResult()

    if _is_dry_run(executor):
        flat = _dry_run_label(spec)
        logger.info("os_deps: [DryRun] would install: %s", flat)
        executor.run(f"echo '[os_deps DryRun] would install: {flat}'")
        return InstallResult(dry_run=True, command=spec.command)

    if spec.command:
        logger.debug("os_deps: using custom installer '%s' for %s", spec.command, spec.pkg_specs)
        installed = _run_custom_cmd(executor, spec.pkg_specs, spec.command, spec.extra_args)
        return InstallResult(installed=installed, command=spec.command)

    backend = _detect_backend(executor)
    logger.debug("os_deps: backend detected: '%s'", backend.key)
    resolved_src = _resolve_src(backend, spec.src, spec.sources)
    resolved_pkgs = _resolve_pkgs(spec.pkg_specs, backend)

    if not resolved_pkgs:
        logger.debug("os_deps: no packages resolved for backend '%s'", backend.key)
        return InstallResult()

    if spec.on_missing == "install":
        logger.debug("os_deps: probing installed state for %s", resolved_pkgs)
        missing = _probe_missing(executor, backend, resolved_pkgs)
        already_present = [p for p in resolved_pkgs if p not in missing]

        if already_present:
            present_versions = _get_installed_versions(executor, backend, already_present)
            present_summary = ", ".join(
                f"{p}={present_versions[p]}" if p in present_versions else p for p in already_present
            )
            logger.info("os_deps: already installed — %s", present_summary)

        if not missing:
            logger.debug("os_deps: all packages already present — no install needed")
            return InstallResult(
                already_present=already_present,
                versions=_get_installed_versions(executor, backend, already_present),
            )

        logger.debug("os_deps: [%s] installing missing packages: %s", backend.key, missing)
        _run_install_cmd(executor, backend, missing, resolved_src, spec.extra_args)

        installed_versions = _get_installed_versions(executor, backend, missing)
        installed_summary = ", ".join(f"{p}={installed_versions[p]}" if p in installed_versions else p for p in missing)
        logger.info("os_deps: installed via %s — %s", backend.key, installed_summary)

        all_versions = {
            **_get_installed_versions(executor, backend, already_present),
            **installed_versions,
        }
        return InstallResult(installed=missing, already_present=already_present, versions=all_versions)

    # Non-install modes: probe only
    logger.debug("os_deps: on_missing='%s' — probing only, no install", spec.on_missing)
    missing = _probe_missing(executor, backend, resolved_pkgs)
    if not missing:
        logger.debug("os_deps: all packages present (on_missing=%s)", spec.on_missing)
        return InstallResult(already_present=resolved_pkgs)

    names = ", ".join(missing)
    msg = (
        f"Required OS packages not installed: [{names}]\n"
        f"Set ROCM_TEST_PKG_ON_MISSING=install or pre-provision with:\n"
        f"  --pre-install pkg={','.join(missing)}"
    )
    if spec.on_missing == "skip":
        logger.warning("os_deps: skipping test — missing packages: %s", missing)
        pytest.skip(msg)
    elif spec.on_missing == "fail":
        logger.warning("os_deps: failing test (on_missing=fail) — missing packages: %s", missing)
        pytest.fail(msg)
    elif spec.on_missing == "probe":
        logger.warning("os_deps: missing packages (probe-only, not gating): %s", missing)
    else:
        raise ValueError(f"Unknown on_missing value: {spec.on_missing!r}")

    return InstallResult(already_present=[p for p in resolved_pkgs if p not in missing])


def _dry_run_label(spec: _InstallSpec) -> str:
    parts = [repr(s) if isinstance(s, dict) else s for s in spec.pkg_specs]
    note = ""
    if spec.command:
        note += f" command={spec.command}"
    if spec.src:
        note += f" src={spec.src}"
    if spec.sources:
        note += f" sources={spec.sources}"
    if spec.extra_args:
        note += f" extra_args={spec.extra_args}"
    return ", ".join(parts) + note


# ---------------------------------------------------------------------------
# Executor helpers
# ---------------------------------------------------------------------------


def _is_dry_run(executor) -> bool:
    return type(executor).__name__ == "DryRunExecutor" or bool(getattr(executor, "_dry_run", False))


def _resolve_executor(request: pytest.FixtureRequest):
    try:
        return request.getfixturevalue("target_executor")
    except pytest.FixtureLookupError:
        return request.getfixturevalue("dry_run_executor")


# ---------------------------------------------------------------------------
# __OS_DEPS__ resolution: function > class > module
# ---------------------------------------------------------------------------


def _resolve_os_deps(request: pytest.FixtureRequest) -> _InstallSpec | None:
    raw = (
        getattr(request.node.function, "__OS_DEPS__", None)
        or getattr(request.cls, "__OS_DEPS__", None)
        or getattr(request.module, "__OS_DEPS__", None)
    )
    if raw is None:
        return None
    if isinstance(raw, _InstallSpec):
        return raw
    if isinstance(raw, OS_DEPS):
        return raw._spec
    return None


# ---------------------------------------------------------------------------
# pytest plugin hooks and autouse fixture
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "os_deps(*pkg_specs, src=None, sources=None, command=None, extra_args=None): "
        "DEPRECATED — prefer @OS_DEPS decorator or __OS_DEPS__ class/module attr. "
        "Kept for backwards compatibility.",
    )


@pytest.fixture(autouse=True)
def _os_deps_autouse(request: pytest.FixtureRequest) -> None:
    """Set executor context and process @OS_DEPS declarations before each test."""
    global _current_executor
    _current_executor = _resolve_executor(request)

    spec = _resolve_os_deps(request)
    if spec is not None:
        try:
            _execute_install(_current_executor, spec)
        except RuntimeError as exc:
            pytest.skip(str(exc))
            return

    # Backwards-compat: @pytest.mark.os_deps marker
    if spec is None:
        marker = request.node.get_closest_marker("os_deps")
        if marker and marker.args:
            compat = _InstallSpec(
                pkg_specs=marker.args,
                src=marker.kwargs.get("src"),
                sources=marker.kwargs.get("sources"),
                command=marker.kwargs.get("command"),
                extra_args=marker.kwargs.get("extra_args") or [],
            )
            try:
                _execute_install(_current_executor, compat)
            except RuntimeError as exc:
                pytest.skip(str(exc))
                return

    return
