# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
os_deps_plugin.py — OS package auto-install for rocm-tests.

Self-contained: copy this single file to use in another repo.

Public API (import from framework.os_adapter.deps for convenience):
    from framework.os_adapter.deps import OS_DEPS

Usage:
    # Function/method decorator
    @OS_DEPS("cmake", "git")
    @OS_DEPS("cmake==3.26")
    @OS_DEPS({"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"})
    @OS_DEPS({"apt": ["libssl-dev", "libdrm-dev"], "dnf": ["openssl-devel", "libdrm-devel"]})
    @OS_DEPS("pkg", command="pip")
    @OS_DEPS("pkg", command="pip", extra_args=["--index-url", "https://..."])
    @OS_DEPS("pkg", src="https://my-repo/")
    @OS_DEPS("pkg", sources={"apt": "deb https://...", "dnf": "https://..."})

    # Class-level (applies to all methods)
    class TestFoo:
        __OS_DEPS__ = OS_DEPS("cmake", "git")

    # Module-level (applies to all tests in file)
    __OS_DEPS__ = OS_DEPS("cmake")

    # Mid-test — same arg forms as decorator
    OS_DEPS().install("cmake")
    OS_DEPS().install("cmake==3.26", "git")
    OS_DEPS().install({"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"})
    OS_DEPS().install({"apt": ["libssl-dev", "libdrm-dev"], "dnf": ["openssl-devel", "libdrm-devel"]})
    OS_DEPS().install(*LIST_OF_PKGS, command="pip", extra_args=["--quiet"])
    OS_DEPS().install("node --classic", command="snap")

pkg_spec forms (str or dict, same everywhere):
    "cmake"              plain name, auto-detect PM
    "cmake==3.26"        version pin; translated per PM:
                           apt → cmake=3.26*
                           dnf/yum → cmake-3.26
                           zypper/apk/pacman → cmake=3.26
    {"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"}   per-PM str value
    {"apt": ["libssl-dev", "libdrm-dev"], "dnf": ["openssl-devel"]}    per-PM list value

command= values:
    None (default)  auto-detect OS PM (apt/dnf/yum/zypper/apk/pacman/winget/choco/scoop)
    "pip"           pip install <pkgs> [extra_args]
    "snap"          snap install <pkgs> [extra_args]
    "conda"         conda install <pkgs> [extra_args]
    any str         <command> install <pkgs> [extra_args]

extra_args=:
    List of additional flags passed verbatim to the package manager.
    Examples: ["--no-install-recommends"], ["--best", "--allowerasing"],
              ["--index-url", "https://..."]

on_missing (env: ROCM_TEST_PKG_ON_MISSING, default "install"):
    "install"   probe then install missing packages
    "skip"      pytest.skip if any missing
    "fail"      pytest.fail if any missing
    "probe"     log warning only, do not gate test

Resolution priority: function.__OS_DEPS__ > class.__OS_DEPS__ > module.__OS_DEPS__

Platform support:
    Linux:   apt/apt-get, dnf, yum, zypper, apk, pacman (auto-detected)
    Windows: winget, choco, scoop (auto-detected; bootstrapped if missing)
"""

from __future__ import annotations

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

# Env-var policy
_ENV_POLICY = os.environ.get("ROCM_TEST_PKG_ON_MISSING", "install").lower()

# Cache attribute on executor to avoid re-detection
_BACKEND_ATTR = "_os_deps_backend"

# ---------------------------------------------------------------------------
# Internal spec
# ---------------------------------------------------------------------------

PkgSpec = str | dict


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
    ) -> None:
        """Install packages immediately using the current test's executor.

        Reads executor from the autouse fixture context — no argument needed.
        Accepts all the same pkg_spec forms as the decorator.
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
        _execute_install(_current_executor, spec)


# ---------------------------------------------------------------------------
# Backend descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Backend:
    key: str
    detect_cmd: str
    probe_cmd_fn: object
    install_cmd_fn: object
    bootstrap_fn: object | None = None
    needs_sudo: bool = True
    platform: str = "linux"


# ---------------------------------------------------------------------------
# Version parsing and translation
# ---------------------------------------------------------------------------


def _parse_version(pkg: str) -> tuple[str, str | None]:
    """'cmake==3.26' → ('cmake', '3.26'); 'cmake' → ('cmake', None)."""
    if "==" in pkg:
        name, ver = pkg.split("==", 1)
        return name.strip(), ver.strip()
    return pkg.strip(), None


def _versioned(name: str, version: str | None, pm_key: str) -> str:
    """Apply version syntax per package manager."""
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
# Resolve pkg_specs → flat list of PM-specific package name strings
# ---------------------------------------------------------------------------


def _resolve_pkgs(pkg_specs: tuple, backend: _Backend) -> list[str]:
    """Resolve all pkg_specs to PM-specific package names for the detected backend."""
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
# Probe functions — check if a package is already installed
# ---------------------------------------------------------------------------


def _probe_apt(pkg_name: str) -> str:
    base, _ = _parse_version(pkg_name)
    return f"dpkg-query -W -f='${{Status}}' {shlex.quote(base)} 2>/dev/null | grep -q 'install ok installed'"


def _probe_rpm(pkg_name: str) -> str:
    base, _ = _parse_version(pkg_name)
    return f"rpm -q {shlex.quote(base)} >/dev/null 2>&1"


def _probe_pacman(pkg_name: str) -> str:
    base, _ = _parse_version(pkg_name)
    return f"pacman -Q {shlex.quote(base)} >/dev/null 2>&1"


def _probe_apk(pkg_name: str) -> str:
    base, _ = _parse_version(pkg_name)
    return f"apk info -e {shlex.quote(base)} >/dev/null 2>&1"


def _probe_none(_: str) -> str:
    return "false"  # always attempt install (Windows PMs, pip, snap, etc.)


# ---------------------------------------------------------------------------
# Install command functions — return a shell command string
# ---------------------------------------------------------------------------


def _apt_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
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


def _dnf_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
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


def _yum_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
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


def _zypper_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
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
        f"bash -c '{sudo}zypper --non-interactive install -y " f"--no-recommends --no-gpg-checks {pkg_str}{extra} 2>&1'"
    )


def _apk_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    if src:
        return f"bash -c '{sudo}apk add --no-cache --repository {shlex.quote(src)} {pkg_str}{extra} 2>&1'"
    return f"bash -c '{sudo}apk add --no-cache {pkg_str}{extra} 2>&1'"


def _pacman_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    if src:
        logger.warning("os_deps: pacman does not support src= URL; ignoring %s", src)
    return f"bash -c '{sudo}pacman -S --noconfirm {pkg_str}{extra} 2>&1'"


def _winget_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
    base = "winget install --silent --accept-package-agreements --accept-source-agreements"
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    cmds = []
    for pkg in pkgs:
        if src:
            cmds.append(f"{base} --source {shlex.quote(src)} --id {shlex.quote(pkg)}{extra}")
        else:
            cmds.append(f"{base} --id {shlex.quote(pkg)}{extra}")
    return " && ".join(cmds)


def _choco_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    if src:
        return f"choco install -y {pkg_str} --source {shlex.quote(src)}{extra}"
    return f"choco install -y {pkg_str}{extra}"


def _scoop_cmd(pkgs: list[str], sudo: str, src: str | None, extra_args: list[str]) -> str:
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    if src:
        bucket = f"os_deps_{hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]}"
        add = f"scoop bucket add {shlex.quote(bucket)} {shlex.quote(src)}"
        install = " && ".join(f"scoop install {shlex.quote(p)}{extra}" for p in pkgs)
        return f"{add} && {install}"
    return " && ".join(f"scoop install {shlex.quote(p)}{extra}" for p in pkgs)


# ---------------------------------------------------------------------------
# Windows bootstrappers
# ---------------------------------------------------------------------------


def _bootstrap_winget(executor) -> None:
    logger.info("os_deps: winget not found — bootstrapping App Installer")
    url = (
        "https://github.com/microsoft/winget-cli/releases/latest/download/"
        "Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle"
    )
    result = executor.run(f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"Add-AppxPackage -Path '{url}'\"")
    if not result.ok:
        raise RuntimeError(
            f"os_deps: failed to bootstrap winget.\n" f"exit_code: {result.exit_code}\nstderr: {result.stderr[-500:]}"
        )


def _bootstrap_choco(executor) -> None:
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


def _bootstrap_scoop(executor) -> None:
    logger.info("os_deps: Scoop not found — bootstrapping")
    result = executor.run('powershell -NoProfile -ExecutionPolicy RemoteSigned -Command "irm get.scoop.sh | iex"')
    if not result.ok:
        raise RuntimeError(
            f"os_deps: failed to bootstrap Scoop.\n" f"exit_code: {result.exit_code}\nstderr: {result.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_LINUX_BACKENDS: list[_Backend] = [
    _Backend(
        key="apt",
        detect_cmd="command -v apt-get",
        probe_cmd_fn=_probe_apt,
        install_cmd_fn=_apt_cmd,
        needs_sudo=True,
        platform="linux",
    ),
    _Backend(
        key="dnf",
        detect_cmd="command -v dnf",
        probe_cmd_fn=_probe_rpm,
        install_cmd_fn=_dnf_cmd,
        needs_sudo=True,
        platform="linux",
    ),
    _Backend(
        key="yum",
        detect_cmd="command -v yum",
        probe_cmd_fn=_probe_rpm,
        install_cmd_fn=_yum_cmd,
        needs_sudo=True,
        platform="linux",
    ),
    _Backend(
        key="zypper",
        detect_cmd="command -v zypper",
        probe_cmd_fn=_probe_rpm,
        install_cmd_fn=_zypper_cmd,
        needs_sudo=True,
        platform="linux",
    ),
    _Backend(
        key="apk",
        detect_cmd="command -v apk",
        probe_cmd_fn=_probe_apk,
        install_cmd_fn=_apk_cmd,
        needs_sudo=True,
        platform="linux",
    ),
    _Backend(
        key="pacman",
        detect_cmd="command -v pacman",
        probe_cmd_fn=_probe_pacman,
        install_cmd_fn=_pacman_cmd,
        needs_sudo=True,
        platform="linux",
    ),
]

_WINDOWS_BACKENDS: list[_Backend] = [
    _Backend(
        key="winget",
        detect_cmd="winget --version",
        probe_cmd_fn=_probe_none,
        install_cmd_fn=_winget_cmd,
        bootstrap_fn=_bootstrap_winget,
        needs_sudo=False,
        platform="windows",
    ),
    _Backend(
        key="choco",
        detect_cmd="choco --version",
        probe_cmd_fn=_probe_none,
        install_cmd_fn=_choco_cmd,
        bootstrap_fn=_bootstrap_choco,
        needs_sudo=False,
        platform="windows",
    ),
    _Backend(
        key="scoop",
        detect_cmd="scoop --version",
        probe_cmd_fn=_probe_none,
        install_cmd_fn=_scoop_cmd,
        bootstrap_fn=_bootstrap_scoop,
        needs_sudo=False,
        platform="windows",
    ),
]


# ---------------------------------------------------------------------------
# Backend detection (cached per executor instance)
# ---------------------------------------------------------------------------


def _detect_backend(executor) -> _Backend:
    cached = getattr(executor, _BACKEND_ATTR, None)
    if cached is not None:
        return cached
    for backend in _LINUX_BACKENDS:
        result = executor.run(f"bash -c '{backend.detect_cmd} >/dev/null 2>&1 && echo OK || echo NO'")
        if result.ok and "OK" in result.stdout:
            _cache_backend(executor, backend)
            logger.debug("os_deps: detected backend '%s'", backend.key)
            return backend
    for backend in _WINDOWS_BACKENDS:
        result = executor.run(f"{backend.detect_cmd} >NUL 2>&1 && echo OK || echo NO")
        if result.ok and "OK" in result.stdout:
            _cache_backend(executor, backend)
            return backend
        if backend.bootstrap_fn is not None:
            try:
                backend.bootstrap_fn(executor)
                result2 = executor.run(f"{backend.detect_cmd} >NUL 2>&1 && echo OK || echo NO")
                if result2.ok and "OK" in result2.stdout:
                    _cache_backend(executor, backend)
                    return backend
            except RuntimeError as exc:
                logger.warning("os_deps: bootstrap of '%s' failed: %s", backend.key, exc)
    raise RuntimeError(
        "os_deps: no supported package manager found. "
        "Tried: apt-get, dnf, yum, zypper, apk, pacman (Linux); "
        "winget, choco, scoop (Windows, with auto-bootstrap)."
    )


def _cache_backend(executor, backend: _Backend) -> None:
    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(executor, _BACKEND_ATTR, backend)


# ---------------------------------------------------------------------------
# Probe missing packages
# ---------------------------------------------------------------------------


def _probe_missing(executor, backend: _Backend, pkgs: list[str]) -> list[str]:
    """Return the subset of pkgs not yet installed on the executor's host."""
    missing = []
    for pkg in pkgs:
        probe = backend.probe_cmd_fn(pkg)
        result = executor.run(f"bash -c '{probe}' 2>/dev/null")
        if not result.ok:
            missing.append(pkg)
    return missing


# ---------------------------------------------------------------------------
# Install command execution
# ---------------------------------------------------------------------------


def _run_install_cmd(
    executor,
    backend: _Backend,
    pkgs: list[str],
    src: str | None,
    extra_args: list[str],
) -> None:
    """Run the install command, trying sudo -n first then without sudo."""
    sudo = "sudo -n " if backend.needs_sudo else ""
    cmd = backend.install_cmd_fn(pkgs, sudo, src, extra_args)
    result = executor.run(cmd)
    if result.ok:
        return
    if backend.needs_sudo:
        cmd_nosudo = backend.install_cmd_fn(pkgs, "", src, extra_args)
        result = executor.run(cmd_nosudo)
        if result.ok:
            return
    raise RuntimeError(
        f"os_deps: failed to install {pkgs} via '{backend.key}'.\n"
        f"exit_code: {result.exit_code}\n"
        f"stdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}\n"
        "Ensure passwordless sudo (sudo -n) is configured, or run as root."
    )


def _resolve_src(backend: _Backend, src: str | None, sources: dict[str, str] | None) -> str | None:
    if sources and backend.key in sources:
        return sources[backend.key]
    return src


# ---------------------------------------------------------------------------
# Custom command installer (pip, snap, conda, any string)
# ---------------------------------------------------------------------------


def _run_custom_cmd(executor, pkg_specs: tuple, command: str, extra_args: list[str]) -> None:
    """Install via a custom command (pip/snap/conda/etc.) instead of OS PM."""
    flat: list[str] = []
    for spec in pkg_specs:
        if isinstance(spec, str):
            flat.append(spec)
        elif isinstance(spec, dict):
            raw = spec.get("*") or next(iter(spec.values()), "")
            if isinstance(raw, list):
                flat.extend(raw)
            else:
                flat.append(str(raw))
    pkg_str = " ".join(shlex.quote(p) for p in flat)
    extra = (" " + " ".join(extra_args)) if extra_args else ""
    cmd = f"{command} install {pkg_str}{extra}"
    result = executor.run(cmd)
    if not result.ok:
        raise RuntimeError(
            f"os_deps: '{command} install' failed for {flat}.\n"
            f"exit_code: {result.exit_code}\nstderr: {result.stderr[-1000:]}"
        )


# ---------------------------------------------------------------------------
# Main install orchestration
# ---------------------------------------------------------------------------


def _execute_install(executor, spec: _InstallSpec) -> None:
    """Install packages described by spec through executor."""
    if not spec.pkg_specs:
        return

    if _is_dry_run(executor):
        flat = _dry_run_label(spec)
        executor.run(f"echo '[os_deps DryRun] would install: {flat}'")
        return

    # Custom command — bypass OS PM entirely
    if spec.command:
        _run_custom_cmd(executor, spec.pkg_specs, spec.command, spec.extra_args)
        return

    backend = _detect_backend(executor)
    resolved_src = _resolve_src(backend, spec.src, spec.sources)
    resolved_pkgs = _resolve_pkgs(spec.pkg_specs, backend)

    if not resolved_pkgs:
        return

    if spec.on_missing == "install":
        missing = _probe_missing(executor, backend, resolved_pkgs)
        if not missing:
            logger.debug("os_deps: all already installed — skipping: %s", resolved_pkgs)
            return
        _run_install_cmd(executor, backend, missing, resolved_src, spec.extra_args)
        return

    # Non-install modes: probe only
    missing = _probe_missing(executor, backend, resolved_pkgs)
    if not missing:
        return
    names = ", ".join(missing)
    msg = (
        f"Required OS packages not installed: [{names}]\n"
        f"Set ROCM_TEST_PKG_ON_MISSING=install or pre-provision with:\n"
        f"  --pre-install pkg={','.join(missing)}"
    )
    if spec.on_missing == "skip":
        pytest.skip(msg)
    elif spec.on_missing == "fail":
        pytest.fail(msg)
    elif spec.on_missing == "probe":
        logger.warning("os_deps: missing packages (probe-only, not gating): %s", missing)
    else:
        raise ValueError(f"Unknown on_missing value: {spec.on_missing!r}")


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
    # Function decorator sets fn.__OS_DEPS__ = _InstallSpec (via __call__)
    # Class/module attr sets __OS_DEPS__ = OS_DEPS(...) instance — extract ._spec
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
