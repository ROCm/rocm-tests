# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pkg_manager.py -- OS/distro-aware package installation via executor.

Public API:
    install_packages(executor, *pkg_specs, dry_run=False, src=None, sources=None)

pkg_spec forms:
    "cmake"                                        same name on all managers
    "libdrm-dev"                                   auto-retries -devel on dnf/yum/zypper if -dev fails
    {"dnf": "libtiff-devel", "*": "libtiff-dev"}  explicit per-manager, "*" = fallback
    {"apt": "build-essential", "dnf": "@development-tools"}  no fallback — listed managers only

Source URL forms:
    src="https://url"                              same URL for all managers
    sources={"apt": "url1", "dnf": "url2"}         per-manager (overrides src= per manager)

Linux backends (auto-detected via command -v):
    apt, dnf, yum, zypper, apk, pacman

Windows backends (auto-detected; bootstrapped if missing):
    winget -> choco -> scoop
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import logging
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

_BACKEND_ATTR = "_os_deps_backend"
_DEVEL_SUFFIX_MANAGERS = {"dnf", "yum", "zypper"}

PkgSpec = str | dict[str, str]


@dataclass(frozen=True)
class Backend:
    key: str
    detect_cmd: str
    install_fn: Callable[[AbstractExecutor, list[str], str, str | None], str]
    bootstrap_fn: Callable[[AbstractExecutor], None] | None = None
    needs_sudo: bool = True
    platform: str = "linux"
    extra_flags: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Linux install functions
# ---------------------------------------------------------------------------


def _apt_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    if src:
        if src.endswith(".deb"):
            return (
                f"bash -c '{sudo}wget -q -O /tmp/_os_deps_pkg.deb {shlex.quote(src)} && "
                f"{sudo}dpkg -i /tmp/_os_deps_pkg.deb && {sudo}apt-get install -f -y 2>&1'"
            )
        repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
        repo_file = f"/etc/apt/sources.list.d/os_deps_{repo_hash}.list"
        return (
            f"bash -c 'grep -qF {shlex.quote(src)} {repo_file} 2>/dev/null || "
            f"echo {shlex.quote(src)} | {sudo}tee {repo_file} >/dev/null; "
            f"{sudo}apt-get update -q -y && "
            f"{sudo}apt-get install -y --no-install-recommends {pkg_str} 2>&1'"
        )
    return f"bash -c '{sudo}apt-get update -q -y && {sudo}apt-get install -y --no-install-recommends {pkg_str} 2>&1'"


def _dnf_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    if src:
        if src.endswith(".rpm"):
            return f"bash -c '{sudo}dnf install -y {shlex.quote(src)} 2>&1'"
        repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
        repo_file = f"/etc/yum.repos.d/os_deps_{repo_hash}.repo"
        repo_content = f"[os_deps_{repo_hash}]\\nname=os_deps custom\\nbaseurl={src}\\nenabled=1\\ngpgcheck=0"
        return (
            f"bash -c 'echo -e {shlex.quote(repo_content)} | {sudo}tee {repo_file} >/dev/null && "
            f"{sudo}dnf install -y {pkg_str} 2>&1'"
        )
    return f"bash -c '{sudo}dnf install -y {pkg_str} 2>&1'"


def _yum_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    if src:
        if src.endswith(".rpm"):
            return f"bash -c '{sudo}yum install -y {shlex.quote(src)} 2>&1'"
        repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
        repo_file = f"/etc/yum.repos.d/os_deps_{repo_hash}.repo"
        repo_content = f"[os_deps_{repo_hash}]\\nname=os_deps custom\\nbaseurl={src}\\nenabled=1\\ngpgcheck=0"
        return (
            f"bash -c 'echo -e {shlex.quote(repo_content)} | {sudo}tee {repo_file} >/dev/null && "
            f"{sudo}yum install -y {pkg_str} 2>&1'"
        )
    return f"bash -c '{sudo}yum install -y {pkg_str} 2>&1'"


def _zypper_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    if src:
        repo_hash = hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]
        repo_alias = f"os_deps_{repo_hash}"
        return (
            f"bash -c '{sudo}zypper --non-interactive addrepo --no-gpgcheck"
            f" {shlex.quote(src)} {repo_alias} 2>&1; "
            f"{sudo}zypper --non-interactive refresh 2>&1 && "
            f"{sudo}zypper --non-interactive install -y --no-recommends --no-gpg-checks {pkg_str} 2>&1'"
        )
    return f"bash -c '{sudo}zypper --non-interactive install -y --no-recommends --no-gpg-checks {pkg_str} 2>&1'"


def _apk_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    if src:
        return f"bash -c '{sudo}apk add --no-cache --repository {shlex.quote(src)} {pkg_str} 2>&1'"
    return f"bash -c '{sudo}apk add --no-cache {pkg_str} 2>&1'"


def _pacman_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    if src:
        logger.warning("os_deps: pacman does not support src= URL; ignoring %s", src)
    return f"bash -c '{sudo}pacman -S --noconfirm {pkg_str} 2>&1'"


# ---------------------------------------------------------------------------
# Windows install functions
# ---------------------------------------------------------------------------


def _winget_install(executor, pkgs, sudo, src):
    base = "winget install --silent --accept-package-agreements --accept-source-agreements"
    cmds = []
    for pkg in pkgs:
        if src:
            cmds.append(f"{base} --source {shlex.quote(src)} --id {shlex.quote(pkg)}")
        else:
            cmds.append(f"{base} --id {shlex.quote(pkg)}")
    return " && ".join(cmds)


def _choco_install(executor, pkgs, sudo, src):
    pkg_str = " ".join(shlex.quote(p) for p in pkgs)
    return f"choco install -y {pkg_str} --source {shlex.quote(src)}" if src else f"choco install -y {pkg_str}"


def _scoop_install(executor, pkgs, sudo, src):
    if src:
        bucket_name = f"os_deps_{hashlib.md5(src.encode(), usedforsecurity=False).hexdigest()[:8]}"
        add_bucket = f"scoop bucket add {shlex.quote(bucket_name)} {shlex.quote(src)}"
        install = " && ".join(f"scoop install {shlex.quote(p)}" for p in pkgs)
        return f"{add_bucket} && {install}"
    return " && ".join(f"scoop install {shlex.quote(p)}" for p in pkgs)


# ---------------------------------------------------------------------------
# Windows bootstrappers
# ---------------------------------------------------------------------------


def _bootstrap_winget(executor):
    logger.info("os_deps: winget not found — bootstrapping App Installer")
    url = "https://github.com/microsoft/winget-cli/releases/latest/download/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle"
    result = executor.run(f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"Add-AppxPackage -Path '{url}'\"")
    if not result.ok:
        raise RuntimeError(
            f"os_deps: failed to bootstrap winget.\n"
            f"exit_code: {result.exit_code}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


def _bootstrap_choco(executor):
    logger.info("os_deps: Chocolatey not found — bootstrapping")
    result = executor.run(
        "powershell -NoProfile -ExecutionPolicy Bypass -Command "
        '"Set-ExecutionPolicy Bypass -Scope Process -Force; '
        "[System.Net.ServicePointManager]::SecurityProtocol = "
        "[System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
        "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))\""
    )
    if not result.ok:
        raise RuntimeError(
            f"os_deps: failed to bootstrap Chocolatey.\n"
            f"exit_code: {result.exit_code}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


def _bootstrap_scoop(executor):
    logger.info("os_deps: Scoop not found — bootstrapping")
    result = executor.run('powershell -NoProfile -ExecutionPolicy RemoteSigned -Command "irm get.scoop.sh | iex"')
    if not result.ok:
        raise RuntimeError(
            f"os_deps: failed to bootstrap Scoop.\n"
            f"exit_code: {result.exit_code}\nstdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------

_LINUX_BACKENDS: list[Backend] = [
    Backend(key="apt", detect_cmd="command -v apt-get", install_fn=_apt_install, needs_sudo=True, platform="linux"),
    Backend(key="dnf", detect_cmd="command -v dnf", install_fn=_dnf_install, needs_sudo=True, platform="linux"),
    Backend(key="yum", detect_cmd="command -v yum", install_fn=_yum_install, needs_sudo=True, platform="linux"),
    Backend(
        key="zypper", detect_cmd="command -v zypper", install_fn=_zypper_install, needs_sudo=True, platform="linux"
    ),
    Backend(key="apk", detect_cmd="command -v apk", install_fn=_apk_install, needs_sudo=True, platform="linux"),
    Backend(
        key="pacman", detect_cmd="command -v pacman", install_fn=_pacman_install, needs_sudo=True, platform="linux"
    ),
]

_WINDOWS_BACKENDS: list[Backend] = [
    Backend(
        key="winget",
        detect_cmd="winget --version",
        install_fn=_winget_install,
        bootstrap_fn=_bootstrap_winget,
        needs_sudo=False,
        platform="windows",
    ),
    Backend(
        key="choco",
        detect_cmd="choco --version",
        install_fn=_choco_install,
        bootstrap_fn=_bootstrap_choco,
        needs_sudo=False,
        platform="windows",
    ),
    Backend(
        key="scoop",
        detect_cmd="scoop --version",
        install_fn=_scoop_install,
        bootstrap_fn=_bootstrap_scoop,
        needs_sudo=False,
        platform="windows",
    ),
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_pkg_names(pkgs: list[str], mgr_name: str) -> list[str]:
    """Legacy helper kept for backwards compat — prefer _resolve_spec."""
    return pkgs


def install_packages(
    executor: AbstractExecutor,
    *pkg_specs: PkgSpec,
    dry_run: bool = False,
    src: str | None = None,
    sources: dict[str, str] | None = None,
) -> None:
    """Install *pkg_specs* on the executor's host via the native package manager.

    Args:
        executor:  Any AbstractExecutor (LocalExecutor, SshExecutor, …).
        pkg_specs: str or dict pkg_specs — see module docstring.
        dry_run:   When True, log the would-be command without executing.
        src:       Optional source URL applied to all managers.
        sources:   Optional per-manager source URLs (overrides src= per manager).

    Raises:
        RuntimeError: No supported package manager found, or install failed.
    """
    if not pkg_specs:
        return
    if dry_run:
        _log_dry_run(executor, pkg_specs, src, sources)
        return
    backend = _detect_backend(executor)
    resolved_src = _resolve_src(backend, src, sources)
    for spec in pkg_specs:
        name = _resolve_spec(spec, backend)
        if not name:
            raise RuntimeError(
                f"os_deps: no mapping defined for backend '{backend.key}' in spec {spec!r}. "
                f"Add a '{backend.key}' key or a '*' fallback key to the dict."
            )
        _run_install_with_heuristic(executor, backend, name, resolved_src)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_spec(spec: PkgSpec, backend: Backend) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get(backend.key) or spec.get("*", "")
    raise TypeError(f"pkg_spec must be str or dict, got {type(spec).__name__!r}")


def _resolve_src(backend: Backend, src: str | None, sources: dict[str, str] | None) -> str | None:
    if sources and backend.key in sources:
        return sources[backend.key]
    return src


def _candidate_names(pkg: str, backend: Backend) -> list[str]:
    """Auto-heuristic: -dev suffix → also try -devel on rpm-based managers."""
    candidates = [pkg]
    if backend.key in _DEVEL_SUFFIX_MANAGERS and pkg.endswith("-dev"):
        candidates.append(pkg[:-4] + "-devel")
    return candidates


def _run_install_with_heuristic(executor, backend, pkg, src):
    candidates = _candidate_names(pkg, backend)
    sudo = "sudo -n " if backend.needs_sudo else ""
    last_result = None
    for name in candidates:
        cmd = backend.install_fn(executor, [name], sudo, src)
        result = executor.run(cmd)
        if result.ok:
            if len(candidates) > 1 and name != candidates[0]:
                logger.debug("os_deps: '%s' not found; installed as '%s'", candidates[0], name)
            return
        last_result = result
    if backend.needs_sudo and last_result is not None:
        cmd_nosudo = backend.install_fn(executor, [candidates[-1]], "", src)
        result = executor.run(cmd_nosudo)
        if result.ok:
            return
        last_result = result
    raise RuntimeError(
        f"os_deps: failed to install {candidates} via '{backend.key}'.\n"
        f"exit_code: {last_result.exit_code if last_result else '?'}\n"
        f"stdout: {(last_result.stdout if last_result else '')[-1000:]}\n"
        f"stderr: {(last_result.stderr if last_result else '')[-1000:]}\n"
        "Ensure passwordless sudo (sudo -n) is configured, or run as root."
    )


def _log_dry_run(executor, pkg_specs, src, sources):
    parts = [repr(s) if isinstance(s, dict) else s for s in pkg_specs]
    src_note = (f" src={src}" if src else "") + (f" sources={sources}" if sources else "")
    executor.run(f"echo '[os_deps DryRun] would install: {', '.join(parts)}{src_note}'")


def _detect_backend(executor: AbstractExecutor) -> Backend:
    cached: Backend | None = getattr(executor, _BACKEND_ATTR, None)
    if cached is not None:
        return cached
    for backend in _LINUX_BACKENDS:
        result = executor.run(f"bash -c '{backend.detect_cmd} >/dev/null 2>&1 && echo OK || echo NO'")
        if result.ok and "OK" in result.stdout:
            _cache_backend(executor, backend)
            logger.debug("os_deps: detected backend '%s' on %s", backend.key, _exec_label(executor))
            return backend
    for backend in _WINDOWS_BACKENDS:
        result = executor.run(f"{backend.detect_cmd} >NUL 2>&1 && echo OK || echo NO")
        if result.ok and "OK" in result.stdout:
            _cache_backend(executor, backend)
            logger.debug("os_deps: detected backend '%s' on %s", backend.key, _exec_label(executor))
            return backend
        if backend.bootstrap_fn is not None:
            try:
                backend.bootstrap_fn(executor)
                result2 = executor.run(f"{backend.detect_cmd} >NUL 2>&1 && echo OK || echo NO")
                if result2.ok and "OK" in result2.stdout:
                    _cache_backend(executor, backend)
                    logger.info("os_deps: bootstrapped backend '%s' on %s", backend.key, _exec_label(executor))
                    return backend
            except RuntimeError as exc:
                logger.warning("os_deps: bootstrap of '%s' failed: %s", backend.key, exc)
    raise RuntimeError(
        f"os_deps: no supported package manager found on '{_exec_label(executor)}'. "
        "Tried: apt-get, dnf, yum, zypper, apk, pacman (Linux); winget, choco, scoop (Windows)."
    )


def _cache_backend(executor, backend):
    import contextlib

    with contextlib.suppress(AttributeError, TypeError):
        object.__setattr__(executor, _BACKEND_ATTR, backend)


def _exec_label(executor) -> str:
    return (
        getattr(executor, "node_id", None)
        or getattr(executor, "hostname", None)
        or getattr(executor, "host", None)
        or type(executor).__name__
    )
