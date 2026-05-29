# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
install_plugin.py -- Pre-session parallel ROCm and OS package installation.

Handles --pre-install rocm=X (ROCm version upgrade) and --pre-install pkg=A,B
(apt packages). Runs in parallel across all nodes in NodePool before any tests.
Skips nodes already at the requested ROCm version.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import re
from typing import Any

import pytest

from framework.common.helpers import ExecutionResult

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rocm-install", "ROCm pre-session install options")
    group.addoption(
        "--pre-install",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Pre-install packages or ROCm on all fleet nodes before tests run. "
            "Repeat for multiple installs. "
            "Examples: --pre-install rocm=6.4.0  --pre-install pkg=curl,wget"
        ),
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Broadcast check+install to all nodes in parallel at session start."""
    config = session.config
    pre_install_args: list[str] = config.getoption("--pre-install", default=[])

    if not pre_install_args:
        return  # default: no-op

    if config.getoption("--no-gpu", default=False):
        logger.info("install_plugin: --no-gpu active — skipping pre-install")
        return

    from framework.nodes.node_pool import NodePool

    pool: NodePool | None = getattr(config, "_node_pool", None)
    if pool is None:
        logger.warning("install_plugin: NodePool not ready — skipping pre-install")
        return

    # Parse all --pre-install arguments into install specs
    specs = _parse_install_specs(pre_install_args)
    if not specs:
        return

    print(f"\n[pre-install] Running on {len(pool.node_specs)} node(s): {specs}")

    failed_nodes: list[str] = []

    def _work_for_node(node_spec) -> tuple[str, bool]:
        """Per-node worker: check + conditionally install each spec."""
        ssh = pool._ssh_sessions.get(node_spec.label)
        label = node_spec.label
        all_ok = True
        for spec in specs:
            ok = _handle_spec(spec, ssh, label)
            if not ok:
                all_ok = False
        return label, all_ok

    max_workers = max(len(pool.node_specs), 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_work_for_node, spec): spec for spec in pool.node_specs}
        for fut in as_completed(futures):
            label, ok = fut.result()
            if not ok:
                failed_nodes.append(label)

    if failed_nodes:
        pytest.exit(
            f"[pre-install] Installation failed on node(s): {failed_nodes}. "
            "Fix the nodes before re-running or remove --pre-install.",
            returncode=4,
        )
    else:
        print("[pre-install] All nodes ready. Proceeding with test collection.")


def _parse_install_specs(args: list[str]) -> list[dict[str, Any]]:
    """Parse ``--pre-install`` arg strings into typed spec dicts.

    Returns list of dicts, each with keys ``type`` and ``value``.
    Unknown prefixes are logged as warnings and ignored.

    Examples:
        "rocm=6.4.0"    → {"type": "rocm", "value": "6.4.0"}
        "pkg=curl,wget" → {"type": "pkg", "value": ["curl", "wget"]}
    """
    specs: list[dict[str, Any]] = []
    for arg in args:
        if "=" not in arg:
            logger.warning("pre-install: ignoring malformed arg %r (expected KEY=VALUE)", arg)
            continue
        key, _, val = arg.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if key == "rocm":
            specs.append({"type": "rocm", "value": val})
        elif key == "pkg":
            pkgs = [p.strip() for p in val.split(",") if p.strip()]
            if pkgs:
                specs.append({"type": "pkg", "value": pkgs})
        else:
            logger.warning("pre-install: unknown install type %r — supported: rocm, pkg", key)
    return specs


def _run(ssh, command: str, timeout: float = 120.0) -> ExecutionResult:
    """Run *command* via *ssh* (or locally when ssh is None)."""
    if ssh is not None:
        return ssh.run(command, timeout=timeout)  # type: ignore[no-any-return]
    from framework.executors.cpu_executor import CpuExecutor

    return CpuExecutor().run(command, timeout=timeout)


def _handle_spec(spec: dict, ssh, label: str) -> bool:
    """Dispatch to the correct handler based on ``spec["type"]``."""
    if spec["type"] == "rocm":
        return _install_rocm(spec["value"], ssh, label)
    if spec["type"] == "pkg":
        return _install_pkgs(spec["value"], ssh, label)
    return True  # unknown types already warned in _parse_install_specs


def _install_rocm(version: str, ssh, label: str) -> bool:
    """Check ROCm version on *label*; install if different from *version*.

    Check: query rocm-core package version via dpkg or rpm.
    Skip: print message when already at *version*.
    Install: apt-get / dnf depending on distro.
    """
    # Check installed version
    check_cmd = (
        "dpkg-query --show --showformat='${Version}' rocm-core 2>/dev/null "
        "|| rpm -q --queryformat '%{VERSION}' rocm-core 2>/dev/null "
        "|| echo ''"
    )
    result = _run(ssh, check_cmd, timeout=30.0)
    current_raw = result.stdout.strip()
    match = re.search(r"(\d+\.\d+\.\d+)", current_raw)
    current = match.group(1) if match else ""

    if current == version:
        print(f"[pre-install] {label}: ROCm {version} already installed — skip")
        return True

    installed_str = current if current else "not installed"
    print(f"[pre-install] {label}: ROCm current={installed_str} → installing {version}")

    # Distro-agnostic install command: apt or dnf
    install_cmd = (
        f"bash -c '"
        f"if command -v apt-get >/dev/null 2>&1; then "
        f"  apt-get install -y rocm-core={version}* rocm-dev={version}* 2>&1; "
        f"elif command -v dnf >/dev/null 2>&1; then "
        f'  dnf install -y "rocm-core-{version}" "rocm-dev-{version}" 2>&1; '
        f"else echo unsupported-distro && exit 1; fi'"
    )
    result = _run(ssh, install_cmd, timeout=600.0)
    if result.ok:
        print(f"[pre-install] {label}: ROCm {version} installed successfully")
        return True

    print(f"[pre-install] {label}: ROCm install FAILED (rc={result.exit_code})")
    logger.error("pre-install[%s] stderr: %s", label, result.stderr[:500])
    return False


def _install_pkgs(pkgs: list[str], ssh, label: str) -> bool:
    """Check each package; install any that are missing on *label*.

    Check: dpkg-query / rpm -q.
    Install: apt-get / dnf for missing packages only.
    """
    missing = []
    for pkg in pkgs:
        check_cmd = f"dpkg-query -W {pkg} >/dev/null 2>&1 " f"|| rpm -q {pkg} >/dev/null 2>&1"
        result = _run(ssh, check_cmd, timeout=15.0)
        if result.ok:
            print(f"[pre-install] {label}: {pkg} already installed — skip")
        else:
            print(f"[pre-install] {label}: {pkg} missing — will install")
            missing.append(pkg)

    if not missing:
        return True

    pkg_list = " ".join(missing)
    install_cmd = (
        f"bash -c '"
        f"if command -v apt-get >/dev/null 2>&1; then "
        f"  apt-get install -y {pkg_list} 2>&1; "
        f"elif command -v dnf >/dev/null 2>&1; then "
        f"  dnf install -y {pkg_list} 2>&1; "
        f"else echo unsupported-distro && exit 1; fi'"
    )
    result = _run(ssh, install_cmd, timeout=300.0)
    if result.ok:
        print(f"[pre-install] {label}: {missing} installed successfully")
        return True

    print(f"[pre-install] {label}: pkg install FAILED (rc={result.exit_code})")
    logger.error("pre-install[%s] stderr: %s", label, result.stderr[:500])
    return False
