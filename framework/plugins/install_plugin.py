# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
install_plugin.py -- Pre-session parallel ROCm, OS package, and PyTorch install.

Handles --pre-install rocm=X (ROCm version upgrade), --pre-install pkg=A,B
(apt packages), and --pre-install pytorch=... (node-local PyTorch
provisioning). Runs in parallel across all nodes in NodePool before any tests.
Skips nodes already at the requested ROCm version. PyTorch provisioning is
NON-FATAL: failures let dependent tests fail rather than aborting the whole
session.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import pathlib
import re
from threading import Lock
from typing import Any

import pytest

from framework.common.helpers import ExecutionResult

logger = logging.getLogger(__name__)
_SHARED_RESULTS_NAME = "framework-provision-results.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("rocm-install", "ROCm pre-session install options")
    group.addoption(
        "--pre-install",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Pre-install packages, ROCm, or an ML framework on all fleet nodes before tests run. "
            "Repeat for multiple installs. "
            "Examples: --pre-install rocm=6.4.0  --pre-install pkg=curl,wget  "
            "--pre-install pytorch=mode=multiarch,device=gfx942,torch=2.11.0+rocm7.13"
        ),
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    """Broadcast check+install to all nodes in parallel at session start."""
    config = session.config
    pre_install_args: list[str] = config.getoption("--pre-install", default=[])

    if not pre_install_args:
        return  # default: no-op

    specs = _parse_install_specs(pre_install_args)
    config._framework_preinstall_requested = any(s["type"] == "pytorch" for s in specs)  # type: ignore[attr-defined]
    if not hasattr(config, "_framework_provision_results"):
        config._framework_provision_results = {}  # type: ignore[attr-defined]

    if config.getoption("--no-gpu", default=False):
        logger.info("install_plugin: --no-gpu active — skipping pre-install")
        return

    if hasattr(config, "workerinput"):
        logger.info("install_plugin: xdist worker — skipping pre-install execution")
        return

    from framework.nodes.node_pool import NodePool

    pool: NodePool | None = getattr(config, "_node_pool", None)
    if pool is None:
        logger.warning("install_plugin: NodePool not ready — skipping pre-install")
        return

    if not specs:
        return

    _run_pre_install(config, pool, specs)


def _run_pre_install(config, pool, specs: list[dict[str, Any]]) -> None:
    """Execute pre-install specs across all pool nodes in parallel."""
    print(f"\n[pre-install] Running on {len(pool.node_specs)} node(s): {specs}")

    from framework.config.loader import load_config

    framework_config = load_config(config_path=config.getoption("--rocm-config", default=None))

    failed_nodes: list[str] = []
    nonfatal_failures: list[str] = []
    results_lock = Lock()

    def _work_for_node(node_spec) -> tuple[str, bool]:
        """Per-node worker: check + conditionally install each spec."""
        ssh = pool._ssh_sessions.get(node_spec.label)
        label = node_spec.label
        all_ok = True
        for spec in specs:
            ok, critical = _handle_spec(spec, ssh, label, config, framework_config, results_lock)
            if not ok and critical:
                all_ok = False
            elif not ok:
                with results_lock:
                    nonfatal_failures.append(label)
        return label, all_ok

    with ThreadPoolExecutor(max_workers=max(len(pool.node_specs), 1)) as executor:
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
    if nonfatal_failures:
        print(
            "[pre-install] Non-fatal framework provisioning failure on node(s): "
            f"{sorted(set(nonfatal_failures))}. Framework-dependent tests will fail."
        )
    _write_shared_results(config, framework_config)
    print("[pre-install] All required pre-installs ready. Proceeding with test collection.")


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
            raise pytest.UsageError(f"Malformed --pre-install argument {arg!r}; expected KEY=VALUE")
        key, _, val = arg.partition("=")
        key = key.strip().lower()
        val = val.strip()
        if key == "rocm":
            if not val:
                raise pytest.UsageError("--pre-install rocm= requires a ROCm version")
            specs.append({"type": "rocm", "value": val})
        elif key == "pkg":
            pkgs = [p.strip() for p in val.split(",") if p.strip()]
            if not pkgs:
                raise pytest.UsageError("--pre-install pkg= requires at least one package name")
            specs.append({"type": "pkg", "value": pkgs})
        elif key == "pytorch":
            from tests.common.ml_provisioning.spec import parse_framework_spec

            try:
                parsed = parse_framework_spec(val, framework="pytorch")
            except ValueError as exc:
                raise pytest.UsageError(f"Invalid --pre-install pytorch=... specification: {exc}") from exc
            specs.append({"type": "pytorch", "value": val, "framework_spec": parsed})
        else:
            raise pytest.UsageError(
                f"Unknown --pre-install type {key!r}; supported install types are rocm, pkg, pytorch"
            )
    return specs


def _run(ssh, command: str, timeout: float = 120.0) -> ExecutionResult:
    """Run *command* via *ssh* (or locally when ssh is None)."""
    if ssh is not None:
        return ssh.run(command, timeout=timeout)  # type: ignore[no-any-return]
    from framework.executors.cpu_executor import CpuExecutor

    return CpuExecutor().run(command, timeout=timeout)


def _handle_spec(spec: dict, ssh, label: str, config, framework_config, results_lock: Lock) -> tuple[bool, bool]:
    """Dispatch to the correct handler; return ``(ok, critical)``.

    ``critical=True`` aborts the session on failure (rocm/pkg); ``critical=False``
    is non-fatal (pytorch) so dependent tests skip/fail gracefully.
    """
    if spec["type"] == "rocm":
        return _install_rocm(spec["value"], ssh, label), True
    if spec["type"] == "pkg":
        return _install_pkgs(spec["value"], ssh, label), True
    if spec["type"] == "pytorch":
        return (
            _install_framework(
                "pytorch",
                spec["value"],
                spec.get("framework_spec"),
                ssh,
                label,
                config,
                framework_config,
                results_lock,
            ),
            False,
        )
    return True, True  # unknown types already warned in _parse_install_specs


def _install_framework(
    framework: str, value: str, parsed_spec, ssh, label: str, config, framework_config, results_lock: Lock
) -> bool:
    """Provision *framework* non-fatally on *label* via the ML provisioner."""
    from framework.executors.cpu_executor import CpuExecutor
    from tests.common.ml_provisioning.fixtures import channels_from_config, with_config_requirements
    from tests.common.ml_provisioning.provisioner import provision_framework, result_to_dict

    spec = with_config_requirements(parsed_spec, framework_config)

    runner = ssh if ssh is not None else CpuExecutor()
    gpu_arch = config.getoption("--gpu-arch", default=None)
    rock_dir = _resolve_rock_dir(config, framework_config)
    rocm_version_hint = getattr(framework_config.therock, "rocm_version", "")
    print(
        f"[pre-install] {label}: starting {framework} provisioning "
        f"(mode={spec.mode}, source={value!r}); progress is logged under "
        f"{framework_config.framework.artifact_dir}/pre-install/{label}/",
        flush=True,
    )
    result = provision_framework(
        runner=runner,
        node_label=label,
        artifact_dir=framework_config.framework.artifact_dir,
        spec=spec,
        channels=channels_from_config(framework_config),
        source="pre-install",
        gpu_arch=gpu_arch,
        rock_dir=rock_dir,
        rocm_version_hint=rocm_version_hint,
        log_name=f"{framework}-pre-install-{label}",
    )
    with results_lock:
        config._framework_provision_results[label] = result_to_dict(result)
    if result.ok:
        action = "reused" if result.skipped_reinstall else "installed"
        packages = _framework_package_summary(result.spec)
        package_detail = f"; {packages}" if packages else ""
        print(
            f"[pre-install] {label}: {framework} {action}{package_detail}; "
            f"python={result.python}; log={result.log_path}"
        )
        return True
    outcome = "runtime incompatible" if result.should_fail_test() else "provisioning unavailable"
    print(f"[pre-install] {label}: {framework} {outcome} non-fatally; log={result.log_path}")
    return False


def _framework_package_summary(spec: dict[str, Any]) -> str:
    """Return a concise framework package version summary for console output."""
    packages = []
    for name in ("torch", "torchvision", "torchaudio"):
        value = spec.get(name, "")
        if value:
            packages.append(f"{name}={value}")
    return ", ".join(packages)


def _shared_results_path(framework_config) -> pathlib.Path:
    return pathlib.Path(framework_config.framework.artifact_dir) / "pre-install" / _SHARED_RESULTS_NAME


def _write_shared_results(config, framework_config) -> None:
    """Persist framework pre-install results for xdist workers."""
    path = _shared_results_path(framework_config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework_preinstall_requested": bool(getattr(config, "_framework_preinstall_requested", False)),
        "framework_provision_results": getattr(config, "_framework_provision_results", {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[pre-install] Framework provision results: {path}")


def _resolve_rock_dir(config, framework_config) -> str:
    return (
        config.getoption("--rock-dir", default=None)
        or os.environ.get("ROCK_DIR")
        or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
        or framework_config.therock.rock_dir
        or ""
    )


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
