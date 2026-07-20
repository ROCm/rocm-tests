# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Execution-node PyTorch provisioning for ROCm tests.

The core handles node discovery, fingerprinted venvs, idempotent reuse, and a
per-node result cache. PyTorch package/validation logic lives in
:mod:`providers`; channel behaviour (``multiarch``/``family`` pip indexes and
explicit ``staging``) is selected per candidate in configurable preference
order. Failures are reported, never raised, so the caller (pre-install plugin or
lazy fixture) can skip/fail gracefully.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import os
import pathlib
import re
import time
from typing import Any, cast

from framework.common.helpers import ExecutionResult

from .providers import FrameworkProvider, get_provider
from .spec import (
    AUTO_PIP_ORDER,
    PIP_MODES,
    ChannelConfig,
    FrameworkSpec,
    auto_spec,
    default_channels,
    gfx_family_for_arch,
    normalize_device_extra,
)

_WORKSPACE_DIR = "run-rocm-tests"
_METADATA_NAME = "rocm-tests-framework.json"
_INVALID_KERNEL_IMAGE_MARKERS = (
    "device kernel image is invalid",
    "hipErrorInvalidImage",
)
_RUNTIME_INCOMPATIBLE_PREFIX = "Framework ROCm runtime incompatible"
_VALIDATION_FAILED_PREFIX = "Framework validation failed"
_INVALID_SPEC_PREFIX = "Invalid framework install spec"


@dataclass
class FrameworkProvisionResult:
    """Result of a framework provision attempt."""

    ok: bool
    source: str
    node_label: str
    framework: str = "pytorch"
    python: str = "python3"
    env: dict[str, str] = field(default_factory=dict)
    spec: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    log_path: str = ""
    error: str = ""
    skipped_reinstall: bool = False

    def skip_reason(self) -> str:
        """Return a concise pytest.skip reason."""
        detail = f"; see {self.log_path}" if self.log_path else ""
        return self.error or f"{self.framework} provisioning failed on {self.node_label}{detail}"

    def should_fail_test(self) -> bool:
        """Return True when provisioning succeeded far enough to expose a test failure."""
        return self.error.startswith(
            (
                _RUNTIME_INCOMPATIBLE_PREFIX,
                _VALIDATION_FAILED_PREFIX,
                _INVALID_SPEC_PREFIX,
            )
        )


# Back-compat alias for the original PyTorch-only result name.
PyTorchProvisionResult = FrameworkProvisionResult


class _ProvisionLog:
    """Small append-only coordinator-side log helper."""

    def __init__(self, path: str) -> None:
        self.path = path
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        """Append *message* plus newline."""
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(message.rstrip() + "\n")

    def command(self, command: str, result: ExecutionResult, display_command: str | None = None) -> None:
        """Append command, exit code, and bounded output.

        When *display_command* is provided it is written instead of the raw
        *command* string — use it to show a short label for long inline scripts
        without losing the full command text from caller-side logging.
        """
        self.write(f"\n$ {display_command if display_command is not None else command}")
        self.write(f"exit_code={result.exit_code} duration={result.duration:.2f}s")
        if result.stdout:
            self.write("[stdout]")
            self.write(result.stdout[-6000:])
        if result.stderr:
            self.write("[stderr]")
            self.write(result.stderr[-6000:])


def _progress(log: _ProvisionLog, message: str) -> None:
    """Emit user-visible provision progress and mirror it to the provision log."""
    log.write(f"progress={message}")
    print(f"[framework-provision] {message}", flush=True)


def _exception_message(exc: Exception) -> str:
    """Return a useful exception message even for exceptions whose string is empty."""
    message = str(exc).strip()
    return message or repr(exc)


def provision_framework(
    *,
    runner,
    node_label: str,
    artifact_dir: str,
    spec: FrameworkSpec | None = None,
    channels: ChannelConfig | None = None,
    source: str = "auto",
    gpu_arch: str | None = None,
    rock_dir: str | None = None,
    rocm_version_hint: str = "",
    log_name: str | None = None,
) -> FrameworkProvisionResult:
    """Install or reuse a node-local framework env; report failures without raising."""
    spec = spec or auto_spec()
    channels = channels or default_channels()
    provider = get_provider(spec.framework)
    safe_label = _safe_name(node_label)
    safe_log = _safe_name(log_name or f"{spec.framework}-{source}-{spec.mode}")
    log = _ProvisionLog(os.path.join(artifact_dir, "pre-install", safe_label, f"{safe_log}.log"))
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    log.write(f"{spec.framework} provisioning started: {started}")
    log.write(f"node={node_label} source={source} raw={spec.raw!r}")

    try:
        _progress(log, f"{node_label}: discovering {spec.framework} provisioning context")
        discovery = _discover(runner, log, gpu_arch=gpu_arch, rock_dir=rock_dir, version_hint=rocm_version_hint)
        log.write(f"discovery={json.dumps(discovery, sort_keys=True)}")

        workspace = _workspace_root(runner, log)
        log.write(f"workspace={workspace}")

        last_error: Exception | None = None
        host_validation: _Validation | None = None
        candidates = _candidate_specs(spec, discovery, channels)
        for idx, normalized in enumerate(candidates):
            log.write(f"normalized_spec={json.dumps(asdict(normalized), sort_keys=True)}")
            _progress(
                log,
                f"{node_label}: evaluating {normalized.framework} candidate "
                f"mode={normalized.mode} index={_effective_index(normalized)}",
            )
            try:
                if host_validation is None:
                    _progress(log, f"{node_label}: checking host python for existing {provider.name}")
                    host_validation = _validate_python(runner, "python3", provider, normalized, log, discovery)
                existing = host_validation
                if existing.ok:
                    log.write("Existing host python satisfies framework requirement; skipping reinstall.")
                    _progress(log, f"{node_label}: host python already satisfies {provider.name}; reusing it")
                    return FrameworkProvisionResult(
                        ok=True,
                        source=source,
                        node_label=node_label,
                        framework=spec.framework,
                        python="python3",
                        spec=asdict(normalized),
                        metadata=existing.metadata,
                        log_path=log.path,
                        skipped_reinstall=True,
                    )

                result = _provision_pip(
                    runner,
                    provider,
                    normalized,
                    discovery,
                    workspace,
                    channels,
                    node_label,
                    source,
                    log,
                    allow_next_candidate=idx < len(candidates) - 1,
                )
                if result is not None:
                    return result
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                log.write(
                    f"candidate_failed mode={normalized.mode!r} "
                    f"index={_effective_index(normalized)!r}: {_exception_message(exc)}"
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError("No framework provisioning candidates were generated")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        error = _exception_message(exc)
        log.write(f"ERROR: {error}")
        return FrameworkProvisionResult(
            ok=False,
            source=source,
            node_label=node_label,
            framework=spec.framework,
            spec=asdict(spec),
            log_path=log.path,
            error=f"{spec.framework} provisioning failed on {node_label}: {error}; see {log.path}",
        )


def provision_pytorch(**kwargs) -> FrameworkProvisionResult:
    """Back-compat wrapper: provision the PyTorch framework."""
    spec = kwargs.get("spec")
    if spec is None:
        kwargs["spec"] = auto_spec("pytorch")
    return provision_framework(**kwargs)


def result_from_dict(data: dict[str, Any]) -> FrameworkProvisionResult:
    """Rehydrate a provision result stored on ``pytest.Config``."""
    return FrameworkProvisionResult(**data)


def result_to_dict(result: FrameworkProvisionResult) -> dict[str, Any]:
    """Serialize a provision result for ``pytest.Config`` storage."""
    return asdict(result)


# ---------------------------------------------------------------------------
# Per-channel provisioning
# ---------------------------------------------------------------------------


def _provision_pip(  # pylint: disable=too-many-arguments
    runner,
    provider: FrameworkProvider,
    normalized: FrameworkSpec,
    discovery: dict[str, str],
    workspace: str,
    channels: ChannelConfig,
    node_label: str,
    source: str,
    log: _ProvisionLog,
    *,
    allow_next_candidate: bool = False,
) -> FrameworkProvisionResult | None:
    """Install into a fingerprinted venv from a pip index (multiarch/family)."""
    _progress(
        log,
        f"{node_label}: selecting {provider.primary_package} wheel from {_effective_index(normalized)}",
    )
    latest = normalized.torch or _select_primary_version(runner, provider, normalized, discovery, log)
    if not latest and not normalized.torch:
        raise RuntimeError(
            f"No installable {provider.primary_package} candidate found for ROCm hint "
            f"{discovery.get('rocm_version', '')!r} from {_effective_index(normalized)}"
        )
    auto_versions = (
        _select_companion_versions(runner, provider, normalized, latest, log)
        if latest and (not normalized.torchvision or not normalized.torchaudio)
        else {}
    )
    install_spec = replace(
        normalized,
        torch=latest or normalized.torch,
        torchvision=normalized.torchvision or auto_versions.get("torchvision", ""),
        torchaudio=normalized.torchaudio or auto_versions.get("torchaudio", ""),
    )
    _progress(
        log,
        f"{node_label}: selected {provider.primary_package}={install_spec.torch or latest or 'unknown'} "
        f"for ROCm hint {discovery.get('rocm_version', '') or 'unknown'}",
    )

    venv_dir = _venv_dir(workspace, install_spec, discovery)
    venv_python = f"{venv_dir}/bin/python"
    metadata_path = f"{venv_dir}/{_METADATA_NAME}"
    log.write(f"venv_dir={venv_dir}")

    metadata = _read_remote_json(runner, metadata_path, log)
    if metadata and metadata.get("fingerprint") == _fingerprint(install_spec, discovery):
        _progress(log, f"{node_label}: found existing managed {provider.name} venv; validating reuse")
        reused = _validate_python(runner, venv_python, provider, install_spec, log, discovery)
        if reused.ok:
            log.write("Existing managed venv satisfies framework requirement; reusing.")
            _progress(log, f"{node_label}: existing managed {provider.name} venv is valid; reusing it")
            return FrameworkProvisionResult(
                ok=True,
                source=source,
                node_label=node_label,
                framework=install_spec.framework,
                python=venv_python,
                spec=asdict(install_spec),
                metadata=reused.metadata,
                log_path=log.path,
                skipped_reinstall=True,
            )

    lock_dir = f"{workspace.rstrip('/')}/output/generated/framework-locks/{_fingerprint(install_spec, discovery)}.lock"
    _progress(log, f"{node_label}: acquiring provision lock for {provider.name} venv {venv_dir}")
    with _remote_lock(runner, lock_dir, log):
        metadata = _read_remote_json(runner, metadata_path, log)
        if metadata and metadata.get("fingerprint") == _fingerprint(install_spec, discovery):
            _progress(log, f"{node_label}: validating managed {provider.name} venv after lock wait")
            reused = _validate_python(runner, venv_python, provider, install_spec, log, discovery)
            if reused.ok:
                log.write("Existing managed venv satisfies framework requirement after lock wait; reusing.")
                _progress(log, f"{node_label}: managed {provider.name} venv became available; reusing it")
                return FrameworkProvisionResult(
                    ok=True,
                    source=source,
                    node_label=node_label,
                    framework=install_spec.framework,
                    python=venv_python,
                    spec=asdict(install_spec),
                    metadata=reused.metadata,
                    log_path=log.path,
                    skipped_reinstall=True,
                )

        _progress(
            log, f"{node_label}: installing {provider.name} wheels into {venv_dir}; this can take several minutes"
        )
        _install_into_venv(runner, provider, venv_dir, venv_python, install_spec, log)
        _progress(log, f"{node_label}: validating installed {provider.name} environment")
        validated = _validate_python(runner, venv_python, provider, install_spec, log, discovery)
        if not validated.ok:
            if _metadata_has_invalid_kernel_image(validated.metadata):
                if normalized.torch:
                    failure = (
                        f"FAIL: {node_label}: pinned {provider.name} wheel failed GPU sanity with "
                        f"hipErrorInvalidImage; not falling back because torch was explicitly requested. "
                        f"{_package_summary(install_spec)}"
                    )
                    log.write(failure)
                    _progress(log, failure)
                    return FrameworkProvisionResult(
                        ok=False,
                        source=source,
                        node_label=node_label,
                        framework=install_spec.framework,
                        spec=asdict(install_spec),
                        metadata=validated.metadata,
                        log_path=log.path,
                        error=_validation_failure_message(node_label, validated.metadata, log.path, discovery),
                    )
                # hipErrorInvalidImage: the installed wheel's GPU code objects do not
                # match this device ISA or firmware.  Try once with the latest available
                # ROCm version from the index (next nightly family up) before giving up.
                warning = (
                    f"WARNING: {node_label}: initial {provider.name} wheel failed GPU sanity with "
                    f"hipErrorInvalidImage; trying fallback. "
                    f"{_package_summary(install_spec)}"
                )
                log.write(warning)
                _progress(log, warning)
                fallback = _fallback_newer_rocm(
                    runner, provider, install_spec, discovery, workspace, channels, node_label, source, log
                )
                if fallback is not None:
                    return fallback
                if allow_next_candidate:
                    warning = (
                        f"WARNING: {node_label}: {provider.name} candidate mode={install_spec.mode} "
                        "failed GPU sanity and fallback did not recover; trying next auto candidate"
                    )
                    log.write(warning)
                    _progress(log, warning)
                    return None
            if allow_next_candidate:
                warning = (
                    f"WARNING: {node_label}: {provider.name} candidate mode={install_spec.mode} failed validation; "
                    "trying next auto candidate"
                )
                log.write(warning)
                _progress(log, warning)
                return None
            return FrameworkProvisionResult(
                ok=False,
                source=source,
                node_label=node_label,
                framework=install_spec.framework,
                spec=asdict(install_spec),
                metadata=validated.metadata,
                log_path=log.path,
                error=_validation_failure_message(node_label, validated.metadata, log.path, discovery),
            )

        final_metadata = {
            "fingerprint": _fingerprint(install_spec, discovery),
            "source": source,
            "node_label": node_label,
            "spec": asdict(install_spec),
            "discovery": discovery,
            "validation": validated.metadata,
        }
        _write_remote_json(runner, metadata_path, final_metadata, log)
        success = f"{node_label}: {provider.name} ready; {_package_summary(install_spec)}"
        log.write(success)
        _progress(log, success)
        return FrameworkProvisionResult(
            ok=True,
            source=source,
            node_label=node_label,
            framework=install_spec.framework,
            python=venv_python,
            spec=asdict(install_spec),
            metadata=final_metadata,
            log_path=log.path,
        )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _run(
    runner, command: str, log: _ProvisionLog, timeout: float = 120.0, display_command: str | None = None
) -> ExecutionResult:
    result = cast(ExecutionResult, runner.run(command, timeout=timeout))
    log.command(command, result, display_command=display_command)
    return result


def _discover(
    runner, log: _ProvisionLog, *, gpu_arch: str | None, rock_dir: str | None, version_hint: str = ""
) -> dict[str, str]:
    py = _run(
        runner,
        _py_cmd("python3", 'import sys; print(str(sys.version_info.major) + "." + str(sys.version_info.minor))'),
        log,
    )
    python_version = py.stdout.strip().splitlines()[-1] if py.ok and py.stdout.strip() else ""

    rocm_version, rocm_source = _discover_rocm_version(runner, log, rock_dir=rock_dir, version_hint=version_hint)

    arch = gpu_arch or ""
    # ISA string (e.g. "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-") for diagnostics.
    isa_string = ""
    if not arch:
        isa_code = (
            "import re, sys; "
            "text=sys.stdin.read(); "
            "m=re.search(r'Name:\\s+(gfx[0-9A-Za-z]+)', text); "
            "isa=re.search(r'(amdgcn-amd-amdhsa--gfx[^\\s]+)', text); "
            "print((m.group(1) if m else '') + '|' + (isa.group(1) if isa else ''))"
        )
        arch_result = _run(
            runner,
            f"rocminfo 2>/dev/null | {_py_cmd('python3', isa_code)} || true",
            log,
            timeout=30.0,
        )
        raw = arch_result.stdout.strip().splitlines()[-1] if arch_result.stdout.strip() else ""
        if "|" in raw:
            arch, isa_string = raw.split("|", 1)
        else:
            arch = raw
    arch = normalize_device_extra(arch)

    return {
        "python_version": python_version,
        "rocm_version": rocm_version,
        "rocm_source": rocm_source,
        "rock_dir": rock_dir or "",
        "gfx_arch": arch,
        "gfx_family": gfx_family_for_arch(arch) if arch else "",
        "isa_string": isa_string,
    }


def _discover_rocm_version(
    runner, log: _ProvisionLog, *, rock_dir: str | None, version_hint: str = ""
) -> tuple[str, str]:
    # Priority 0: explicit hint from rocm-test.toml [therock] rocm_version.
    # Lets operators pin the exact date-format version when the build tree only
    # exposes a build-number (e.g. hipconfig returns 7.14.60850).
    if version_hint:
        log.write(f"rocm_version_hint from config: {version_hint!r}")
        return version_hint, "config_hint"

    if rock_dir:
        rock = _run(runner, _rock_dir_version_cmd(rock_dir), log, timeout=90.0)
        version = _first_rocm_version(rock.stdout)
        if version:
            return version, "rock_dir"

    torch_hip_cmd = _py_cmd("python3", 'import torch; print(getattr(torch.version, "hip", "") or "")')
    rocm = _run(
        runner,
        f"python3 -m rocm_sdk version 2>/dev/null || {torch_hip_cmd} 2>/dev/null || true",
        log,
    )
    return _first_rocm_version(rocm.stdout), "python"


def _rock_dir_version_cmd(rock_dir: str) -> str:
    code = (
        "import json, pathlib, subprocess; "
        f"root=pathlib.Path({_py(rock_dir)}); "
        "texts=[]; "
        # Priority 0: therock_manifest.json carries rocm_package_version in
        # exact date format (e.g. '7.14.0a20260612') — always prefer this over
        # hipconfig which only exposes the build-number (e.g. '7.14.60850').
        "manifest=root/'share'/'therock'/'therock_manifest.json'; "
        "\nif manifest.is_file():\n"
        "    try:\n"
        "        d=json.loads(manifest.read_text(errors='ignore'))\n"
        "        pkg_ver=d.get('rocm_package_version','')\n"
        "        if pkg_ver:\n"
        "            texts.insert(0, f'ROCm {pkg_ver}')\n"
        "    except Exception:\n"
        "        pass\n"
        "candidates=[root/'.info'/'version-rocm']; "
        "\nif (root/'.info').exists():\n"
        "    candidates.extend(sorted((root/'.info').glob('version-*')))\n"
        "candidates.extend(root.glob('lib/python*/site-packages/rocm*.dist-info/METADATA'))\n"
        "for path in candidates:\n"
        "    if not path.is_file():\n"
        "        continue\n"
        "    try:\n"
        "        texts.append(path.read_text(errors='ignore')[:4000])\n"
        "    except Exception:\n"
        "        pass\n"
        "bins=[root/'bin'/'hipconfig', root/'bin'/'hipcc', root/'bin'/'rocminfo']; "
        "\nfor exe in bins:\n"
        "    if exe.exists():\n"
        "        for arg in ('--version','version'):\n"
        "            try:\n"
        "                out=subprocess.run([str(exe), arg], text=True, capture_output=True, timeout=15)\n"
        "                texts.append(out.stdout + '\\n' + out.stderr)\n"
        "            except Exception:\n"
        "                pass\n"
        "print('\\n'.join(texts))"
    )
    return f"test -d {_dq(rock_dir)} && {_py_cmd('python3', code)} || true"


# ---------------------------------------------------------------------------
# Candidate specs
# ---------------------------------------------------------------------------


def _normalize_spec(spec: FrameworkSpec, discovery: dict[str, str], channels: ChannelConfig) -> FrameworkSpec:
    mode = spec.mode
    device = spec.device or discovery.get("gfx_arch", "")
    gfx_family = spec.gfx_family or discovery.get("gfx_family", "")
    index_url = spec.index_url
    if not index_url and mode in PIP_MODES:
        if mode == "family":
            index_url = channels.family_index_base
        elif mode == "staging":
            # staging uses a separate pre-promotion multi-arch index; otherwise
            # behaves identically to multiarch (pip extras, fingerprinted venv).
            index_url = channels.staging_index
        else:
            index_url = channels.multiarch_index
    return replace(spec, mode=mode, index_url=index_url, device=device, gfx_family=gfx_family)


def _candidate_specs(spec: FrameworkSpec, discovery: dict[str, str], channels: ChannelConfig) -> list[FrameworkSpec]:
    """Return install candidates in preference order for auto mode.

    Explicit ``mode=``/``index=``/``find_links=`` bypasses the loop (single
    candidate). Auto mode is intentionally limited to the production wheel path:
    multi-arch wheels first, then family-specific v2 wheels.
    """
    if spec.mode != "auto" or spec.index_url or spec.find_links_url:
        return [_normalize_spec(spec, discovery, channels)]

    candidates: list[FrameworkSpec] = []
    for mode in AUTO_PIP_ORDER:
        candidates.append(_normalize_spec(replace(spec, mode=mode), discovery, channels))
    return candidates


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class _Validation:
    ok: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def _validate_python(
    runner,
    python: str,
    provider: FrameworkProvider,
    spec: FrameworkSpec,
    log: _ProvisionLog,
    discovery: dict[str, str] | None = None,
) -> _Validation:
    cmd = _py_cmd(python, provider.sanity_snippet(spec))
    rock_dir = (discovery or {}).get("rock_dir", "")
    if rock_dir:
        cmd = f"env LD_LIBRARY_PATH={_dq(str(pathlib.PurePosixPath(rock_dir) / 'lib'))} {cmd}"
    result = _run(
        runner,
        cmd,
        log,
        timeout=90.0,
        display_command=f'{python} -c "<{spec.framework} sanity check>"',
    )
    metadata = _json_from_stdout(result.stdout)
    ok = result.ok
    _log_sanity_summary(log, python, spec.framework, metadata, ok)
    return _Validation(ok=ok, metadata=metadata)


def _log_sanity_summary(log: _ProvisionLog, python: str, framework: str, metadata: dict[str, Any], ok: bool) -> None:
    """Write a concise human-readable sanity-check result to the provision log."""
    smoke = "PASS" if ok else "FAIL"
    torch_ver = metadata.get("torch_version", "?")
    hip_ver = metadata.get("torch_hip", "?")
    cuda = metadata.get("cuda_available", False)
    device = metadata.get("device_name", "")
    device_str = f' device="{device}"' if device else ""
    bar = "=" * 60
    tag = framework.upper()
    if ok:
        log.write(
            f"\n{bar}\n"
            f"  {tag} SANITY CHECK PASSED\n"
            f"  python : {python}\n"
            f"  torch  : {torch_ver}\n"
            f"  hip    : {hip_ver}\n"
            f"  cuda   : {cuda}{device_str}\n"
            f"  smoke  : {smoke}\n"
            f"{bar}"
        )
    else:
        error = metadata.get("error", "")
        log.write(
            f"\n{bar}\n"
            f"  {tag} SANITY CHECK FAILED\n"
            f"  torch  : {torch_ver}\n"
            f"  hip    : {hip_ver}\n"
            f"  cuda   : {cuda}{device_str}\n"
            f"  smoke  : {smoke}\n" + (f"  reason : {error}\n" if error else "") + f"{bar}"
        )


def _validation_failure_message(
    node_label: str, metadata: dict[str, Any], log_path: str, discovery: dict[str, str] | None = None
) -> str:
    """Return a specific provisioning diagnostic for failed validation."""
    detail = _validation_metadata_detail(metadata)
    log_detail = f"; see {log_path}" if log_path else ""
    if _metadata_has_invalid_kernel_image(metadata):
        isa = (discovery or {}).get("isa_string", "")
        isa_detail = f", isa={isa}" if isa else ""
        return (
            f"{_RUNTIME_INCOMPATIBLE_PREFIX} on {node_label}: device smoke kernel failed with "
            f"hipErrorInvalidImage after the framework installed successfully{detail}{isa_detail}. "
            "The GPU code object in the selected wheel does not match this device ISA or firmware; "
            f"dependent tests must FAIL rather than SKIP{log_detail}"
        )
    return f"{_VALIDATION_FAILED_PREFIX} on {node_label}{detail}{log_detail}"


def _metadata_has_invalid_kernel_image(metadata: dict[str, Any]) -> bool:
    combined = json.dumps(metadata, sort_keys=True)
    return any(marker in combined for marker in _INVALID_KERNEL_IMAGE_MARKERS)


def _validation_metadata_detail(metadata: dict[str, Any]) -> str:
    pieces = [
        ("torch", metadata.get("torch_version")),
        ("torch_hip", metadata.get("torch_hip")),
        ("cuda_available", metadata.get("cuda_available")),
        ("version_matches", metadata.get("version_matches")),
    ]
    rendered = [f"{name}={value}" for name, value in pieces if value not in (None, "")]
    return f" ({', '.join(rendered)})" if rendered else ""


# ---------------------------------------------------------------------------
# Workspace + venv + install
# ---------------------------------------------------------------------------


def _workspace_root(runner, log: _ProvisionLog) -> str:
    if hasattr(runner, "remote_workspace_root"):
        root = cast(str, runner.remote_workspace_root())
    else:
        result = _run(runner, 'printf "%s" "$HOME"', log, timeout=30.0)
        home = result.stdout.strip() if result.ok else "$HOME"
        root = f"{home.rstrip('/')}/{_WORKSPACE_DIR}"
    _run(
        runner,
        f"mkdir -p {_dq(root)}/output/generated/framework-envs "
        f"{_dq(root)}/output/generated/framework-cache "
        f"{_dq(root)}/output/generated/framework-locks",
        log,
    )
    return root


def _venv_dir(workspace: str, spec: FrameworkSpec, discovery: dict[str, str]) -> str:
    return f"{workspace.rstrip('/')}/output/generated/framework-envs/{_fingerprint(spec, discovery)}"


def _fingerprint(spec: FrameworkSpec, discovery: dict[str, str]) -> str:
    payload = {
        "spec": asdict(spec),
        "python_version": discovery.get("python_version", ""),
        "rocm_version": discovery.get("rocm_version", ""),
        "gfx_arch": discovery.get("gfx_arch", ""),
        "gfx_family": discovery.get("gfx_family", ""),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _select_primary_version(
    runner, provider: FrameworkProvider, spec: FrameworkSpec, discovery: dict[str, str], log: _ProvisionLog
) -> str:
    index = _effective_index(spec)
    versions = _query_package_versions(runner, provider.primary_package, index, log, pre=spec.pre)
    rocm_versions = set(_query_package_versions(runner, "rocm", index, log, pre=spec.pre))
    candidates = _rank_torch_versions(versions, discovery.get("rocm_version", ""), rocm_versions)
    selected = candidates[0] if candidates else ""
    log.write(
        "selected_primary="
        f"{selected!r} rocm_hint={discovery.get('rocm_version', '')!r} "
        f"rocm_source={discovery.get('rocm_source', '')!r} index={index}"
    )
    if versions and rocm_versions and selected != versions[0]:
        skipped = [v for v in versions[:5] if _torch_rocm_version(v) and _torch_rocm_version(v) not in rocm_versions]
        if skipped:
            log.write(f"skipped_versions_without_matching_rocm_dependency={skipped}")
    return selected


def _select_companion_versions(
    runner, provider: FrameworkProvider, spec: FrameworkSpec, primary_version: str, log: _ProvisionLog
) -> dict[str, str]:
    primary_rocm = _torch_rocm_version(primary_version)
    if not primary_rocm:
        return {}

    index = _effective_index(spec)
    selected: dict[str, str] = {}
    for package in provider.companion_packages:
        versions = _query_package_versions(runner, package, index, log, pre=spec.pre)
        candidates = [version for version in versions if _torch_rocm_version(version) == primary_rocm]
        if candidates:
            selected[package] = candidates[0]
            log.write(f"selected_{package}={candidates[0]!r} for rocm={primary_rocm!r}")
        else:
            log.write(f"selected_{package}='' no candidate for rocm={primary_rocm!r}")
    return selected


def _query_package_versions(runner, package: str, index: str, log: _ProvisionLog, *, pre: bool = False) -> list[str]:
    pre_arg = " --pre" if pre else ""
    cmd = f"python3 -m pip index versions {package}{pre_arg} --index-url {_dq(index)}"
    result = _run(runner, cmd, log, timeout=120.0)
    versions = _parse_pip_versions(result.stdout + "\n" + result.stderr)
    log.write(f"{package}_versions_count={len(versions)} index={index}")
    return versions


def _rank_torch_versions(versions: list[str], rocm_hint: str, rocm_versions: set[str]) -> list[str]:
    # When the hint is in build-number format (e.g. "7.14.60850" from hipconfig),
    # _rocm_build_date() returns 0 and date-proximity ranking is silently disabled.
    # Resolve an effective date-format hint from the available rocm package versions
    # on the same pip index: pick the latest rocm package in the same family so that
    # the date-proximity scoring tier works correctly.
    effective_hint = rocm_hint
    if rocm_hint and not _rocm_build_date(rocm_hint) and _BUILD_NUM_VERSION_RE.fullmatch(rocm_hint):
        hint_family = _rocm_family(rocm_hint)
        if hint_family:
            family_candidates = [v for v in rocm_versions if _rocm_family(v) == hint_family and _rocm_build_date(v)]
            if family_candidates:
                effective_hint = max(family_candidates, key=_rocm_build_date)

    def usable(version: str) -> bool:
        torch_rocm = _torch_rocm_version(version)
        return not torch_rocm or not rocm_versions or torch_rocm in rocm_versions

    return sorted(
        [version for version in versions if usable(version)],
        key=lambda version: _torch_version_score(version, effective_hint, versions.index(version)),
    )


def _torch_version_score(version: str, rocm_hint: str, index: int) -> tuple[int, int, int, int, int]:
    torch_rocm = _torch_rocm_version(version)
    prerelease_penalty = int(_torch_core_is_prerelease(version))
    if not rocm_hint:
        return (0, prerelease_penalty, 0, 0, index)
    if torch_rocm == rocm_hint:
        return (0, prerelease_penalty, 0, 0, index)
    if not torch_rocm:
        return (4, 1, prerelease_penalty, 0, index)

    hint_family = _rocm_family(rocm_hint)
    torch_family = _rocm_family(torch_rocm)
    if hint_family and torch_family == hint_family:
        hint_date = _rocm_build_date(rocm_hint)
        torch_date = _rocm_build_date(torch_rocm)
        if hint_date and torch_date:
            newer_than_hint = int(torch_date > hint_date)
            return (1, newer_than_hint, prerelease_penalty, abs(torch_date - hint_date), index)
        return (1, prerelease_penalty, 0, 0, index)

    torch_parts = _pad_parts(_numeric_parts(torch_rocm))
    hint_parts = _pad_parts(_numeric_parts(rocm_hint))
    distance = sum(abs(left - right) for left, right in zip(torch_parts, hint_parts, strict=False))
    return (3, prerelease_penalty, 0, distance, index)


def _torch_rocm_version(version: str) -> str:
    match = re.search(r"\+rocm([0-9][A-Za-z0-9.]*)", version)
    return match.group(1) if match else ""


def _torch_core_is_prerelease(version: str) -> bool:
    core = version.split("+", 1)[0]
    return bool(re.search(r"(?:a|b|rc)\d*", core))


def _rocm_family(version: str) -> tuple[int, int] | None:
    parts = _numeric_parts(version)
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def _rocm_build_date(version: str) -> int:
    match = re.search(r"(?:a|~)(\d{8})", version)
    return int(match.group(1)) if match else 0


def _numeric_parts(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", version))


def _pad_parts(parts: tuple[int, ...], size: int = 5) -> tuple[int, ...]:
    return parts + (0,) * max(size - len(parts), 0)


def _install_into_venv(
    runner, provider: FrameworkProvider, venv_dir: str, python: str, spec: FrameworkSpec, log: _ProvisionLog
) -> None:
    _progress(log, f"creating managed {provider.name} virtualenv: {venv_dir}")
    create_cmd = f"rm -rf {_dq(venv_dir)} && mkdir -p {_dq(venv_dir)} && python3 -m venv {_dq(venv_dir)}"
    _ensure_ok(runner, create_cmd, log, timeout=300.0)
    installer = _prepare_installer(runner, provider, venv_dir, python, log)

    # Stale-wheel guard: drop any lingering (possibly CUDA) framework wheels so pip
    # does not treat them as "already satisfied" when switching index/channel.
    stale = _stale_packages(provider)
    if stale:
        _run(runner, f"{_dq(python)} -m pip uninstall -y {' '.join(_dq(p) for p in stale)}", log, timeout=300.0)

    packages = provider.packages(spec)
    install_parts = [_dq(python), "-m", "pip", "install"]
    if spec.pre:
        install_parts.append("--pre")
    index = _effective_index(spec)
    if index:
        install_parts.extend(["--index-url", _dq(index)])
    if spec.find_links_url:
        install_parts.extend(["--find-links", _dq(spec.find_links_url)])
    install_parts.extend(_dq(pkg) for pkg in packages)
    _progress(log, f"installing {provider.name} packages: {', '.join(packages)}")
    _run_install_with_fallback(
        runner,
        " ".join(install_parts),
        _uv_install_command(installer, install_parts[4:]),
        log,
        timeout=1800.0,
        label=f"{provider.name} packages",
    )

    for req in spec.requirements:
        _install_requirement(runner, python, venv_dir, req, log, installer)


def _install_requirement(
    runner, python: str, venv_dir: str, req: str, log: _ProvisionLog, installer: str | None
) -> None:
    """Install a requirements file, staging local files for remote runners."""
    _progress(log, f"installing framework requirement file: {req}")
    local_path = pathlib.Path(req)
    if local_path.is_file():
        content = local_path.read_text(encoding="utf-8")
        remote_req = f"{venv_dir}/requirements-{_safe_name(local_path.name)}"
        _write_remote_text(runner, remote_req, content, log)
        pip_cmd = f"{_dq(python)} -m pip install -r {_dq(remote_req)}"
        uv_cmd = _uv_install_command(installer, ["-r", _dq(remote_req)])
        _run_install_with_fallback(runner, pip_cmd, uv_cmd, log, timeout=900.0, label=f"requirements {req}")
        return

    pip_cmd = f"{_dq(python)} -m pip install -r {_dq(req)}"
    uv_cmd = _uv_install_command(installer, ["-r", _dq(req)])
    _run_install_with_fallback(runner, pip_cmd, uv_cmd, log, timeout=900.0, label=f"requirements {req}")


def _prepare_installer(
    runner, provider: FrameworkProvider, venv_dir: str, python: str, log: _ProvisionLog
) -> str | None:
    """Install uv into the target-node venv; return a uv command prefix or None for pip-only."""
    _progress(log, f"upgrading pip tooling in managed {provider.name} virtualenv")
    _ensure_ok(runner, f"{_dq(python)} -m pip install --upgrade pip wheel setuptools", log, timeout=300.0)

    uv_path = f"{venv_dir.rstrip('/')}/bin/uv"
    _progress(log, f"installing uv accelerator in managed {provider.name} virtualenv")
    result = _run(runner, f"{_dq(python)} -m pip install --upgrade uv", log, timeout=300.0)
    if not result.ok:
        log.write(
            f"WARNING: uv bootstrap failed for {provider.name}; falling back to pip " f"(exit={result.exit_code})"
        )
        _progress(log, f"WARNING: uv bootstrap failed for {provider.name}; using pip installer")
        return None

    probe = _run(runner, f"{_dq(uv_path)} --version", log, timeout=30.0)
    if not probe.ok:
        log.write(f"WARNING: uv probe failed for {provider.name}; falling back to pip (exit={probe.exit_code})")
        _progress(log, f"WARNING: uv probe failed for {provider.name}; using pip installer")
        return None

    version = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else "uv"
    log.write(f"using_installer={version}")
    _progress(log, f"using uv installer for {provider.name}: {version}")
    return f"{_dq(uv_path)} pip install --python {_dq(python)}"


def _uv_install_command(installer: str | None, args: list[str]) -> str | None:
    """Build a uv pip install command for args already shell-quoted as needed."""
    if installer is None:
        return None
    return f"{installer} {' '.join(args)}"


def _run_install_with_fallback(
    runner,
    pip_cmd: str,
    uv_cmd: str | None,
    log: _ProvisionLog,
    *,
    timeout: float,
    label: str,
) -> None:
    """Run uv install by default; fall back to pip if uv is unavailable or fails."""
    if uv_cmd is not None:
        _progress(log, f"installing {label} with uv")
        result = _run(runner, uv_cmd, log, timeout=timeout)
        if result.ok:
            return
        log.write(f"WARNING: uv install failed for {label}; retrying with pip (exit={result.exit_code})")
        _progress(log, f"WARNING: uv install failed for {label}; retrying with pip")

    _progress(log, f"installing {label} with pip")
    _ensure_ok(runner, pip_cmd, log, timeout=timeout)


def _stale_packages(provider: FrameworkProvider) -> list[str]:
    """Return package names to uninstall before a fresh install (ABI safety)."""
    if provider.name == "pytorch":
        return ["torch", "torchvision", "torchaudio", "pytorch-triton-rocm", "triton"]
    return []


def _package_summary(spec: FrameworkSpec) -> str:
    """Return concise selected package versions for progress messages."""
    packages = []
    for name in ("torch", "torchvision", "torchaudio"):
        value = getattr(spec, name)
        if value:
            packages.append(f"{name}={value}")
    return ", ".join(packages) or f"mode={spec.mode}"


def _fallback_newer_rocm(
    runner,
    provider: FrameworkProvider,
    original_spec: FrameworkSpec,
    discovery: dict[str, str],
    workspace: str,
    channels: ChannelConfig,
    node_label: str,
    source: str,
    log: _ProvisionLog,
) -> FrameworkProvisionResult | None:
    """Retry with the latest available ROCm version when a wheel fails hipErrorInvalidImage.

    ``hipErrorInvalidImage`` means the code objects bundled in the matched wheel
    are incompatible with the device ISA or firmware — not necessarily that the
    version *selection* was wrong.  A newer nightly may ship code objects built
    with updated target flags that work.  This function queries the index for the
    single latest stable ``torch`` wheel (any ROCm), installs it, and re-runs the
    smoke test.  Returns the successful :class:`FrameworkProvisionResult` or
    ``None`` if the fallback also fails (caller then reports the original error).
    """
    index = _effective_index(original_spec)
    versions = _query_package_versions(runner, provider.primary_package, index, log, pre=original_spec.pre)
    if not versions:
        log.write("fallback_newer_rocm: no versions available on index — giving up")
        return None

    # Find the latest version that is DIFFERENT from the one we just tried.
    original_torch = original_spec.torch or ""
    candidates = [v for v in versions if v != original_torch]
    if not candidates:
        log.write("fallback_newer_rocm: no alternative versions available — giving up")
        return None

    fallback_torch = candidates[0]
    log.write(
        f"fallback_newer_rocm: original={original_torch!r} failed with hipErrorInvalidImage; "
        f"retrying with latest available={fallback_torch!r}"
    )
    _progress(
        log,
        f"fallback: retrying {provider.name} with latest available torch={fallback_torch}",
    )

    auto_versions = _select_companion_versions(runner, provider, original_spec, fallback_torch, log)
    from dataclasses import replace  # pylint: disable=import-outside-toplevel

    fallback_spec = replace(
        original_spec,
        torch=fallback_torch,
        torchvision=auto_versions.get("torchvision", ""),
        torchaudio=auto_versions.get("torchaudio", ""),
    )

    # Re-use a distinct venv dir (different fingerprint due to new torch version).
    fallback_discovery = dict(discovery)
    fallback_discovery["rocm_version"] = _torch_rocm_version(fallback_torch)
    venv_dir = _venv_dir(workspace, fallback_spec, fallback_discovery)
    venv_python = f"{venv_dir}/bin/python"
    metadata_path = f"{venv_dir}/{_METADATA_NAME}"

    fingerprint = _fingerprint(fallback_spec, fallback_discovery)
    lock_dir = f"{workspace.rstrip('/')}/output/generated/framework-locks/{fingerprint}.lock"
    try:
        with _remote_lock(runner, lock_dir, log):
            metadata = _read_remote_json(runner, metadata_path, log)
            if metadata and metadata.get("fingerprint") == _fingerprint(fallback_spec, fallback_discovery):
                reused = _validate_python(runner, venv_python, provider, fallback_spec, log, fallback_discovery)
                if reused.ok:
                    log.write(f"fallback_newer_rocm: reused existing fallback env torch={fallback_torch!r}")
                    return FrameworkProvisionResult(
                        ok=True,
                        source=source,
                        node_label=node_label,
                        framework=fallback_spec.framework,
                        python=venv_python,
                        spec=asdict(fallback_spec),
                        metadata=reused.metadata,
                        log_path=log.path,
                        skipped_reinstall=True,
                    )

            _install_into_venv(runner, provider, venv_dir, venv_python, fallback_spec, log)
            validated = _validate_python(runner, venv_python, provider, fallback_spec, log, fallback_discovery)
            if not validated.ok:
                failure = (
                    f"FAIL: fallback {provider.name} wheel failed GPU sanity; "
                    f"{_package_summary(fallback_spec)}; error={validated.metadata.get('error', '')!r}"
                )
                log.write(failure)
                _progress(log, failure)
                return None

            final_metadata = {
                "fingerprint": _fingerprint(fallback_spec, fallback_discovery),
                "source": source,
                "node_label": node_label,
                "spec": asdict(fallback_spec),
                "discovery": fallback_discovery,
                "validation": validated.metadata,
                "fallback_reason": "hipErrorInvalidImage on original wheel",
                "original_torch": original_torch,
            }
            _write_remote_json(runner, metadata_path, final_metadata, log)
            success = f"fallback_newer_rocm: succeeded; {_package_summary(fallback_spec)}"
            log.write(success)
            _progress(log, f"{node_label}: {provider.name} fallback ready; {_package_summary(fallback_spec)}")
            return FrameworkProvisionResult(
                ok=True,
                source=source,
                node_label=node_label,
                framework=fallback_spec.framework,
                python=venv_python,
                spec=asdict(fallback_spec),
                metadata=final_metadata,
                log_path=log.path,
            )
    except RuntimeError as exc:
        log.write(f"fallback_newer_rocm: install failed: {exc}")
        return None

    return None


def _ensure_ok(runner, command: str, log: _ProvisionLog, timeout: float) -> ExecutionResult:
    result = _run(runner, command, log, timeout=timeout)
    if not result.ok:
        raise RuntimeError(f"command failed with exit code {result.exit_code}: {command}")
    return result


def _effective_index(spec: FrameworkSpec) -> str:
    index = spec.index_url.rstrip("/")
    if spec.mode == "family" and spec.gfx_family and not index.endswith(spec.gfx_family):
        index = f"{index}/{spec.gfx_family}"
    return f"{index}/" if index else ""


# ---------------------------------------------------------------------------
# Remote helpers
# ---------------------------------------------------------------------------


def _read_remote_json(runner, path: str, log: _ProvisionLog) -> dict[str, Any]:
    code = f"import json; print(json.dumps(json.load(open({_py(path)}))))"
    result = _run(runner, f"test -f {_dq(path)} && {_py_cmd('python3', code)} || true", log)
    return _json_from_stdout(result.stdout)


def _write_remote_json(runner, path: str, data: dict[str, Any], log: _ProvisionLog) -> None:
    payload = _dq(json.dumps(data, sort_keys=True))
    cmd = f"mkdir -p {_dq(os.path.dirname(path))} && printf %s {payload} > {_dq(path)}"
    _run(runner, cmd, log)


def _write_remote_text(runner, path: str, data: str, log: _ProvisionLog) -> None:
    payload = _dq(data)
    cmd = f"mkdir -p {_dq(os.path.dirname(path))} && printf %s {payload} > {_dq(path)}"
    _ensure_ok(runner, cmd, log, timeout=30.0)


class _RemoteLock:
    """Atomic mkdir-based lock on the execution host."""

    def __init__(self, runner, lock_dir: str, log: _ProvisionLog, timeout_secs: int = 1800) -> None:
        self.runner = runner
        self.lock_dir = lock_dir
        self.log = log
        self.timeout_secs = timeout_secs

    def __enter__(self):
        deadline = time.monotonic() + self.timeout_secs
        next_notice = 0.0
        while True:
            cmd = (
                "bash -c '"
                f"lock={_dq(self.lock_dir)}; "
                'mkdir -p "$(dirname "$lock")"; '
                'if mkdir "$lock" 2>/dev/null; then '
                '  printf "%s" "$$" > "$lock/pid"; '
                '  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$lock/created_at"; '
                "  exit 0; "
                "fi; "
                'printf "framework-lock-held:%s" "$lock"; '
                'test -f "$lock/created_at" && printf " created_at=%s" "$(cat "$lock/created_at")"; '
                'test -f "$lock/pid" && printf " pid=%s" "$(cat "$lock/pid")"; '
                "exit 75"
                "'"
            )
            result = _run(self.runner, cmd, self.log, timeout=30.0)
            if result.ok:
                break
            if result.exit_code != 75:
                raise RuntimeError(
                    f"Failed to acquire framework lock {self.lock_dir}: {result.stderr or result.stdout}"
                )

            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for framework lock {self.lock_dir}; "
                    "another provisioning process may still be running or a prior run may have been interrupted"
                )
            if now >= next_notice:
                detail = (result.stdout or result.stderr).strip()
                suffix = f" ({detail})" if detail else ""
                _progress(
                    self.log,
                    f"waiting for provision lock {self.lock_dir}; another process may be installing this env{suffix}",
                )
                next_notice = now + 30.0
            time.sleep(5.0)
        self.log.write(f"acquired_lock={self.lock_dir}")
        _progress(self.log, f"acquired provision lock {self.lock_dir}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        result = _run(self.runner, f"rm -rf {_dq(self.lock_dir)}", self.log, timeout=30.0)
        if result.ok:
            self.log.write(f"released_lock={self.lock_dir}")
            _progress(self.log, f"released provision lock {self.lock_dir}")


def _remote_lock(runner, lock_dir: str, log: _ProvisionLog) -> _RemoteLock:
    return _RemoteLock(runner, lock_dir, log)


def _parse_pip_versions(stdout: str) -> list[str]:
    for line in stdout.splitlines():
        if line.startswith("Available versions:"):
            raw = line.replace("Available versions:", "", 1)
            return [v.strip() for v in raw.split(",") if v.strip()]
    return []


def _first_version(text: str) -> str:
    match = re.search(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", text)
    return match.group(0) if match else ""


# Matches nightly date-format versions: 7.14.0a20260624
_DATE_VERSION_RE = re.compile(r"\d+\.\d+\.\d+[a-z]\d{8}")
# Matches build-number-format versions: 7.14.60850 (hipconfig on TheRock builds)
_BUILD_NUM_VERSION_RE = re.compile(r"\d+\.\d+\.\d{5,}")


def _first_rocm_version(text: str) -> str:
    """Extract the ROCm version string from arbitrary text.

    Prefers the nightly date-format (``7.14.0a20260624``) over the
    build-number format (``7.14.60850``) when both are present.  TheRock
    ``hipconfig --version`` emits the build-number form; the ``rocm`` pip
    package and ``.info/version-rocm`` files may emit either form.  The
    date-format maps directly to PyPI wheel version suffixes so it produces
    exact matches in :func:`_rank_torch_versions`; the build-number form
    only allows family-level matching.
    """
    # Pass 1: scan keyword lines for a date-format version first.
    for line in text.splitlines():
        if re.search(r"\b(ROCm|HIP|rocm-core|rocm_sdk)\b", line, flags=re.IGNORECASE):
            m = _DATE_VERSION_RE.search(line)
            if m:
                return m.group(0)

    # Pass 2: scan ALL lines for a date-format version (e.g. from .info files).
    for line in text.splitlines():
        m = _DATE_VERSION_RE.search(line)
        if m:
            return m.group(0)

    # Pass 3: fall back to any keyword line with a build-number or plain version.
    for line in text.splitlines():
        if re.search(r"\b(ROCm|HIP|rocm-core|rocm_sdk)\b", line, flags=re.IGNORECASE):
            version = _first_version(line)
            if version:
                return version

    return _first_version(text)


def _json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            return cast(dict[str, Any], json.loads(line))
        except json.JSONDecodeError:
            continue
    return {}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def _dq(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return f'"{escaped}"'


def _py(value: str) -> str:
    return json.dumps(value)


def _py_cmd(python: str, code: str) -> str:
    return f"{_dq(python)} -c {_dq(code)}"
