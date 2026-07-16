# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared PyTorch pytest fixture helpers.

Two paths feed the same provisioner core through :func:`ensure_pytorch_env`:

**Path A — lazy / test-scoped (default, no ``--pre-install``):**
    The ``pytorch_env`` fixture is function-scoped.  The first test that
    requests it runs Phase 3 (lazy install from the channels configured in
    ``rocm-test.toml``).  If install and sanity both pass, the result is written
    to ``config._framework_sanity_ok`` — a session-lived dict keyed by
    ``"pytorch:{node_label}"``.  Every subsequent fixture call hits Phase 1
    (O(1) dict lookup, no subprocess).  If the install or sanity fails the cache
    is **not** written, so the next test retries Phase 3 (transient-failure
    recovery semantics).

**Path B — explicit / session-scoped (``--pre-install pytorch=...``):**
    ``install_plugin.pytest_sessionstart`` runs before collection and stores the
    provision result in ``config._framework_provision_results[node_label]``.
    The fixture's Phase 2 detects this, and on success promotes the result into
    ``config._framework_sanity_ok`` so that all tests share it via Phase 1.
    A failed pre-install result is never promoted — tests fail
    permanently for that session (the pre-install outcome is definitive).

Phase 1 is the shared fast path: after the first success from either path,
every subsequent fixture invocation short-circuits via the sanity cache.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
import pathlib
import time

import pytest

from .provisioner import FrameworkProvisionResult, provision_framework, result_from_dict, result_to_dict
from .spec import VALID_MODES, ChannelConfig, FrameworkSpec, auto_spec

# Name of the session-level sanity cache attribute on pytest.Config.
# Written ONLY after a successful provision AND sanity check (result.ok is True).
# Key: f"{framework}:{node_label}" (e.g. "pytorch:localhost")
# Value: result_to_dict(result) where result.ok is True
_SANITY_CACHE = "_framework_sanity_ok"
_SHARED_RESULTS_NAME = "framework-provision-results.json"


def channels_from_config(framework_config) -> ChannelConfig:
    """Build a :class:`ChannelConfig` from the ``[frameworks]`` config section.

    Falls back to code defaults when the section is absent (older config files).
    Channel URLs come from ``rocm-test.toml`` so no index URL is hardcoded in
    fixture or test code. The ``auto`` order is fixed in the provisioner.
    """
    section = getattr(framework_config, "frameworks", None)
    if section is None:
        return ChannelConfig()
    return ChannelConfig(
        multiarch_index=getattr(section, "multiarch_index", "") or ChannelConfig().multiarch_index,
        family_index_base=getattr(section, "family_index_base", "") or ChannelConfig().family_index_base,
        staging_index=getattr(section, "staging_index", "") or ChannelConfig().staging_index,
    )


def spec_from_config(framework_config, framework: str = "pytorch") -> FrameworkSpec:
    """Build the lazy-provisioning spec from ``[frameworks]`` defaults."""
    section = getattr(framework_config, "frameworks", None)
    mode = getattr(section, "default_mode", "auto") if section is not None else "auto"
    if mode not in VALID_MODES:
        raise pytest.UsageError(f"[frameworks].default_mode must be one of {sorted(VALID_MODES)}, got {mode!r}")
    return with_config_requirements(replace(auto_spec(framework), mode=mode), framework_config)


def with_config_requirements(spec: FrameworkSpec, framework_config) -> FrameworkSpec:
    """Attach configured ancillary requirements when the CLI spec omits them."""
    if spec.requirements:
        return spec

    section = getattr(framework_config, "frameworks", None)
    if section is None:
        return spec

    if spec.framework != "pytorch":
        return spec

    requirements_file = getattr(section, "requirements_pytorch", "")
    if not requirements_file:
        return spec

    # Prefer repo-relative paths for local and remote runs. The provisioner stages
    # local files onto remote nodes before invoking pip.
    path = pathlib.Path(requirements_file)
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return replace(spec, requirements=(str(path),))


def ensure_framework_env(
    request: pytest.FixtureRequest, target_executor, framework_config, framework: str = "pytorch"
) -> FrameworkProvisionResult:
    """Return a pre-installed or lazily provisioned framework environment.

    Three phases are executed in order; each may short-circuit:

    **Phase 1 — sanity cache hit (fastest path, shared by both paths):**
        If ``config._framework_sanity_ok`` already has a result for this
        framework+node, return it immediately.  Both Path A and Path B
        populate this cache on first success, so all subsequent fixture calls
        land here regardless of how provisioning was originally triggered.

    **Phase 2 — pre-install result available (Path B):**
        If ``install_plugin.pytest_sessionstart`` stored a result for this node
        in ``config._framework_provision_results``, resolve it.  On success,
        promote to the sanity cache and return.  On failure, skip or fail the
        test immediately — the pre-install outcome is definitive, never retried.

    **Phase 3 — lazy install (Path A, retried per test until first success):**
        Provision the framework from channels/order in ``rocm-test.toml``.  On
        success, write the result to the sanity cache and return.  On failure,
        do **not** write to the cache — the next test that requests this fixture
        will retry Phase 3 (transient-failure recovery semantics).

    Args:
        request:          The pytest fixture request object.
        target_executor:  The executor group for the current test node.
        framework_config: Merged ``FrameworkConfig`` from the session.
        framework:        Framework name (default ``"pytorch"``).

    Returns:
        A :class:`~.provisioner.FrameworkProvisionResult` with ``ok=True``.
        Never returns a result with ``ok=False``; raises ``pytest.fail`` instead.
    """
    executor = _first_executor(target_executor)
    node_label = _node_label(executor)
    sanity_key = f"{framework}:{node_label}"

    # ------------------------------------------------------------------
    # Phase 1: sanity cache hit — O(1) dict lookup, no subprocess.
    # Populated by Phase 2 or Phase 3 on first success.
    # ------------------------------------------------------------------
    sanity_cache: dict = getattr(request.config, _SANITY_CACHE, {})
    if sanity_key in sanity_cache:
        return result_from_dict(sanity_cache[sanity_key])

    preinstall_requested = bool(getattr(request.config, "_framework_preinstall_requested", False))
    provision_results: dict = getattr(request.config, "_framework_provision_results", {})
    if preinstall_requested and not provision_results:
        provision_results = _load_shared_preinstall_results(request.config, framework_config)

    # ------------------------------------------------------------------
    # Phase 2: pre-install result available (Path B).
    # install_plugin keys by node_spec.label; apply a single-result
    # fallback for the common single-node case where the label may
    # differ slightly between the plugin and the fixture.
    # ------------------------------------------------------------------
    stored = provision_results.get(node_label)
    if stored is None and len(provision_results) == 1:
        stored = next(iter(provision_results.values()))

    if stored is not None:
        result = result_from_dict(stored)
        if result.ok:
            # Sanity passed during pre-install; promote to sanity cache so all
            # subsequent tests short-circuit at Phase 1.
            _write_sanity_cache(request.config, sanity_key, result)
            return result
        # Pre-install ran but the framework is not usable — never retry.
        pytest.fail(result.skip_reason())

    if preinstall_requested:
        # --pre-install was passed but produced no result for this node
        # (e.g. NodePool parallelism issue); fail rather than fall through
        # to a lazy install the operator did not request.
        pytest.fail(f"{framework} pre-install was requested but no environment is available on {node_label}")

    # ------------------------------------------------------------------
    # Phase 3: lazy install (Path A).
    # Channel URLs come from rocm-test.toml via channels_from_config();
    # the auto-mode priority is fixed in spec.AUTO_PIP_ORDER.
    # Called per test (function-scoped fixture) until the first success.
    # ------------------------------------------------------------------
    gpu_arch = request.config.getoption("--gpu-arch", default=None)
    rock_dir = _resolve_rock_dir(request.config, framework_config)
    rocm_version_hint = getattr(framework_config.therock, "rocm_version", "")
    result = provision_framework(
        runner=executor,
        node_label=node_label,
        artifact_dir=framework_config.framework.artifact_dir,
        spec=spec_from_config(framework_config, framework),
        channels=channels_from_config(framework_config),
        source="auto",
        gpu_arch=gpu_arch,
        rock_dir=rock_dir,
        rocm_version_hint=rocm_version_hint,
        log_name=f"{framework}-auto-{node_label}",
    )

    if result.ok:
        # Write to sanity cache ONLY on success — all future tests hit Phase 1.
        _write_sanity_cache(request.config, sanity_key, result)
        return result

    # Failure: do NOT write to sanity cache — next test retries Phase 3.
    pytest.fail(result.skip_reason())
    return result  # unreachable; pytest.fail raises


def ensure_pytorch_env(request: pytest.FixtureRequest, target_executor, framework_config) -> FrameworkProvisionResult:
    """Return a pre-installed or lazily provisioned PyTorch environment."""
    return ensure_framework_env(request, target_executor, framework_config, framework="pytorch")


def torch_python(pytorch_env: FrameworkProvisionResult) -> str:
    """Return the Python executable for a provisioned PyTorch environment."""
    return pytorch_env.python or "python3"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_sanity_cache(config, sanity_key: str, result: FrameworkProvisionResult) -> None:
    """Write *result* into the sanity cache on *config*.

    Initialises the dict attribute on first call.  Callers must only call this
    when ``result.ok`` is True.
    """
    if not hasattr(config, _SANITY_CACHE):
        config._framework_sanity_ok = {}
    config._framework_sanity_ok[sanity_key] = result_to_dict(result)


def _first_executor(target_executor):
    return next(iter(target_executor))


def _node_label(executor) -> str:
    test_logger = getattr(executor, "test_logger", None)
    if test_logger is not None:
        label: str = getattr(test_logger, "node_label", "")
        if label:
            return label
    host: str = getattr(executor, "host", "")
    if host:
        return host
    return "localhost"


def _resolve_rock_dir(config, framework_config) -> str:
    return (
        config.getoption("--rock-dir", default=None)
        or os.environ.get("ROCK_DIR")
        or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
        or framework_config.therock.rock_dir
        or ""
    )


def _load_shared_preinstall_results(config, framework_config) -> dict:
    """Load master-written framework provision results for xdist workers."""
    path = pathlib.Path(framework_config.framework.artifact_dir) / "pre-install" / _SHARED_RESULTS_NAME
    deadline = time.monotonic() + 30.0
    while True:
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}
            results = payload.get("framework_provision_results", {})
            if isinstance(results, dict):
                config._framework_provision_results = results
                config._framework_preinstall_requested = bool(payload.get("framework_preinstall_requested", True))
                return results
            return {}
        if time.monotonic() >= deadline:
            return {}
        time.sleep(1.0)
