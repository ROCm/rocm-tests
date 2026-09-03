# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Root conftest: plugin loading and dotted-marker patch.

All pytest plugin modules from ``framework/plugins/`` are declared here via
``pytest_plugins``. This allows ``git clone && pip install -r requirements.txt &&
pytest`` to work without ``pip install -e .`` because ``pythonpath = ["."]`` in
``pyproject.toml`` adds the repo root to ``sys.path`` at pytest startup.

Plugin responsibilities (registration order — markers_plugin MUST be first):
    markers_plugin      -- Category-profile marker injection (CATEGORY_PROFILES in taxonomy.py)
    session_plugin      -- framework_config, run_ctx, _attach_test_log (foundational session fixtures)
    gpu_plugin          -- GPU acquisition, --no-gpu/--gpu-arch/--mock-gpu options
    remote_node_plugin  -- --remote-node/--gpu-acquire-timeout, NodePool, target_executor
    scheduling_plugin   -- --schedule-policy/--collect-runtimes, unified collection hook + runtime collector
    executor_plugin     -- --container-mode/--container-image, cpu_executor/container_executor
    os_plugin           -- os_adapter/platform_name fixtures, os.* marker skip hook
    health_plugin       -- Pre/post GPU health gates (temp, ECC, VRAM, clocks)
    artifacts_plugin    -- Allure attachment of GPU state dumps on failure
    prereqs_plugin      -- Session-level prerequisite checks (driver, ROCm version)
    retry_plugin        -- --retry-count option, retry_fixture
    reports_plugin      -- Allure label mapping, outcome_fixture
    builder_plugin      -- --rock-dir/--compiler-build-dir, rock_dir/compile_binary/ld_path
    install_plugin      -- --pre-install rocm=X/pkg=X, parallel pre-session node install

``framework_config``, ``run_ctx``, and ``_attach_test_log`` live in
``session_plugin`` so they are globally visible to any consumer regardless of
directory layout (plugin fixtures are not bound to a conftest subtree).
"""

from __future__ import annotations

from _pytest.mark.structures import MarkDecorator
import pytest


def _mark_getattr(self: MarkDecorator, name: str) -> MarkDecorator:
    """Enable @pytest.mark.dim.val dotted syntax (e.g. @pytest.mark.ci.pr).

    pytest 7+ removed MarkDecorator.__getattr__, so `pytest.mark.ci` returns a
    MarkDecorator and `.pr` raises AttributeError. This restores the behaviour
    by delegating to pytest.mark with the fully-qualified dotted name.
    """
    return getattr(pytest.mark, f"{self.mark.name}.{name}")


MarkDecorator.__getattr__ = _mark_getattr  # type: ignore[assignment]

# Declare all plugin modules — loaded by pytest before test collection begins.
# Each module is a standard Python dotted path resolvable via PYTHONPATH=".".
#
# ORDERING CONSTRAINT: markers_plugin MUST be first.
# pytest calls pytest_collection_modifyitems hooks in plugin-registration order.
# markers_plugin injects hw.*/ci.*/layer.* markers from CATEGORY_PROFILES; any
# plugin that reads those markers (scheduling_plugin sorts by hw.*, gpu_plugin
# skips by hw.gpu) must be registered AFTER markers_plugin so that tests relying
# on category profiles are fully annotated before they are sorted or skipped.
# Do not move markers_plugin below scheduling_plugin or gpu_plugin.
#
# session_plugin is second: framework_config and run_ctx must be available
# before remote_node_plugin, executor_plugin, health_plugin, builder_plugin,
# and install_plugin resolve their own fixtures.
pytest_plugins = [
    "framework.plugins.markers_plugin",  # FIRST: category-profile marker injection (CATEGORY_PROFILES in taxonomy.py)  # noqa: E501
    "framework.plugins.session_plugin",  # framework_config, run_ctx, _attach_test_log
    "framework.plugins.gpu_plugin",  # --no-gpu/--gpu-arch/--mock-gpu, gpu_arch/dry_run_executor
    "framework.plugins.remote_node_plugin",  # --remote-node/--gpu-acquire-timeout, node_pool/target_executor  # noqa: E501
    "framework.plugins.scheduling_plugin",  # --schedule-policy/--collect-runtimes, unified collection hook + runtime collector  # noqa: E501
    "framework.plugins.executor_plugin",  # --container-mode/--container-image/--container-runtime, cpu_executor/container_executor  # noqa: E501
    "framework.plugins.os_plugin",  # os_adapter/platform_name fixtures, os.* marker skip hook
    "framework.plugins.health_plugin",  # health_fixture (temp/ECC/VRAM gates)
    "framework.plugins.artifacts_plugin",  # artifacts_fixture, allure_reporter fixture
    "framework.plugins.prereqs_plugin",  # prereqs_fixture (session prereq checks)
    "framework.plugins.retry_plugin",  # retry_fixture, --retry-count option
    "framework.plugins.reports_plugin",  # Allure label mapping, outcome_fixture
    "framework.plugins.builder_plugin",  # --rock-dir/--compiler-build-dir, rock_dir/compile_binary/ld_path fixtures
    "framework.plugins.install_plugin",  # --pre-install rocm=X / pkg=X, parallel pre-session node install
]
