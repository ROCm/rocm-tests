# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Root conftest: plugin loading and shared session fixtures.

All pytest plugin modules from ``framework/plugins/`` are declared here via
``pytest_plugins``. This allows ``git clone && pip install -r requirements.txt &&
pytest`` to work without ``pip install -e .`` because ``pythonpath = ["."]`` in
``pyproject.toml`` adds the repo root to ``sys.path`` at pytest startup.

Plugin responsibilities (registration order — markers_plugin MUST be first):
    markers_plugin      -- Category-profile marker injection (CATEGORY_PROFILES in taxonomy.py)
    gpu_plugin          -- GPU acquisition, --no-gpu/--gpu-arch/--mock-gpu options
    remote_node_plugin  -- --remote-node/--gpu-acquire-timeout, NodePool, target_executor
    scheduling_plugin   -- --schedule-policy/--collect-runtimes, unified collection hook + runtime collector
    executor_plugin     -- --container-mode/--container-image, cpu_executor/container_executor
    os_plugin           -- os_adapter/platform_name fixtures, os.* marker skip hook
    health_plugin       -- Pre/post GPU health gates (temp, ECC, VRAM, clocks)
    baseline_plugin     -- Regression comparison against baselines/*.yaml
    artifacts_plugin    -- Allure attachment of GPU state dumps on failure
    prereqs_plugin      -- Session-level prerequisite checks (driver, ROCm version)
    retry_plugin        -- --retry-count option, retry_fixture
    reports_plugin      -- Allure label mapping, outcome_fixture
    builder_plugin      -- --rock-dir/--compiler-build-dir, rock_dir/compile_binary/ld_path
    install_plugin      -- --pre-install rocm=X/pkg=X, parallel pre-session node install

Fixtures defined here (``framework_config``, ``run_ctx``, ``_attach_test_log``)
are available to every test in the suite without any import.
"""
from __future__ import annotations

import logging

import pytest
from _pytest.mark.structures import MarkDecorator


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
pytest_plugins = [
    "framework.plugins.markers_plugin",      # FIRST: category-profile marker injection (CATEGORY_PROFILES in taxonomy.py)
    "framework.plugins.gpu_plugin",          # --no-gpu/--gpu-arch/--mock-gpu, gpu_fixture/dry_run_executor
    "framework.plugins.remote_node_plugin",  # --remote-node/--gpu-acquire-timeout, node_pool/target_executor/multi_gpu_fixture/multi_node_fixture
    "framework.plugins.scheduling_plugin",   # --schedule-policy/--collect-runtimes, unified collection hook + runtime collector
    "framework.plugins.executor_plugin",     # --container-mode/--container-image, cpu_executor/session_executor/remote_pool/container_executor
    "framework.plugins.os_plugin",           # os_adapter/platform_name fixtures, os.* marker skip hook
    "framework.plugins.health_plugin",       # health_fixture (temp/ECC/VRAM gates)
    "framework.plugins.baseline_plugin",     # baseline_fixture (YAML regression compare)
    "framework.plugins.artifacts_plugin",    # artifacts_fixture, allure_reporter fixture
    "framework.plugins.prereqs_plugin",      # prereqs_fixture (session prereq checks)
    "framework.plugins.retry_plugin",        # retry_fixture, --retry-count option
    "framework.plugins.reports_plugin",      # Allure label mapping, outcome_fixture
    "framework.plugins.builder_plugin",      # --rock-dir/--compiler-build-dir, rock_dir/compile_binary/ld_path fixtures
    "framework.plugins.install_plugin",      # --pre-install rocm=X / pkg=X, parallel pre-session node install
]

logger = logging.getLogger("rocm_test")


# ---------------------------------------------------------------------------
# Session-scoped fixtures — created once per pytest run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def framework_config(request):
    """Load the merged framework configuration (file → env vars → CLI flags).

    Priority cascade (lowest → highest):
        1. Code defaults in FrameworkConfig
        2. rocm-test.toml (CWD or path from --rocm-config)
        3. ROCM_TEST_* environment variables
        4. pytest CLI flags (--rocm-config, etc.)

    Returns:
        FrameworkConfig: Validated, fully-merged config dataclass.
    """
    from framework.config.loader import load_config

    config_path = request.config.getoption("--rocm-config", default=None)
    return load_config(config_path=config_path)


@pytest.fixture(scope="session")
def run_ctx(framework_config):
    """Create a unique run context (run_id, start timestamp) for this session.

    The run context is passed down to fixtures that need to correlate artifacts,
    results, and notifications across tests within the same pytest invocation.

    Returns:
        RunContext: Immutable dataclass with run_id and start_time.
    """
    return framework_config.new_run_context()


# ---------------------------------------------------------------------------
# Function-scoped fixtures — applied to every individual test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _attach_test_log(caplog, request):
    """Autouse: capture Python log output and attach it to Allure per test.

    Runs for every test automatically. The Allure attachment is a no-op when
    allure-pytest is not installed (e.g. lightweight CI environments), so this
    fixture degrades gracefully.

    Args:
        caplog: pytest built-in log capture fixture.
        request: pytest request object (provides test metadata).

    Yields:
        None: Runs setup before test, teardown after.
    """
    caplog.set_level(logging.DEBUG)
    yield
    if caplog.text:
        try:
            import allure

            allure.attach(
                caplog.text,
                name="test.log",
                attachment_type=allure.attachment_type.TEXT,
            )
        except ImportError:
            pass
