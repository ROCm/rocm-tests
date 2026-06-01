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
pytest_plugins = [
    "framework.plugins.markers_plugin",      # FIRST: category-profile marker injection (CATEGORY_PROFILES in taxonomy.py)  # noqa: E501
    "framework.plugins.gpu_plugin",          # --no-gpu/--gpu-arch/--mock-gpu, gpu_arch/dry_run_executor
    "framework.plugins.remote_node_plugin",  # --remote-node/--gpu-acquire-timeout, node_pool/target_executor  # noqa: E501
    "framework.plugins.scheduling_plugin",   # --schedule-policy/--collect-runtimes, unified collection hook + runtime collector  # noqa: E501
    "framework.plugins.executor_plugin",     # --container-mode/--container-image/--container-runtime, cpu_executor/container_executor  # noqa: E501
    "framework.plugins.os_plugin",           # os_adapter/platform_name fixtures, os.* marker skip hook
    "framework.plugins.health_plugin",       # health_fixture (temp/ECC/VRAM gates)
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
def _attach_test_log(request, framework_config, caplog):
    """Autouse: attach the per-test log file to Allure after each test.

    ``framework.logging.test_logger.TestLogger`` writes full verbatim output
    (block headers, stdout, stderr) to a per-test log file and sets
    ``_BASE_LOGGER.propagate = False`` to avoid double-stamping through
    pytest's root handler.  Because propagation is off, ``caplog`` never
    receives those records, so reading caplog would produce an empty attachment
    for every GPU or CPU executor test.

    This fixture reads directly from the per-test log file, which always
    contains the complete output regardless of executor type.  Falls back to
    ``caplog`` for tests that emit records through standard Python logging
    without a ``TestLogger`` attached.

    The Allure attachment is a no-op when allure-pytest is not installed.

    Args:
        request:          pytest request object (provides test metadata).
        framework_config: Session-scoped merged config (avoids per-test reload).
        caplog:           pytest log capture fixture (fallback for non-executor tests).

    Yields:
        None: Runs setup before test, teardown after.
    """
    caplog.set_level(logging.DEBUG)
    yield

    # Prefer per-test log file written by TestLogger — complete, verbatim output.
    # Use _executor_log_file (pure, no truncation side-effect) so the teardown
    # reader does not wipe the file it is trying to read.
    log_content: str | None = None
    try:
        from framework.common.helpers import _executor_log_file

        p = _executor_log_file(framework_config.framework.artifact_dir, request.node.name, request.node.nodeid)
        try:
            st = p.stat()
            if st.st_size > 0:
                log_content = p.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            pass
    except Exception:  # pylint: disable=broad-except
        pass

    # Fallback: caplog records for tests that do not use a TestLogger.
    if not log_content and caplog.text:
        log_content = caplog.text

    if log_content:
        try:
            import allure

            allure.attach(
                log_content,
                name="test.log",
                attachment_type=allure.attachment_type.TEXT,
            )
        except ImportError:
            pass
