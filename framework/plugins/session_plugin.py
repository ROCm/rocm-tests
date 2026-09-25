# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
session_plugin.py -- Foundational session fixtures for rocm-test.

Provides the three fixtures that every other framework plugin depends on:

    framework_config  (session-scoped)
        Load and merge the framework configuration from rocm-test.toml,
        ROCM_TEST_* environment variables, and pytest CLI flags.  All plugins
        that need config values (artifact dirs, GPU thresholds, build paths, …)
        receive it through this fixture.

    run_ctx  (session-scoped)
        A unique RunContext (run_id + start_time) created once per pytest
        session.  Used to correlate artifacts, results, and notifications
        across tests within the same invocation.

    _attach_test_log  (function-scoped, autouse)
        Attach the per-test executor log (or caplog fallback) to Allure after
        every test.  This fixture is invisible to test authors; it runs
        automatically for every test function regardless of markers.

Why a dedicated plugin?
    Previously these three fixtures lived in the root ``conftest.py``.  pytest
    exposes conftest-defined fixtures only to tests *at or below* that
    conftest's directory.  Any consumer repo that places test directories
    outside the public ``rocm_public/`` subtree (e.g. an internal enterprise
    repo with a sibling ``tests_enterprise/`` tree) would never see
    ``framework_config`` or ``run_ctx``, causing hard fixture-not-found errors
    the moment any GPU fixture (``target_executor``, ``health_fixture``,
    ``compile_binary``, …) is requested.

    Moving them into this plugin — registered in ``pytest_plugins`` —
    makes them globally visible to every test regardless of directory layout,
    exactly like any other plugin-provided fixture.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture(scope="session")
def framework_config(request):
    """Return the merged framework configuration (file → env vars → CLI flags).

    Priority cascade (lowest → highest):
        1. Code defaults in FrameworkConfig
        2. rocm-test.toml (CWD or path from --rocm-config)
        3. ROCM_TEST_* environment variables
        4. pytest CLI flags (--rocm-config, etc.)

    ``gpu_plugin.pytest_configure`` loads and caches this config on
    ``config._framework_config`` before the session starts so that
    ``GpuDetector`` and this fixture share the exact same object.  We reuse
    that cached instance to guarantee a single ``load_config()`` call per
    session.  If the cache is absent (--mock-gpu or future unit-test contexts
    that skip gpu_plugin), we fall back to loading fresh.

    Returns:
        FrameworkConfig: Validated, fully-merged config dataclass.
    """
    cached = getattr(request.config, "_framework_config", None)
    if cached is not None:
        return cached

    from framework.config.loader import load_config  # pylint: disable=import-outside-toplevel

    # --rocm-config is registered by gpu_plugin; guard so session_plugin has no
    # hard dependency on gpu_plugin being present (e.g. unit-test contexts).
    try:
        config_path = request.config.getoption("--rocm-config", default=None)
    except ValueError:
        config_path = None
    return load_config(config_path=config_path)


@pytest.fixture(scope="session")
def run_ctx(framework_config):  # pylint: disable=redefined-outer-name
    """Create a unique run context (run_id, start timestamp) for this session.

    The run context is passed down to fixtures that need to correlate artifacts,
    results, and notifications across tests within the same pytest invocation.

    Returns:
        RunContext: Immutable dataclass with run_id and start_time.
    """
    return framework_config.new_run_context()


@pytest.fixture(autouse=True)
def _attach_test_log(request, framework_config, caplog):  # pylint: disable=redefined-outer-name
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
    # Use executor_log_file (pure, no truncation side-effect) so the teardown
    # reader does not wipe the file it is trying to read.
    log_content: str | None = None
    try:
        from framework.common.helpers import executor_log_file

        p = executor_log_file(framework_config.framework.artifact_dir, request.node.name, request.node.nodeid)
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
