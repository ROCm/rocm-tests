# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_config_loader.py -- Unit tests for the framework configuration loader.

Validates:
    1. Code defaults are returned when no config file and no env vars are set.
    2. ENV vars (ROCM_TEST_*) override file-based defaults.
    3. new_run_context() returns a unique RunContext per call.

Markers: ci.pr, layer.runtime, hw.cpu_only, runtime.fast, os.linux
"""

from __future__ import annotations

import pytest

from framework.config.loader import FrameworkConfig, load_config


@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
@pytest.mark.os.linux
@pytest.mark.parametrize("_unused", [None])  # ensures pytest collects this as a test
class TestConfigLoaderDefaults:
    """Group: default config values when no file and no env vars are set."""

    def test_returns_framework_config_instance(self, _unused, monkeypatch, tmp_path):
        """load_config() should return a FrameworkConfig instance."""
        monkeypatch.chdir(tmp_path)  # CWD has no rocm-test.toml
        cfg = load_config()
        assert isinstance(cfg, FrameworkConfig)

    def test_default_log_level(self, _unused, monkeypatch, tmp_path):
        """Default log_level should be 'normal'."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert cfg.framework.log_level == "normal"

    def test_default_allure_results_dir(self, _unused, monkeypatch, tmp_path):
        """Default allure_results_dir should be under output/artifacts/."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        assert "allure" in cfg.reporting.allure_results_dir


@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
@pytest.mark.os.linux
@pytest.mark.parametrize(
    ("env_key", "env_val", "attr_path"),
    [
        ("ROCM_TEST_FRAMEWORK_LOG_LEVEL", "verbose", ("framework", "log_level")),
        ("ROCM_TEST_GPU_MAX_TEMP_CELSIUS", "80", ("gpu", "max_temp_celsius")),
        ("ROCM_TEST_REPORTING_HISTORY_DEPTH", "10", ("reporting", "history_depth")),
    ],
)
class TestEnvVarOverrides:
    """Group: ENV var overrides take priority over file-based config."""

    def test_env_override(self, env_key, env_val, attr_path, monkeypatch, tmp_path):
        """ROCM_TEST_* env vars should override the default config value."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(env_key, env_val)
        cfg = load_config()
        section, field = attr_path
        actual = getattr(getattr(cfg, section), field)
        # Compare as string for simplicity (all our test values are simple types)
        assert str(actual) == env_val or actual == type(actual)(env_val)


@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
@pytest.mark.os.linux
class TestRunContext:
    """Group: new_run_context() generates unique, timestamped run IDs."""

    def test_run_context_has_run_id(self, monkeypatch, tmp_path):
        """RunContext.run_id should include the configured prefix."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        ctx = cfg.new_run_context()
        assert cfg.framework.run_id_prefix in ctx.run_id

    def test_run_contexts_are_unique(self, monkeypatch, tmp_path):
        """Two consecutive new_run_context() calls should yield different run_ids."""
        monkeypatch.chdir(tmp_path)
        cfg = load_config()
        ctx1 = cfg.new_run_context()
        ctx2 = cfg.new_run_context()
        assert ctx1.run_id != ctx2.run_id


# Standalone marker form (for compatibility with marker linter)
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_config_loader_smoke(monkeypatch, tmp_path):
    """Smoke: load_config() returns a valid FrameworkConfig in a clean directory."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert isinstance(cfg, FrameworkConfig)
    assert cfg.framework.log_level in ("quiet", "normal", "verbose")
    assert cfg.gpu.max_temp_celsius > 0
