# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
loader.py -- Framework configuration with ENV → file → defaults cascade.

Priority (lowest → highest):
    1. Code defaults in FrameworkConfig / sub-dataclasses
    2. rocm-test.toml found in CWD, then $HOME
    3. ROCM_TEST_<SECTION>_<KEY> environment variables
       e.g. ROCM_TEST_GPU_MAX_TEMP_CELSIUS=85
            ROCM_TEST_FRAMEWORK_LOG_LEVEL=verbose
    4. pytest CLI flag --rocm-config <path> (overrides file search)

Secrets (webhook URLs, API tokens) come from environment variables ONLY.
They are never read from or written to the config file.

Usage:
    from framework.config.loader import load_config

    config = load_config()                       # auto-find rocm-test.toml
    config = load_config("/ci/rocm-test.toml")  # explicit path

    # In pytest, use the framework_config session fixture instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import datetime
import logging
import os
import pathlib
import uuid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclasses — each section of rocm-test.toml maps to one dataclass
# ---------------------------------------------------------------------------


@dataclass
class FrameworkSection:
    """Top-level [framework] config section."""

    log_level: str = "normal"
    run_id_prefix: str = "rocm-test"
    artifact_dir: str = "output/artifacts/"
    session_log: str = "output/artifacts/session.log"


@dataclass
class GpuSection:
    """[gpu] config section."""

    detection: str = "auto"
    max_temp_celsius: int = 90
    max_ecc_errors: int = 0
    min_vram_free_mb: int = 512
    # --gpu-health-metrics: point-in-time snapshots before/after each test
    health_metrics: list[str] = field(default_factory=lambda: ["temp", "vram", "util", "ecc", "clock"])
    # --monitor-gpu: continuous background poller during test execution
    monitor_metrics: list[str] = field(default_factory=lambda: ["temp", "vram", "util", "ecc", "clock"])
    monitor_interval_secs: float = 15.0
    monitor_duration_secs: float = 0.0


@dataclass
class ResultsSection:
    """[results] config section."""

    upload_mode: str = "auto"
    local_dir: str = "output/results/"
    sqlite_db: str = "output/rocm_test.db"


@dataclass
class BaselinesSection:
    """[baselines] config section."""

    regression_pct: float = 5.0
    baseline_dir: str = "tests/performance/baselines/"


@dataclass
class ReportingSection:
    """[reporting] config section."""

    allure_results_dir: str = "output/artifacts/allure-results/"
    history_depth: int = 5


@dataclass
class NotificationsSection:
    """[notifications] config section."""

    webhook_url: str = ""
    notify_on: list[str] = field(default_factory=lambda: ["FAIL", "REGRESSION", "HEALTH_FAIL"])


@dataclass
class TheRockSection:
    """[therock] config section — path to a TheRock/ROCm installation.

    rock_dir: Path to the TheRock/ROCm install tree (contains bin/hipcc, lib/).
              Also settable via CLI --rock-dir or env ROCK_DIR /
              ROCM_TEST_THEROCK_ROCK_DIR.
    build_dir: Output directory for compiled test binaries.
    build_timeout_secs: Wall-clock timeout for a single hipcc compilation (default 2 h).
                        Also settable via ROCM_TEST_THEROCK_BUILD_TIMEOUT_SECS.
    build_inactivity_timeout_secs: Kill the compiler after this many seconds of no
                        output — catches OOM-stalled linkers (default 10 min).
                        Also settable via ROCM_TEST_THEROCK_BUILD_INACTIVITY_TIMEOUT_SECS.
    """

    rock_dir: str = ""
    build_dir: str = "output/test-binaries/"
    build_timeout_secs: float = 7200.0
    build_inactivity_timeout_secs: float = 600.0


@dataclass
class RunContext:
    """Immutable per-session execution context created by FrameworkConfig.new_run_context()."""

    run_id: str
    start_time: datetime.datetime

    def __str__(self) -> str:
        return f"{self.run_id} started at {self.start_time.isoformat()}"


@dataclass
class FrameworkConfig:
    """Fully-merged framework configuration.

    All fields are populated by load_config() before being returned.
    Tests should access this via the ``framework_config`` session fixture.
    """

    framework: FrameworkSection = field(default_factory=FrameworkSection)
    gpu: GpuSection = field(default_factory=GpuSection)
    results: ResultsSection = field(default_factory=ResultsSection)
    baselines: BaselinesSection = field(default_factory=BaselinesSection)
    reporting: ReportingSection = field(default_factory=ReportingSection)
    notifications: NotificationsSection = field(default_factory=NotificationsSection)
    therock: TheRockSection = field(default_factory=TheRockSection)

    def new_run_context(self) -> RunContext:
        """Create a unique run context for the current test session.

        Returns:
            RunContext: run_id built from the configured prefix + short UUID,
                        and the UTC start timestamp.
        """
        short_id = str(uuid.uuid4())[:8]
        run_id = f"{self.framework.run_id_prefix}-{short_id}"
        return RunContext(run_id=run_id, start_time=datetime.datetime.now(datetime.timezone.utc))


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(config_path: str | None = None) -> FrameworkConfig:
    """Load and merge framework configuration from file + environment.

    Args:
        config_path: Explicit path to a rocm-test.toml file.
                     If None, searches CWD then $HOME for ``rocm-test.toml``.

    Returns:
        FrameworkConfig: Merged configuration with env var overrides applied.
    """
    raw: dict = {}

    # Step 1 — read TOML file (if found)
    toml_path = _resolve_config_path(config_path)
    if toml_path:
        raw = _read_toml(toml_path)
        logger.info("Loaded config from %s", toml_path)
    else:
        logger.info("No rocm-test.toml found; using code defaults")

    # Step 2 — build dataclass from raw dict (code defaults fill missing keys)
    cfg = _build_config(raw)

    # Step 3 — apply env var overrides (ROCM_TEST_<SECTION>_<KEY>)
    _apply_env_overrides(cfg)

    return cfg


# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------


def _resolve_config_path(explicit: str | None) -> pathlib.Path | None:
    """Return a resolved Path to rocm-test.toml, or None if not found."""
    if explicit:
        p = pathlib.Path(explicit)
        if p.is_file():
            return p
        logger.warning("--rocm-config path not found: %s", explicit)
        return None

    for candidate in [
        pathlib.Path.cwd() / "rocm-test.toml",
        pathlib.Path.home() / "rocm-test.toml",
    ]:
        if candidate.is_file():
            return candidate
    return None


def _read_toml(path: pathlib.Path) -> dict:
    """Read a TOML file and return its contents as a nested dict."""
    try:
        import tomllib  # Python 3.11+  # pylint: disable=import-outside-toplevel
    except ImportError:
        import tomli as tomllib  # pylint: disable=import-outside-toplevel

    with path.open("rb") as fh:
        return tomllib.load(fh)  # type: ignore[no-any-return]


def _build_config(raw: dict) -> FrameworkConfig:
    """Construct FrameworkConfig from a raw TOML dict, filling defaults for missing keys."""

    def _merge(default_cls, section_key: str):
        section_raw = raw.get(section_key, {})
        obj = default_cls()
        for key, value in section_raw.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj

    return FrameworkConfig(
        framework=_merge(FrameworkSection, "framework"),
        gpu=_merge(GpuSection, "gpu"),
        results=_merge(ResultsSection, "results"),
        baselines=_merge(BaselinesSection, "baselines"),
        reporting=_merge(ReportingSection, "reporting"),
        notifications=_merge(NotificationsSection, "notifications"),
        therock=_merge(TheRockSection, "therock"),
    )


def _apply_env_overrides(cfg: FrameworkConfig) -> None:
    """Apply ROCM_TEST_<SECTION>_<KEY> env vars to the config in-place.

    Mapping examples:
        ROCM_TEST_GPU_MAX_TEMP_CELSIUS=85  → cfg.gpu.max_temp_celsius = 85
        ROCM_TEST_FRAMEWORK_LOG_LEVEL=verbose → cfg.framework.log_level = "verbose"
        ROCM_TEST_NOTIFICATIONS_WEBHOOK_URL=https://... → cfg.notifications.webhook_url

    Type conversion is inferred from the existing default type.
    """
    section_map = {
        "FRAMEWORK": cfg.framework,
        "GPU": cfg.gpu,
        "RESULTS": cfg.results,
        "BASELINES": cfg.baselines,
        "REPORTING": cfg.reporting,
        "NOTIFICATIONS": cfg.notifications,
        "THEROCK": cfg.therock,
    }

    for section_key, section_obj in section_map.items():
        for attr in vars(section_obj):
            env_key = f"ROCM_TEST_{section_key}_{attr.upper()}"
            raw_value = os.environ.get(env_key)
            if raw_value is None:
                continue

            current = getattr(section_obj, attr)
            try:
                converted: object
                if isinstance(current, bool):
                    converted = raw_value.lower() in ("1", "true", "yes")
                elif isinstance(current, int):
                    converted = int(raw_value)
                elif isinstance(current, float):
                    converted = float(raw_value)
                elif isinstance(current, list):
                    converted = [v.strip() for v in raw_value.split(",")]
                else:
                    converted = raw_value

                setattr(section_obj, attr, converted)
                logger.debug("ENV override: %s=%r", env_key, converted)
            except (ValueError, TypeError) as exc:
                logger.warning("Could not apply env override %s=%r: %s", env_key, raw_value, exc)
