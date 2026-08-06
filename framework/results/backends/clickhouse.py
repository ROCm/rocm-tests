#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Insert ROCm nightly workflow results into ClickHouse.

Writes a single converged table ``rocmtests_nightly_results`` that holds two row
types, distinguished by the ``row_type`` column:

``row_type = 'summary'``
    One row per platform per nightly run.  Contains per-platform aggregate
    counts (pass/fail/skip/error), GitHub Actions job metadata, artifact URLs,
    and run-level context.  A synthetic ``TOTAL`` summary row is also written
    (``platform = 'TOTAL'``).

``row_type = 'case'``
    One row per individual test function per platform per nightly run.  Contains
    the pytest node id, test file, outcome, duration, failure message, and the
    marker dimensions captured from ``pytest-json-report`` (hw.*, ci.*, layer.*,
    runtime.*).

Both row types share the primary sort key ``(run_id, platform, row_type,
nodeid)`` so the table can serve as a single source of truth for dashboards
without any cross-table JOINs.  A supplementary ``SummingMergeTree``
Materialized View (``rocmtests_job_rollup`` / ``rocmtests_job_rollup_mv``)
pre-aggregates per-file outcome counts for sub-second dashboard rollup queries.

All connection parameters must be supplied via environment variables or CLI
flags — no values are hardcoded.

Required environment variables::

    CLICKHOUSE_HOST      ClickHouse server hostname or IP
    CLICKHOUSE_PORT      HTTPS port (default 8443)
    CLICKHOUSE_USERNAME  ClickHouse user with INSERT privileges
    CLICKHOUSE_PASSWORD  Corresponding password
    CLICKHOUSE_DATABASE  Target database name
    CLICKHOUSE_SECURE    Set to "true" for TLS (default true)

Optional environment variables::

    CLICKHOUSE_CA_CERT              Path to a custom CA bundle (PEM)
    CLICKHOUSE_VERIFY_SSL           Set to "false" to skip TLS verification (test envs only)
    CLICKHOUSE_CONNECT_TIMEOUT      TCP connect timeout in seconds (default 10)
    CLICKHOUSE_SEND_RECEIVE_TIMEOUT Query timeout in seconds (default 30)
    GITHUB_TOKEN                    GitHub PAT for the Jobs/Runs API
    GITHUB_VERIFY_SSL               Set to "false" to skip GitHub API TLS verification

Examples:
    Typical nightly CI invocation::

        python -m framework.results.backends.clickhouse \\
            --run-id 12345678 \\
            --github-repo ROCm/rocm-tests \\
            --source-repo ROCm/rockrel \\
            --artifact-source multi_arch_release.yml \\
            --table rocmtests_nightly_results \\
            --testplan testplan.ini \\
            --counts-dir all-counts \\
            --reports-dir all-reports

    Create tables on first use::

        python -m framework.results.backends.clickhouse \\
            ... \\
            --create-table

    Drop and recreate (schema migration)::

        python -m framework.results.backends.clickhouse \\
            ... \\
            --recreate-table
"""

from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import ssl
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import clickhouse_connect

# ── Constants ────────────────────────────────────────────────────────────────

CLICKHOUSE_TABLE_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?")

# Maximum characters stored in error_message (longrepr can be very long)
_ERROR_MSG_LIMIT = 500

# Regex to extract the first marker value for each dimension from pytest keywords
_MARKER_RE = re.compile(r"^(hw|ci|layer|runtime)\.(\w+)$")

# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetConfig:
    """Target/platform data from ``testplan.ini``."""

    name: str
    runs_on: str
    artifact_group: str
    gpu_arch: str
    tests_filters: str
    target_available: bool


@dataclass(frozen=True)
class PlatformCounts:
    """Per-platform pytest summary parsed from ``counts.json``."""

    platform: str
    passed: int
    skipped: int
    failed: int
    error: int
    total: int


@dataclass(frozen=True)
class PlatformJobMetadata:
    """Per-platform GitHub Actions job metadata."""

    url: str
    status: str
    conclusion: str


@dataclass(frozen=True)
class TestCaseResult:
    """Per-test result parsed from a ``pytest-json-report`` JSON file.

    Attributes:
        nodeid: Full pytest node id, e.g.
            ``tests/e2e/hip_runtime/test_foo.py::TestClass::test_bar[param]``.
        test_file: Path component of the node id, e.g.
            ``tests/e2e/hip_runtime/test_foo.py``.
        test_class: Class name extracted from the node id, or empty string.
        test_name: Function + parametrize component, e.g. ``test_bar[param]``.
        outcome: One of ``passed``, ``failed``, ``error``, ``skipped``.
        duration_secs: Total wall-clock time for the test (setup + call + teardown).
        phase: Which lifecycle phase produced the failure: ``call``, ``setup``,
            ``teardown``, or empty string for passing/skipped tests.
        error_type: Exception class name, e.g. ``AssertionError``, or empty.
        error_message: First ``_ERROR_MSG_LIMIT`` chars of the failure ``longrepr``.
        hw_marker: Value of the ``hw.*`` marker, e.g. ``gpu``, ``multi_gpu``.
        ci_marker: Value of the ``ci.*`` marker, e.g. ``nightly``, ``pr``.
        layer_marker: Value of the ``layer.*`` marker, e.g. ``runtime``.
        runtime_marker: Value of the ``runtime.*`` marker, e.g. ``fast``.
        markers_raw: JSON array string of all keyword strings on the test.
    """

    nodeid: str
    test_file: str
    test_class: str
    test_name: str
    outcome: str
    duration_secs: float
    phase: str
    error_type: str
    error_message: str
    hw_marker: str
    ci_marker: str
    layer_marker: str
    runtime_marker: str
    markers_raw: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Convert a GitHub ISO timestamp to a UTC-aware datetime."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def int_or_zero(value: Any) -> int:
    """Coerce a value to int, returning 0 on failure."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def str_or_empty(value: Any) -> str:
    """Coerce nullable values to ClickHouse-safe empty strings."""
    if value is None:
        return ""
    return str(value)


def validate_clickhouse_table_name(table: str) -> str:
    """Validate an unquoted ClickHouse table identifier (optionally database-qualified)."""
    if not CLICKHOUSE_TABLE_PATTERN.fullmatch(table):
        raise ValueError(
            "ClickHouse table name must be an unquoted identifier or database.table pair "
            "using only letters, numbers, and underscores"
        )
    return table


def default_ca_cert() -> str | None:
    """Return certifi's CA bundle path when certifi is installed."""
    try:
        import certifi
    except ImportError:
        return None
    return certifi.where()


def _rollup_table_name(table: str) -> str:
    """Derive the rollup table name from the main table name."""
    base = table.split(".")[-1]
    prefix = table[: -len(base)] if "." in table else ""
    return f"{prefix}{base}_rollup"


def _rollup_mv_name(table: str) -> str:
    """Derive the rollup MV name from the main table name."""
    return f"{_rollup_table_name(table)}_mv"


# ── ClickHouse connection ────────────────────────────────────────────────────


def build_client():
    """Create a ClickHouse client from environment variables.

    All connection parameters must be provided via environment variables; no
    defaults are assumed for host, username, or database so that credentials
    are never embedded in source code.
    """

    def require_env(name: str) -> str:
        value = os.environ.get(name, "").strip()
        if not value:
            raise RuntimeError(f"{name} must be set in the environment")
        return value

    verify_ssl = env_bool("CLICKHOUSE_VERIFY_SSL", True)
    ca_cert = os.environ.get("CLICKHOUSE_CA_CERT")
    if verify_ssl and ca_cert is None:
        ca_cert = default_ca_cert()

    kwargs: dict[str, Any] = {
        "host": require_env("CLICKHOUSE_HOST"),
        "port": int(os.environ.get("CLICKHOUSE_PORT", "8443")),
        "username": require_env("CLICKHOUSE_USERNAME"),
        "password": require_env("CLICKHOUSE_PASSWORD"),
        "database": require_env("CLICKHOUSE_DATABASE"),
        "secure": env_bool("CLICKHOUSE_SECURE", True),
        "verify": verify_ssl,
        "connect_timeout": int(os.environ.get("CLICKHOUSE_CONNECT_TIMEOUT", "10")),
        "send_receive_timeout": int(os.environ.get("CLICKHOUSE_SEND_RECEIVE_TIMEOUT", "30")),
    }
    if ca_cert:
        kwargs["ca_cert"] = ca_cert
    return clickhouse_connect.get_client(**kwargs)


# ── GitHub API helpers ───────────────────────────────────────────────────────


def github_headers() -> dict[str, str]:
    """Build GitHub API request headers."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "rocmtests-nightly-clickhouse-ingest",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def ssl_context() -> ssl.SSLContext | None:
    """Return an SSL context using certifi, or a permissive context when GITHUB_VERIFY_SSL=false."""
    if not env_bool("GITHUB_VERIFY_SSL", True):
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    ca_cert = default_ca_cert()
    if ca_cert is None:
        return None
    return ssl.create_default_context(cafile=ca_cert)


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch a JSON object from the GitHub API."""
    request = Request(url, headers=github_headers())
    with urlopen(
        request, timeout=30, context=ssl_context()
    ) as response:  # nosec B310 — URL always https://api.github.com/; never from user input
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return payload


def paged_github_items(url: str, key: str) -> list[dict[str, Any]]:
    """Collect all pages of a GitHub API list response."""
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        separator = "&" if "?" in url else "?"
        data = fetch_json(f"{url}{separator}per_page=100&page={page}")
        batch = data.get(key, [])
        items.extend(batch)
        if len(batch) < 100:
            return items
        page += 1


# ── testplan.ini helpers ─────────────────────────────────────────────────────


def load_testplan(testplan_path: Path) -> dict[str, TargetConfig]:
    """Read target/platform config from ``testplan.ini``."""
    if not testplan_path.is_file():
        print(f"WARNING: testplan not found at {testplan_path}; gpu_arch mapping will be partial", flush=True)
        return {}
    parser = configparser.ConfigParser()
    parser.read(testplan_path)
    targets: dict[str, TargetConfig] = {}
    for section in parser.sections():
        target_available = parser.get(section, "target_available", fallback="true").strip().lower() != "false"
        targets[section] = TargetConfig(
            name=section,
            runs_on=parser.get(section, "runs_on", fallback=""),
            artifact_group=parser.get(section, "artifact_group", fallback=""),
            gpu_arch=parser.get(section, "gpu_arch", fallback=""),
            tests_filters=parser.get(section, "tests_filters", fallback=""),
            target_available=target_available,
        )
    return targets


def resolve_target(platform: str, targets: dict[str, TargetConfig]) -> TargetConfig | None:
    """Find the closest testplan target for a workflow platform name."""
    if platform in targets:
        return targets[platform]
    matches = [t for name, t in targets.items() if platform.startswith(name)]
    if not matches:
        return None
    return max(matches, key=lambda t: len(t.name))


# ── counts.json helpers ──────────────────────────────────────────────────────


def parse_counts(payload: dict[str, Any], fallback_platform: str) -> PlatformCounts:
    """Parse a workflow ``counts.json`` payload."""
    return PlatformCounts(
        platform=str(payload.get("platform") or fallback_platform),
        passed=int_or_zero(payload.get("passed")),
        skipped=int_or_zero(payload.get("skipped")),
        failed=int_or_zero(payload.get("failed")),
        error=int_or_zero(payload.get("error")),
        total=int_or_zero(payload.get("total")),
    )


def read_local_counts(counts_dir: Path) -> list[tuple[PlatformCounts, str, str]]:
    """Read all ``counts.json`` files from downloaded GitHub artifacts.

    Returns:
        List of ``(PlatformCounts, artifact_name, file_path_str)`` tuples.
    """
    rows: list[tuple[PlatformCounts, str, str]] = []
    for counts_path in sorted(counts_dir.rglob("counts.json")):
        payload = json.loads(counts_path.read_text(encoding="utf-8"))
        relative_parts = counts_path.relative_to(counts_dir).parts
        artifact_name = relative_parts[0] if relative_parts else counts_path.parent.name
        counts = parse_counts(payload, artifact_name)
        rows.append((counts, artifact_name, str(counts_path)))
    return rows


# ── pytest-json-report helpers ───────────────────────────────────────────────


def _extract_markers(keywords: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Extract structured marker values from a pytest-json-report keywords dict.

    Returns:
        Tuple of ``(hw_marker, ci_marker, layer_marker, runtime_marker, markers_raw_json)``.
    """
    hw = ci = layer = runtime = ""
    all_markers: list[str] = []
    for key in keywords:
        m = _MARKER_RE.match(key)
        if m:
            dim, val = m.group(1), m.group(2)
            all_markers.append(key)
            if dim == "hw" and not hw:
                hw = val
            elif dim == "ci" and not ci:
                ci = val
            elif dim == "layer" and not layer:
                layer = val
            elif dim == "runtime" and not runtime:
                runtime = val
    return hw, ci, layer, runtime, json.dumps(sorted(all_markers))


def _parse_nodeid(nodeid: str) -> tuple[str, str, str]:
    """Split a pytest node id into ``(test_file, test_class, test_name)``.

    Examples::

        "tests/e2e/hip/test_foo.py::test_bar"           → ("tests/e2e/hip/test_foo.py", "", "test_bar")
        "tests/e2e/hip/test_foo.py::MyClass::test_bar"  → ("tests/e2e/hip/test_foo.py", "MyClass", "test_bar")
        "tests/e2e/hip/test_foo.py::test_bar[p0]"       → ("tests/e2e/hip/test_foo.py", "", "test_bar[p0]")
    """
    parts = nodeid.split("::")
    test_file = parts[0] if parts else nodeid
    if len(parts) == 1:
        return test_file, "", ""
    if len(parts) == 2:
        return test_file, "", parts[1]
    # 3+ parts: middle parts are class/method hierarchy
    return test_file, parts[1], "::".join(parts[2:])


def _extract_failure(test: dict[str, Any]) -> tuple[str, str, str]:
    """Extract ``(phase, error_type, error_message)`` from a failed test entry.

    Inspects ``setup``, ``call``, and ``teardown`` lifecycle dicts in order.
    Returns the first one that contains a ``longrepr`` or ``crash`` key.
    """
    for phase in ("call", "setup", "teardown"):
        phase_data = test.get(phase)
        if not isinstance(phase_data, dict):
            continue
        longrepr = phase_data.get("longrepr") or ""
        crash = phase_data.get("crash") or {}
        if not longrepr and not crash:
            continue

        # Prefer structured crash info when available
        if isinstance(crash, dict):
            error_type = str(crash.get("typename") or "")
            raw_msg = str(crash.get("message") or longrepr)
        else:
            # longrepr is a plain string; try to parse the exception type from the last E-line
            error_type = ""
            last_e_line = ""
            for line in str(longrepr).splitlines():
                stripped = line.lstrip()
                if stripped.startswith("E "):
                    last_e_line = stripped[2:]
            if last_e_line:
                colon_idx = last_e_line.find(":")
                if colon_idx > 0:
                    candidate = last_e_line[:colon_idx].strip()
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate):
                        error_type = candidate
            raw_msg = str(longrepr)

        error_message = raw_msg[:_ERROR_MSG_LIMIT]
        return phase, error_type, error_message

    return "", "", ""


def parse_json_report(report_path: Path) -> list[TestCaseResult]:
    """Parse a ``pytest-json-report`` JSON file into a list of :class:`TestCaseResult`.

    Args:
        report_path: Path to the ``test_report.json`` produced by
            ``--json-report --json-report-file=test_report.json``.

    Returns:
        One :class:`TestCaseResult` per test entry in the report.
    """
    data = json.loads(report_path.read_text(encoding="utf-8"))
    results: list[TestCaseResult] = []

    for test in data.get("tests", []):
        nodeid = str(test.get("nodeid", ""))
        test_file, test_class, test_name = _parse_nodeid(nodeid)

        outcome = str(test.get("outcome", "unknown")).lower()
        # pytest-json-report uses "error" for collection/setup errors
        if outcome not in ("passed", "failed", "error", "skipped"):
            outcome = "error"

        # Total duration = sum of all phases that have a duration
        duration = 0.0
        for phase_key in ("setup", "call", "teardown"):
            phase_data = test.get(phase_key)
            if isinstance(phase_data, dict):
                duration += float(phase_data.get("duration") or 0.0)

        phase, error_type, error_message = ("", "", "")
        if outcome in ("failed", "error"):
            phase, error_type, error_message = _extract_failure(test)

        keywords = test.get("keywords") or {}
        if isinstance(keywords, list):
            keywords = {k: 1 for k in keywords}
        hw, ci, layer, runtime, markers_raw = _extract_markers(keywords)

        results.append(
            TestCaseResult(
                nodeid=nodeid,
                test_file=test_file,
                test_class=test_class,
                test_name=test_name,
                outcome=outcome,
                duration_secs=round(duration, 4),
                phase=phase,
                error_type=error_type,
                error_message=error_message,
                hw_marker=hw,
                ci_marker=ci,
                layer_marker=layer,
                runtime_marker=runtime,
                markers_raw=markers_raw,
            )
        )

    return results


def read_local_reports(reports_dir: Path) -> list[tuple[str, list[TestCaseResult]]]:
    """Scan ``reports_dir`` for per-platform ``test_report.json`` files.

    Artifact directories are named ``test-report-<platform>``; the platform
    name is recovered by stripping the ``test-report-`` prefix from the
    containing directory name.

    Returns:
        List of ``(platform, [TestCaseResult, ...])`` tuples.
    """
    platform_cases: list[tuple[str, list[TestCaseResult]]] = []
    for report_path in sorted(reports_dir.rglob("test_report.json")):
        relative_parts = report_path.relative_to(reports_dir).parts
        dir_name = relative_parts[0] if relative_parts else report_path.parent.name
        platform = dir_name.removeprefix("test-report-") if dir_name.startswith("test-report-") else dir_name
        try:
            cases = parse_json_report(report_path)
            platform_cases.append((platform, cases))
            print(f"  Parsed {len(cases)} test cases for platform '{platform}'", flush=True)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"WARNING: could not parse {report_path}: {exc}", flush=True)
    return platform_cases


# ── GitHub metadata helpers ──────────────────────────────────────────────────


def fetch_run_metadata(repo: str, run_id: int) -> dict[str, Any]:
    """Fetch GitHub workflow-run metadata."""
    return fetch_json(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}")


def fetch_workflow_jobs(repo: str, run_id: int) -> list[dict[str, Any]]:
    """Fetch GitHub workflow jobs for a run (handles pagination)."""
    jobs_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs"
    try:
        return paged_github_items(jobs_url, "jobs")
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"WARNING: could not fetch workflow jobs: {exc}", flush=True)
        return []


def fetch_artifact_source_run_id(jobs: list[dict[str, Any]]) -> int | None:
    """Parse the resolved artifact-source run id from the summary job name."""
    for job in jobs:
        name = str_or_empty(job.get("name"))
        match = re.search(r"Nightly Test Summary\s+—\s+(\d+)", name)
        if match:
            return int(match.group(1))
    return None


def build_platform_job_metadata(jobs: list[dict[str, Any]]) -> dict[str, PlatformJobMetadata]:
    """Map platform names from the matrix to their GitHub Actions job metadata."""
    platform_jobs: dict[str, PlatformJobMetadata] = {}
    suffix = " / e2e tests"
    for job in jobs:
        name = str_or_empty(job.get("name"))
        if not name.endswith(suffix):
            continue
        platform_jobs[name[: -len(suffix)]] = PlatformJobMetadata(
            url=str_or_empty(job.get("html_url") or job.get("url")),
            status=str_or_empty(job.get("status")),
            conclusion=str_or_empty(job.get("conclusion")),
        )
    return platform_jobs


# ── DDL helpers ──────────────────────────────────────────────────────────────

_COLUMNS = [
    # ── Discriminator
    "row_type",
    # ── Shared (on every row)
    "run_id",
    "platform",
    "gpu_arch",
    "head_branch",
    "head_sha",
    "time_started",
    "run_url",
    "repository",
    "workflow_name",
    "run_number",
    "run_attempt",
    # ── Summary-only (empty/zero on case rows)
    "source_repo",
    "artifact_source",
    "artifact_source_run_id",
    "artifact_source_url",
    "event",
    "status",
    "conclusion",
    "time_updated",
    "target_name",
    "runs_on",
    "artifact_group",
    "tests_filters",
    "artifact_name",
    "artifact_url",
    "tests_pass",
    "tests_fail",
    "tests_error",
    "tests_skip",
    "total_test_count",
    "result_status",
    "raw_counts_json",
    # ── Case-only (empty/zero on summary rows)
    "nodeid",
    "test_file",
    "test_class",
    "test_name",
    "outcome",
    "duration_secs",
    "phase",
    "error_type",
    "error_message",
    "hw_marker",
    "ci_marker",
    "layer_marker",
    "runtime_marker",
    "markers_raw",
]

_COLUMN_INDEX = {name: idx for idx, name in enumerate(_COLUMNS)}


def create_table(client, table: str) -> None:
    """Create the converged ``rocmtests_nightly_results`` table plus the rollup MV.

    Also creates the ``<table>_rollup`` SummingMergeTree target and the
    ``<table>_rollup_mv`` Materialized View that pre-aggregates case-row
    outcome counts per ``(run_id, platform, test_file, outcome)``.

    Args:
        client: Active ClickHouse client.
        table: Validated table name (may be database-qualified).
    """
    table = validate_clickhouse_table_name(table)
    rollup = validate_clickhouse_table_name(_rollup_table_name(table))
    rollup_mv = validate_clickhouse_table_name(_rollup_mv_name(table))

    # Main converged table
    client.command(f"""
CREATE TABLE IF NOT EXISTS {table}
(
    -- Discriminator: 'summary' (aggregate per platform) or 'case' (per test function)
    row_type                 LowCardinality(String),

    -- Shared columns (populated on every row)
    run_id                   UInt64,
    platform                 LowCardinality(String),
    gpu_arch                 LowCardinality(String),
    head_branch              String,
    head_sha                 String,
    time_started             DateTime64(3, 'UTC'),
    run_url                  String,
    repository               String,
    workflow_name            String,
    run_number               UInt64,
    run_attempt              UInt32,

    -- Summary-only columns (empty / zero on case rows)
    source_repo              String,
    artifact_source          String,
    artifact_source_run_id   Nullable(UInt64),
    artifact_source_url      String,
    event                    String,
    status                   String,
    conclusion               String,
    time_updated             Nullable(DateTime64(3, 'UTC')),
    target_name              String,
    runs_on                  String,
    artifact_group           String,
    tests_filters            String,
    artifact_name            String,
    artifact_url             String,
    tests_pass               UInt32,
    tests_fail               UInt32,
    tests_error              UInt32,
    tests_skip               UInt32,
    total_test_count         UInt32,
    result_status            LowCardinality(String),
    raw_counts_json          String,

    -- Case-only columns (empty / zero on summary rows)
    nodeid                   String,
    test_file                String,
    test_class               String,
    test_name                String,
    outcome                  LowCardinality(String),
    duration_secs            Float32,
    phase                    LowCardinality(String),
    error_type               String,
    error_message            String,
    hw_marker                LowCardinality(String),
    ci_marker                LowCardinality(String),
    layer_marker             LowCardinality(String),
    runtime_marker           LowCardinality(String),
    markers_raw              String,

    inserted_at              DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
ORDER BY (run_id, platform, row_type, nodeid)
""")
    print(f"Created table {table}", flush=True)

    # Rollup target table for the MV
    client.command(f"""
CREATE TABLE IF NOT EXISTS {rollup}
(
    run_id       UInt64,
    platform     LowCardinality(String),
    gpu_arch     LowCardinality(String),
    test_file    String,
    outcome      LowCardinality(String),
    test_count   UInt64,
    total_secs   Float64
)
ENGINE = SummingMergeTree((test_count, total_secs))
ORDER BY (run_id, platform, test_file, outcome)
""")
    print(f"Created rollup table {rollup}", flush=True)

    # Materialized View — fires on INSERT of case rows
    client.command(f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS {rollup_mv}
TO {rollup}
AS
SELECT
    run_id,
    platform,
    gpu_arch,
    test_file,
    outcome,
    count()            AS test_count,
    sum(duration_secs) AS total_secs
FROM {table}
WHERE row_type = 'case'
GROUP BY run_id, platform, gpu_arch, test_file, outcome
""")
    print(f"Created rollup materialized view {rollup_mv}", flush=True)


def drop_and_create_table(client, table: str) -> None:
    """Drop and recreate the main table plus rollup objects.

    Use this for schema migrations.  Existing data is permanently deleted.

    Args:
        client: Active ClickHouse client.
        table: Validated table name.
    """
    table = validate_clickhouse_table_name(table)
    rollup = validate_clickhouse_table_name(_rollup_table_name(table))
    rollup_mv = validate_clickhouse_table_name(_rollup_mv_name(table))

    print(f"Dropping {rollup_mv}, {rollup}, {table} …", flush=True)
    client.command(f"DROP VIEW IF EXISTS {rollup_mv}")
    client.command(f"DROP TABLE IF EXISTS {rollup}")
    client.command(f"DROP TABLE IF EXISTS {table}")
    create_table(client, table)


# ── Row builders ─────────────────────────────────────────────────────────────


def _shared_run_values(
    run_metadata: dict[str, Any],
    github_repo: str,
    target: TargetConfig | None,
) -> dict[str, Any]:
    """Build the shared run-level column values used by both row types."""
    run_id = int(run_metadata["id"])
    run_url = str_or_empty(run_metadata.get("html_url") or f"https://github.com/{github_repo}/actions/runs/{run_id}")
    time_started = (
        parse_iso_datetime(run_metadata.get("run_started_at"))
        or parse_iso_datetime(run_metadata.get("created_at"))
        or datetime.now(timezone.utc)
    )
    return {
        "run_id": run_id,
        "run_url": run_url,
        "time_started": time_started,
        "repository": github_repo,
        "workflow_name": str_or_empty(run_metadata.get("name")),
        "run_number": int_or_zero(run_metadata.get("run_number")),
        "run_attempt": int_or_zero(run_metadata.get("run_attempt")),
        "head_branch": str_or_empty(run_metadata.get("head_branch")),
        "head_sha": str_or_empty(run_metadata.get("head_sha")),
        "gpu_arch": target.gpu_arch if target else "",
    }


def build_summary_insert_rows(
    *,
    github_repo: str,
    source_repo: str,
    artifact_source: str,
    run_metadata: dict[str, Any],
    artifact_source_run_id: int | None,
    counts_rows: list[tuple[PlatformCounts, str, str]],
    targets: dict[str, TargetConfig],
    platform_jobs: dict[str, PlatformJobMetadata],
    include_total_row: bool,
) -> tuple[list[str], list[list[Any]]]:
    """Build summary (``row_type = 'summary'``) rows for all platforms.

    Returns:
        ``(columns, rows)`` ready for :func:`insert_rows`.
    """
    run_id = int(run_metadata["id"])
    source_url = (
        f"https://github.com/{source_repo}/actions/runs/{artifact_source_run_id}" if artifact_source_run_id else ""
    )
    shared = _shared_run_values(run_metadata, github_repo, None)
    run_url = shared["run_url"]

    rows: list[list[Any]] = []

    for counts, artifact_name, counts_path in counts_rows:
        target = resolve_target(counts.platform, targets)
        platform_job = platform_jobs.get(counts.platform)
        artifact_url = platform_job.url if platform_job and platform_job.url else counts_path
        result_status = "passed" if counts.failed == 0 and counts.error == 0 and counts.total > 0 else "failed"
        raw_counts_json = json.dumps(
            {
                "platform": counts.platform,
                "passed": counts.passed,
                "skipped": counts.skipped,
                "failed": counts.failed,
                "error": counts.error,
                "total": counts.total,
            },
            sort_keys=True,
        )
        row: list[Any] = [None] * len(_COLUMNS)
        for col, val in [
            ("row_type", "summary"),
            ("run_id", run_id),
            ("platform", counts.platform),
            ("gpu_arch", target.gpu_arch if target else ""),
            ("head_branch", shared["head_branch"]),
            ("head_sha", shared["head_sha"]),
            ("time_started", shared["time_started"]),
            ("run_url", run_url),
            ("repository", github_repo),
            ("workflow_name", shared["workflow_name"]),
            ("run_number", shared["run_number"]),
            ("run_attempt", shared["run_attempt"]),
            ("source_repo", source_repo),
            ("artifact_source", artifact_source),
            ("artifact_source_run_id", artifact_source_run_id),
            ("artifact_source_url", source_url),
            ("event", str_or_empty(run_metadata.get("event"))),
            ("status", platform_job.status if platform_job else ""),
            ("conclusion", platform_job.conclusion if platform_job else ""),
            ("time_updated", parse_iso_datetime(run_metadata.get("updated_at"))),
            ("target_name", target.name if target else counts.platform),
            ("runs_on", target.runs_on if target else ""),
            ("artifact_group", target.artifact_group if target else ""),
            ("tests_filters", target.tests_filters if target else ""),
            ("artifact_name", artifact_name),
            ("artifact_url", artifact_url),
            ("tests_pass", counts.passed),
            ("tests_fail", counts.failed),
            ("tests_error", counts.error),
            ("tests_skip", counts.skipped),
            ("total_test_count", counts.total),
            ("result_status", result_status),
            ("raw_counts_json", raw_counts_json),
            # Case-only columns — empty on summary rows
            ("nodeid", ""),
            ("test_file", ""),
            ("test_class", ""),
            ("test_name", ""),
            ("outcome", ""),
            ("duration_secs", 0.0),
            ("phase", ""),
            ("error_type", ""),
            ("error_message", ""),
            ("hw_marker", ""),
            ("ci_marker", ""),
            ("layer_marker", ""),
            ("runtime_marker", ""),
            ("markers_raw", "[]"),
        ]:
            row[_COLUMN_INDEX[col]] = val
        rows.append(row)

    if include_total_row and rows:

        def _sum(col: str) -> int:
            return sum(r[_COLUMN_INDEX[col]] for r in rows)

        total_pass = _sum("tests_pass")
        total_fail = _sum("tests_fail")
        total_error = _sum("tests_error")
        total_skip = _sum("tests_skip")
        total_count = _sum("total_test_count")
        total_status = "passed" if total_fail == 0 and total_error == 0 and total_count > 0 else "failed"
        total_counts_json = json.dumps(
            {
                "source": "computed_from_platform_counts",
                "platforms": len(rows),
                "passed": total_pass,
                "skipped": total_skip,
                "failed": total_fail,
                "error": total_error,
                "total": total_count,
            },
            sort_keys=True,
        )
        total_row = list(rows[0])
        for col, val in [
            ("platform", "TOTAL"),
            ("gpu_arch", "all"),
            ("target_name", "TOTAL"),
            ("runs_on", ""),
            ("artifact_group", "all"),
            ("tests_filters", ""),
            ("artifact_name", "computed-total"),
            ("artifact_url", run_url),
            ("status", ""),
            ("conclusion", ""),
            ("tests_pass", total_pass),
            ("tests_fail", total_fail),
            ("tests_error", total_error),
            ("tests_skip", total_skip),
            ("total_test_count", total_count),
            ("result_status", total_status),
            ("raw_counts_json", total_counts_json),
        ]:
            total_row[_COLUMN_INDEX[col]] = val
        rows.append(total_row)

    return _COLUMNS, rows


def build_case_insert_rows(
    *,
    run_id: int,
    platform: str,
    cases: list[TestCaseResult],
    run_metadata: dict[str, Any],
    github_repo: str,
    target: TargetConfig | None,
    source_repo: str,
    artifact_source: str,
    artifact_source_run_id: int | None,
    platform_job: PlatformJobMetadata | None,
) -> list[list[Any]]:
    """Build case (``row_type = 'case'``) rows for a single platform.

    Args:
        run_id: GitHub Actions run id.
        platform: Platform name string (e.g. ``linux-gfx942``).
        cases: Parsed test case results from :func:`parse_json_report`.
        run_metadata: GitHub workflow run metadata dict.
        github_repo: Repo name (e.g. ``ROCm/rocm-tests``).
        target: Matching :class:`TargetConfig` from testplan, or ``None``.
        source_repo: Artifact source repo.
        artifact_source: Artifact source workflow/tag.
        artifact_source_run_id: Resolved artifact source run id, or ``None``.
        platform_job: GitHub Actions job metadata for this platform, or ``None``.

    Returns:
        List of row value lists aligned to :data:`_COLUMNS`.
    """
    shared = _shared_run_values(run_metadata, github_repo, target)
    run_url = shared["run_url"]
    source_url = (
        f"https://github.com/{source_repo}/actions/runs/{artifact_source_run_id}" if artifact_source_run_id else ""
    )
    time_started = shared["time_started"]
    artifact_url = platform_job.url if platform_job and platform_job.url else run_url

    rows: list[list[Any]] = []
    for case in cases:
        row: list[Any] = [None] * len(_COLUMNS)
        for col, val in [
            ("row_type", "case"),
            ("run_id", run_id),
            ("platform", platform),
            ("gpu_arch", shared["gpu_arch"]),
            ("head_branch", shared["head_branch"]),
            ("head_sha", shared["head_sha"]),
            ("time_started", time_started),
            ("run_url", run_url),
            ("repository", github_repo),
            ("workflow_name", shared["workflow_name"]),
            ("run_number", shared["run_number"]),
            ("run_attempt", shared["run_attempt"]),
            # Summary-only columns — empty on case rows
            ("source_repo", source_repo),
            ("artifact_source", artifact_source),
            ("artifact_source_run_id", artifact_source_run_id),
            ("artifact_source_url", source_url),
            ("event", str_or_empty(run_metadata.get("event"))),
            ("status", platform_job.status if platform_job else ""),
            ("conclusion", platform_job.conclusion if platform_job else ""),
            ("time_updated", parse_iso_datetime(run_metadata.get("updated_at"))),
            ("target_name", target.name if target else platform),
            ("runs_on", target.runs_on if target else ""),
            ("artifact_group", target.artifact_group if target else ""),
            ("tests_filters", target.tests_filters if target else ""),
            ("artifact_name", ""),
            ("artifact_url", artifact_url),
            ("tests_pass", 0),
            ("tests_fail", 0),
            ("tests_error", 0),
            ("tests_skip", 0),
            ("total_test_count", 0),
            ("result_status", ""),
            ("raw_counts_json", ""),
            # Case-only columns
            ("nodeid", case.nodeid),
            ("test_file", case.test_file),
            ("test_class", case.test_class),
            ("test_name", case.test_name),
            ("outcome", case.outcome),
            ("duration_secs", case.duration_secs),
            ("phase", case.phase),
            ("error_type", case.error_type),
            ("error_message", case.error_message),
            ("hw_marker", case.hw_marker),
            ("ci_marker", case.ci_marker),
            ("layer_marker", case.layer_marker),
            ("runtime_marker", case.runtime_marker),
            ("markers_raw", case.markers_raw),
        ]:
            row[_COLUMN_INDEX[col]] = val
        rows.append(row)
    return rows


# ── Insert helpers ───────────────────────────────────────────────────────────


def insert_rows(client, table: str, columns: list[str], rows: list[list[Any]]) -> None:
    """Insert pre-built rows into ClickHouse.

    Args:
        client: Active ClickHouse client.
        table: Validated table name.
        columns: Column name list (must match row value order).
        rows: Row value lists.

    Raises:
        RuntimeError: When ``rows`` is empty.
    """
    if not rows:
        raise RuntimeError("No rows to insert")
    table = validate_clickhouse_table_name(table)
    client.insert(table, rows, column_names=columns)
    print(f"Inserted {len(rows)} row(s) into {table}", flush=True)


def insert_case_rows(
    client,
    table: str,
    *,
    run_id: int,
    run_metadata: dict[str, Any],
    github_repo: str,
    source_repo: str,
    artifact_source: str,
    artifact_source_run_id: int | None,
    targets: dict[str, TargetConfig],
    platform_jobs: dict[str, PlatformJobMetadata],
    platform_cases: list[tuple[str, list[TestCaseResult]]],
) -> None:
    """Build and insert all case rows across all platforms in a single INSERT.

    Per-platform parse failures are logged as warnings and skipped; the
    remaining platforms are still inserted.

    Args:
        client: Active ClickHouse client.
        table: Validated table name.
        run_id: GitHub Actions run id.
        run_metadata: GitHub workflow run metadata dict.
        github_repo: Repo name.
        source_repo: Artifact source repo.
        artifact_source: Artifact source workflow/tag.
        artifact_source_run_id: Resolved artifact source run id, or ``None``.
        targets: Loaded testplan targets.
        platform_jobs: Per-platform GitHub Actions job metadata.
        platform_cases: Output of :func:`read_local_reports`.
    """
    all_rows: list[list[Any]] = []
    for platform, cases in platform_cases:
        if not cases:
            print(f"  Skipping case insert for '{platform}' — no test cases parsed", flush=True)
            continue
        target = resolve_target(platform, targets)
        platform_job = platform_jobs.get(platform)
        try:
            rows = build_case_insert_rows(
                run_id=run_id,
                platform=platform,
                cases=cases,
                run_metadata=run_metadata,
                github_repo=github_repo,
                target=target,
                source_repo=source_repo,
                artifact_source=artifact_source,
                artifact_source_run_id=artifact_source_run_id,
                platform_job=platform_job,
            )
            all_rows.extend(rows)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"WARNING: failed to build case rows for '{platform}': {exc}", flush=True)

    if all_rows:
        insert_rows(client, table, _COLUMNS, all_rows)
    else:
        print("No case rows to insert (all platforms skipped or empty)", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-id", type=int, required=True, help="GitHub Actions run id")
    parser.add_argument("--github-repo", required=True, help="GitHub repo containing the nightly run")
    parser.add_argument("--source-repo", required=True, help="Artifact source repo")
    parser.add_argument("--artifact-source", required=True, help="Artifact source workflow/tag")
    parser.add_argument(
        "--table",
        required=True,
        help="Destination ClickHouse table name (e.g. rocmtests_nightly_results)",
    )
    parser.add_argument(
        "--testplan",
        type=Path,
        default=Path("testplan.ini"),
        help="Path to testplan.ini (default: %(default)s)",
    )
    parser.add_argument(
        "--counts-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded test-counts-* artifacts",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help="Directory containing downloaded test-report-* artifacts (optional; enables case rows)",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        default=False,
        help="Create the destination table (and rollup MV) if they do not exist",
    )
    parser.add_argument(
        "--recreate-table",
        action="store_true",
        default=False,
        help="DROP and recreate the destination table + rollup MV (data loss — use for schema migrations)",
    )
    parser.add_argument(
        "--no-total-row",
        action="store_true",
        help="Omit the computed TOTAL summary row",
    )
    return parser.parse_args()


def main() -> bool:
    """Script entry point.

    Returns:
        ``True`` on success, ``False`` on error (caller should exit with 1).
    """
    args = parse_args()
    try:
        targets = load_testplan(args.testplan)
        counts_rows = read_local_counts(args.counts_dir)
        print(f"Loaded {len(counts_rows)} platform count row(s) from {args.counts_dir}", flush=True)

        run_metadata = fetch_run_metadata(args.github_repo, args.run_id)
        workflow_jobs = fetch_workflow_jobs(args.github_repo, args.run_id)
        artifact_source_run_id = fetch_artifact_source_run_id(workflow_jobs)
        platform_jobs = build_platform_job_metadata(workflow_jobs)
        run_id = int(run_metadata["id"])

        columns, summary_rows = build_summary_insert_rows(
            github_repo=args.github_repo,
            source_repo=args.source_repo,
            artifact_source=args.artifact_source,
            run_metadata=run_metadata,
            artifact_source_run_id=artifact_source_run_id,
            counts_rows=counts_rows,
            targets=targets,
            platform_jobs=platform_jobs,
            include_total_row=not args.no_total_row,
        )

        client = build_client()

        if args.recreate_table:
            drop_and_create_table(client, args.table)
        elif args.create_table:
            create_table(client, args.table)

        insert_rows(client, args.table, columns, summary_rows)

        if args.reports_dir:
            print(f"Loading per-test JSON reports from {args.reports_dir}", flush=True)
            platform_cases = read_local_reports(args.reports_dir)
            if platform_cases:
                insert_case_rows(
                    client,
                    args.table,
                    run_id=run_id,
                    run_metadata=run_metadata,
                    github_repo=args.github_repo,
                    source_repo=args.source_repo,
                    artifact_source=args.artifact_source,
                    artifact_source_run_id=artifact_source_run_id,
                    targets=targets,
                    platform_jobs=platform_jobs,
                    platform_cases=platform_cases,
                )
            else:
                print("No JSON reports found under --reports-dir", flush=True)

        return True

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if "CERTIFICATE_VERIFY_FAILED" in str(exc):
            print(
                "Install/update certifi or set CLICKHOUSE_CA_CERT to your CA bundle path.",
                file=sys.stderr,
                flush=True,
            )
        return False


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
