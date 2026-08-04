#!/usr/bin/env python3
"""Insert ROCm nightly workflow results into ClickHouse.

The updater is intended to run from the e2e-nightly report job after all
per-platform jobs have uploaded their ``test-counts-*`` artifacts. It inserts
one row per platform plus one computed ``TOTAL`` row.
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

DEFAULT_GITHUB_REPO = "ROCm/rocm-tests"
DEFAULT_SOURCE_REPO = "ROCm/rockrel"
DEFAULT_ARTIFACT_SOURCE = "multi_arch_release.yml"
DEFAULT_TABLE = "rocmtests_nightly_runs"


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


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""

    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_iso_datetime(value: str | None) -> datetime | None:
    """Convert a GitHub ISO timestamp to UTC datetime."""

    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def int_or_zero(value: Any) -> int:
    """Convert ClickHouse/GitHub count-like values to int."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def str_or_empty(value: Any) -> str:
    """Convert nullable GitHub metadata fields to ClickHouse-safe strings."""

    if value is None:
        return ""
    return str(value)


def default_ca_cert() -> str | None:
    """Return certifi's CA bundle path when certifi is installed."""

    try:
        import certifi
    except ImportError:
        return None
    return certifi.where()


def build_client():
    """Create a ClickHouse client from environment variables."""

    password = os.environ.get("CLICKHOUSE_PASSWORD")
    if not password:
        raise RuntimeError("CLICKHOUSE_PASSWORD must be set in the environment")

    verify_ssl = env_bool("CLICKHOUSE_VERIFY_SSL", True)
    ca_cert = os.environ.get("CLICKHOUSE_CA_CERT")
    if verify_ssl and ca_cert is None:
        ca_cert = default_ca_cert()

    kwargs = {
        "host": os.environ.get("CLICKHOUSE_HOST", "loqt5hpxs6.us-east-2.aws.clickhouse.cloud"),
        "port": int(os.environ.get("CLICKHOUSE_PORT", "8443")),
        "username": os.environ.get("CLICKHOUSE_USERNAME", "rocm-tests-hud"),
        "password": password,
        "database": os.environ.get("CLICKHOUSE_DATABASE", "quartz"),
        "secure": env_bool("CLICKHOUSE_SECURE", True),
        "verify": verify_ssl,
        "connect_timeout": int(os.environ.get("CLICKHOUSE_CONNECT_TIMEOUT", "10")),
        "send_receive_timeout": int(os.environ.get("CLICKHOUSE_SEND_RECEIVE_TIMEOUT", "30")),
    }
    if ca_cert:
        kwargs["ca_cert"] = ca_cert

    return clickhouse_connect.get_client(**kwargs)


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
    """Return an SSL context using certifi when available."""

    if not env_bool("GITHUB_VERIFY_SSL", True):
        return ssl._create_unverified_context()
    ca_cert = default_ca_cert()
    if ca_cert is None:
        return None
    return ssl.create_default_context(cafile=ca_cert)


def fetch_json(url: str) -> dict[str, Any]:
    """Fetch JSON from the GitHub API."""

    request = Request(url, headers=github_headers())
    with urlopen(request, timeout=30, context=ssl_context()) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return payload


def paged_github_items(url: str, key: str) -> list[dict[str, Any]]:
    """Collect paged GitHub API results."""

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


def load_testplan(testplan_path: Path) -> dict[str, TargetConfig]:
    """Read target config from rocm-tests ``testplan.ini``."""

    if not testplan_path.is_file():
        print(f"WARNING: testplan.ini not found at {testplan_path}; gpu_arch mapping will be partial", flush=True)
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

    matches = [target for name, target in targets.items() if platform.startswith(name)]
    if not matches:
        return None
    return max(matches, key=lambda target: len(target.name))


def parse_counts(payload: dict[str, Any], fallback_platform: str) -> PlatformCounts:
    """Parse workflow ``counts.json`` data."""

    return PlatformCounts(
        platform=str(payload.get("platform") or fallback_platform),
        passed=int_or_zero(payload.get("passed")),
        skipped=int_or_zero(payload.get("skipped")),
        failed=int_or_zero(payload.get("failed")),
        error=int_or_zero(payload.get("error")),
        total=int_or_zero(payload.get("total")),
    )


def read_local_counts(counts_dir: Path) -> list[tuple[PlatformCounts, str, str]]:
    """Read local ``counts.json`` files from downloaded GitHub artifacts."""

    rows: list[tuple[PlatformCounts, str, str]] = []
    for counts_path in sorted(counts_dir.rglob("counts.json")):
        payload = json.loads(counts_path.read_text(encoding="utf-8"))
        relative_parts = counts_path.relative_to(counts_dir).parts
        artifact_name = relative_parts[0] if relative_parts else counts_path.parent.name
        counts = parse_counts(payload, artifact_name)
        rows.append((counts, artifact_name, str(counts_path)))
    return rows


def fetch_run_metadata(repo: str, run_id: int) -> dict[str, Any]:
    """Fetch GitHub workflow-run metadata."""

    return fetch_json(f"https://api.github.com/repos/{repo}/actions/runs/{run_id}")


def fetch_workflow_jobs(repo: str, run_id: int) -> list[dict[str, Any]]:
    """Fetch GitHub workflow jobs for the run."""

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


def create_table(client, table: str) -> None:
    """Create the destination ClickHouse table if requested."""

    client.command(f"""
CREATE TABLE IF NOT EXISTS {table}
(
    run_id UInt64,
    source_repo String,
    artifact_source String,
    artifact_source_run_id Nullable(UInt64),
    workflow_name String,
    run_number UInt64,
    run_attempt UInt32,
    run_url String,
    artifact_source_url String,
    repository String,
    head_branch String,
    head_sha String,
    event String,
    status String,
    conclusion String,
    time_started DateTime64(3, 'UTC'),
    time_updated Nullable(DateTime64(3, 'UTC')),
    platform String,
    gpu_arch String,
    target_name String,
    runs_on String,
    artifact_group String,
    tests_filters String,
    artifact_name String,
    artifact_url String,
    tests_pass UInt32,
    tests_fail UInt32,
    tests_error UInt32,
    tests_skip UInt32,
    total_test_count UInt32,
    result_status String,
    raw_counts_json String,
    inserted_at DateTime64(3, 'UTC') DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
ORDER BY (run_id, platform)
""")


def build_insert_rows(
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
    """Transform GitHub workflow data into ClickHouse insert rows."""

    run_id = int(run_metadata["id"])
    run_url = str_or_empty(run_metadata.get("html_url") or f"https://github.com/{github_repo}/actions/runs/{run_id}")
    source_url = (
        f"https://github.com/{source_repo}/actions/runs/{artifact_source_run_id}" if artifact_source_run_id else ""
    )
    columns = [
        "run_id",
        "source_repo",
        "artifact_source",
        "artifact_source_run_id",
        "workflow_name",
        "run_number",
        "run_attempt",
        "run_url",
        "artifact_source_url",
        "repository",
        "head_branch",
        "head_sha",
        "event",
        "status",
        "conclusion",
        "time_started",
        "time_updated",
        "platform",
        "gpu_arch",
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
    ]

    rows: list[list[Any]] = []
    for counts, artifact_name, counts_path in counts_rows:
        target = resolve_target(counts.platform, targets)
        platform_job = platform_jobs.get(counts.platform)
        artifact_url = platform_job.url if platform_job and platform_job.url else counts_path
        platform_status = platform_job.status if platform_job else ""
        platform_conclusion = platform_job.conclusion if platform_job else ""
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
        rows.append(
            [
                run_id,
                source_repo,
                artifact_source,
                artifact_source_run_id,
                str_or_empty(run_metadata.get("name")),
                int_or_zero(run_metadata.get("run_number")),
                int_or_zero(run_metadata.get("run_attempt")),
                run_url,
                source_url,
                github_repo,
                str_or_empty(run_metadata.get("head_branch")),
                str_or_empty(run_metadata.get("head_sha")),
                str_or_empty(run_metadata.get("event")),
                platform_status,
                platform_conclusion,
                parse_iso_datetime(run_metadata.get("run_started_at"))
                or parse_iso_datetime(run_metadata.get("created_at"))
                or datetime.now(timezone.utc),
                parse_iso_datetime(run_metadata.get("updated_at")),
                counts.platform,
                target.gpu_arch if target else "",
                target.name if target else counts.platform,
                target.runs_on if target else "",
                target.artifact_group if target else "",
                target.tests_filters if target else "",
                artifact_name,
                artifact_url,
                counts.passed,
                counts.failed,
                counts.error,
                counts.skipped,
                counts.total,
                result_status,
                raw_counts_json,
            ]
        )

    if include_total_row and rows:
        total_pass = sum(row[columns.index("tests_pass")] for row in rows)
        total_fail = sum(row[columns.index("tests_fail")] for row in rows)
        total_error = sum(row[columns.index("tests_error")] for row in rows)
        total_skip = sum(row[columns.index("tests_skip")] for row in rows)
        total_count = sum(row[columns.index("total_test_count")] for row in rows)
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

        def set_total_value(column: str, value: Any) -> None:
            total_row[columns.index(column)] = value

        set_total_value("platform", "TOTAL")
        set_total_value("gpu_arch", "all")
        set_total_value("target_name", "TOTAL")
        set_total_value("runs_on", "")
        set_total_value("artifact_group", "all")
        set_total_value("tests_filters", "")
        set_total_value("artifact_name", "computed-total")
        set_total_value("artifact_url", run_url)
        set_total_value("status", "")
        set_total_value("conclusion", "")
        set_total_value("tests_pass", total_pass)
        set_total_value("tests_fail", total_fail)
        set_total_value("tests_error", total_error)
        set_total_value("tests_skip", total_skip)
        set_total_value("total_test_count", total_count)
        set_total_value("result_status", total_status)
        set_total_value("raw_counts_json", total_counts_json)
        rows.append(total_row)

    return columns, rows


def insert_rows(client, table: str, columns: list[str], rows: list[list[Any]]) -> None:
    """Insert rows into ClickHouse."""

    if not rows:
        raise RuntimeError("No platform count rows were found; nothing to insert")
    client.insert(table, rows, column_names=columns)
    print(f"Inserted {len(rows)} row(s) into ClickHouse table {table}", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, required=True, help="rocm-tests GitHub Actions run id")
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO, help="GitHub repo containing the nightly run")
    parser.add_argument(
        "--source-repo",
        default=DEFAULT_SOURCE_REPO,
        help="Artifact source repo recorded in ClickHouse",
    )
    parser.add_argument("--artifact-source", default=DEFAULT_ARTIFACT_SOURCE, help="Artifact source workflow/tag")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Destination ClickHouse table name")
    parser.add_argument("--testplan", type=Path, default=Path("testplan.ini"), help="Path to rocm-tests testplan.ini")
    parser.add_argument(
        "--counts-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded test-counts artifacts",
    )
    parser.add_argument(
        "--create-table",
        action="store_true",
        default=False,
        help="Create the destination table before inserting. Disabled by default.",
    )
    parser.add_argument("--no-total-row", action="store_true", help="Do not add a computed TOTAL row")
    return parser.parse_args()


def main() -> bool:
    """Script entry point."""

    args = parse_args()
    try:
        targets = load_testplan(args.testplan)
        counts_rows = read_local_counts(args.counts_dir)
        print(f"Loaded {len(counts_rows)} platform count row(s) from {args.counts_dir}", flush=True)

        run_metadata = fetch_run_metadata(args.github_repo, args.run_id)
        workflow_jobs = fetch_workflow_jobs(args.github_repo, args.run_id)
        artifact_source_run_id = fetch_artifact_source_run_id(workflow_jobs)
        platform_jobs = build_platform_job_metadata(workflow_jobs)
        columns, rows = build_insert_rows(
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
        if args.create_table:
            create_table(client, args.table)
        insert_rows(client, args.table, columns, rows)
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
