#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run trivy for the current checkout.

- Download + cache the pinned `trivy` release binary and verify it.
- Require `trivy.yaml` at the repo root (hard error if missing).
- Derive change sets from the GitHub event for changed/all scans; in
  'changed' mode scan the whole subtree when any audited file changed
  (trivy needs the full tree to resolve deps / cross-file IaC), else
  no-op. Run per requested format and emit SARIF/non-SARIF paths plus
  a severity tally.

Default scanners are `misconfig,vuln`; `secret` is omitted because
gitleaks already covers secret detection here.

Exit codes: 0 clean/empty changed set; 1 findings at/above threshold or
bad formats/scanners; 2 input/config/runtime errors.

Inputs come from CLI flags with `TRIVY_*` env var defaults. All
`TRIVY_*` vars are stripped from the trivy subprocess environment so our
workflow inputs don't double-apply via trivy's own env-driven CLI.
"""

import argparse
import fnmatch
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

THEROCK_DIR = Path(__file__).resolve().parent.parent.parent

# Add build_tools to path for github_actions imports.
sys.path.insert(0, str(THEROCK_DIR / "build_tools"))
from github_actions.github_actions_api import (  # noqa: E402
    gha_append_step_summary,
    gha_load_github_event,
    gha_set_output,
)

log = logging.getLogger(__name__)

# Map user-visible report formats to file extensions.
_SUPPORTED_FORMATS: dict[str, str] = {
    "sarif": "sarif",
    "json": "json",
    "table": "txt",
    "cyclonedx": "cdx.json",
    "spdx-json": "spdx.json",
    "github": "github.json",
}
_TRIVY_VERSION = "0.70.0"
_CONFIG_PATH = "trivy.yaml"
# Ascending severity order; threshold comparisons rely on it. Trivy has a
# CRITICAL tier (unlike bandit/zizmor); UNKNOWN never satisfies a threshold.
_SEVERITY_ORDER: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
_SEVERITY_CHOICES: tuple[str, ...] = tuple(s.lower() for s in _SEVERITY_ORDER)
_DEFAULT_SEVERITY_THRESHOLD = "high"
# Trivy scanners: vuln (CVEs), misconfig (IaC), secret, license.
_SUPPORTED_SCANNERS: tuple[str, ...] = ("vuln", "misconfig", "secret", "license")
# Omits 'secret': gitleaks already covers secret detection here.
_DEFAULT_SCANNERS = "misconfig,vuln"
# Internal JSON tally pass output; cleaned up before returning.
_INTERNAL_TALLY_PATH = "trivy-tally.json"
_DOWNLOAD_TIMEOUT_SECONDS = 60
# Diff filter for 'changed' mode: dependency manifests/lockfiles (drive
# vuln) plus container/IaC sources (drive misconfig). Broad globs like
# **/*.yaml are excluded so unrelated YAML changes don't defeat the
# no-op fast path; trivy_main.yml runs 'all' to catch the rest.
_AUDITED_PATTERNS: tuple[str, ...] = (
    # Python
    "**/pyproject.toml", "pyproject.toml",
    "**/requirements*.txt", "requirements*.txt",
    "**/Pipfile", "**/Pipfile.lock",
    "**/poetry.lock",
    "**/setup.py", "**/setup.cfg",
    # JavaScript / Node
    "**/package.json", "**/package-lock.json", "**/yarn.lock",
    "**/pnpm-lock.yaml", "**/npm-shrinkwrap.json",
    # Rust
    "**/Cargo.toml", "**/Cargo.lock",
    # Go
    "**/go.mod", "**/go.sum",
    # Java / Kotlin
    "**/pom.xml",
    "**/build.gradle", "**/build.gradle.kts",
    "**/gradle.lockfile",
    # .NET
    "**/*.csproj", "**/packages.config", "**/packages.lock.json",
    # Ruby
    "**/Gemfile", "**/Gemfile.lock", "**/*.gemspec",
    # PHP
    "**/composer.json", "**/composer.lock",
    # Container manifests
    "**/Dockerfile", "**/Dockerfile.*", "**/*.dockerfile",
    "**/Containerfile", "**/Containerfile.*",
    # Terraform
    "**/*.tf", "**/*.tfvars", "**/*.tf.json",
    # The trivy config itself, so config-only PRs still trigger a run.
    "trivy.yaml", "trivy.yml",
)


@dataclass(frozen=True)
class _ReportTarget:
    """A single `(format, on-disk path)` pair the runner will produce."""

    fmt: str
    path: Path


def _md_code_fence(content: str) -> str:
    """Return a fence longer than any backtick run in `content`."""
    longest = max((len(m) for m in re.findall(r"`+", content)), default=0)
    return "`" * max(3, longest + 1)


def _emit_non_sarif_reports(non_sarif: list[_ReportTarget]) -> None:
    """Surface non-SARIF reports in logs and step summary."""
    summary_chunks: list[str] = []
    for target in non_sarif:
        path = target.path
        if not path.is_file():
            log.warning("Non-SARIF report '%s' missing; skipping", path)
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        print(f"::group::Trivy report: {path}")
        print(content)
        print("::endgroup::")
        fence = _md_code_fence(content)
        summary_chunks.append(
            f"### Trivy report: `{path}`\n\n{fence}\n{content}\n{fence}"
        )
    if summary_chunks:
        gha_append_step_summary("\n\n".join(summary_chunks))


def _trivy_release_url(version: str) -> str:
    return (
        f"https://github.com/aquasecurity/trivy/releases/download/v{version}/"
        f"trivy_{version}_Linux-64bit.tar.gz"
    )


def _ensure_trivy() -> Path:
    """Return the path to the pinned trivy binary, downloading if needed.

    The binary is cached under `$RUNNER_TEMP` (or the system temp dir),
    keyed by version, mirroring the gitleaks installer's layout.
    Smoke-tests the binary so we fail fast on a corrupt download.
    """
    cache_root = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    install_dir = cache_root / f"trivy-{_TRIVY_VERSION}"
    binary = install_dir / "trivy"
    if binary.is_file() and os.access(binary, os.X_OK):
        log.info("Using cached trivy binary at %s", binary)
        return binary

    install_dir.mkdir(parents=True, exist_ok=True)
    url = _trivy_release_url(_TRIVY_VERSION)
    log.info("Downloading trivy v%s from %s", _TRIVY_VERSION, url)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tarball_path = Path(tmp.name)
    try:
        with (
            urlopen(Request(url), timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp,
            open(tarball_path, "wb") as out,
        ):
            shutil.copyfileobj(resp, out)
        with tarfile.open(tarball_path, mode="r:gz") as tar:
            # filter='data' rejects unsafe member metadata.
            member = tar.getmember("trivy")
            tar.extract(member, path=install_dir, filter="data")
    finally:
        tarball_path.unlink(missing_ok=True)

    if not binary.is_file():
        raise RuntimeError(
            f"trivy tarball for v{_TRIVY_VERSION} did not contain a 'trivy' file "
            f"at {binary}"
        )
    binary.chmod(0o755)

    try:
        result = subprocess.run(
            [str(binary), "--version"],
            check=True,
            capture_output=True,
            text=True,
            env=_trivy_subprocess_env(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"trivy at {binary} failed to execute after install: {exc}"
        ) from exc
    log.info(
        "Installed %s at %s",
        result.stdout.strip().splitlines()[0] if result.stdout.strip() else "trivy",
        binary,
    )
    return binary


def _trivy_subprocess_env() -> dict[str, str]:
    """Return a subprocess env with our `TRIVY_*` workflow vars stripped.

    Trivy natively reads `TRIVY_*` env vars as defaults for its own CLI
    flags (e.g. `TRIVY_SEVERITY`, `TRIVY_SCANNERS`), so leaving our
    workflow inputs in its env risks double-application or silent
    overrides. We always pass the canonical flags explicitly.
    """
    env = dict(os.environ)
    for key in list(env.keys()):
        if key.startswith("TRIVY_"):
            del env[key]
    return env


def _parse_report_formats(raw: str) -> list[_ReportTarget]:
    """Parse a comma-separated `report_formats` value into report targets.

    Whitespace is trimmed, duplicates collapse to the first occurrence,
    and unknown formats raise :class:`ValueError`.
    """
    targets: list[_ReportTarget] = []
    seen: set[str] = set()
    for raw_fmt in raw.split(","):
        fmt = raw_fmt.strip()
        if not fmt or fmt in seen:
            continue
        seen.add(fmt)
        ext = _SUPPORTED_FORMATS.get(fmt)
        if ext is None:
            raise ValueError(
                f"Invalid report_format '{fmt}' "
                f"(expected one of: {', '.join(sorted(_SUPPORTED_FORMATS))})"
            )
        targets.append(_ReportTarget(fmt=fmt, path=Path(f"trivy-report.{ext}")))
    if not targets:
        raise ValueError(
            "report_formats is empty (expected one or more of: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))})"
        )
    return targets


def _parse_scanners(raw: str) -> list[str]:
    """Parse a comma-separated `scanners` value into a canonical list.

    Whitespace is trimmed, names are lower-cased, duplicates collapse
    to the first occurrence, and unknown scanners raise
    :class:`ValueError`.
    """
    scanners: list[str] = []
    seen: set[str] = set()
    for raw_s in raw.split(","):
        s = raw_s.strip().lower()
        if not s or s in seen:
            continue
        if s not in _SUPPORTED_SCANNERS:
            raise ValueError(
                f"Invalid scanner '{s}' "
                f"(expected one or more of: {', '.join(_SUPPORTED_SCANNERS)})"
            )
        seen.add(s)
        scanners.append(s)
    if not scanners:
        raise ValueError(
            "scanners is empty (expected one or more of: "
            f"{', '.join(_SUPPORTED_SCANNERS)})"
        )
    return scanners


def _resolve_config_path() -> str:
    if not Path(_CONFIG_PATH).is_file():
        raise FileNotFoundError(
            f"trivy config not found at '{_CONFIG_PATH}'. "
            "Run from the repo root so the config is resolvable."
        )
    log.info("Using trivy config: %s", _CONFIG_PATH)
    return _CONFIG_PATH


def _diff_range(event_name: str, event: dict[str, Any]) -> tuple[str, str] | None:
    """Return `(base_sha, head_sha)` for the calling event, or `None`.

    `None` means "no diff range applicable" (workflow_dispatch,
    schedule, new ref push, missing SHAs, etc.) and instructs callers
    to fall back to a full scan.
    """
    if event_name in ("pull_request", "pull_request_target"):
        pr = event.get("pull_request") or {}
        base = ((pr.get("base") or {}).get("sha")) or ""
        head = ((pr.get("head") or {}).get("sha")) or ""
        if not base or not head:
            log.warning("PR event missing base/head SHA; falling back to full scan")
            return None
        return (base, head)
    if event_name == "push":
        before = event.get("before") or ""
        after = event.get("after") or ""
        if not before or not after:
            log.warning("Push event missing before/after SHA; falling back to full scan")
            return None
        # All-zero SHA means a new ref: nothing to diff against.
        if set(before) <= {"0"}:
            log.info("Push created a new ref; falling back to full scan")
            return None
        return (before, after)
    log.info(
        "Event '%s' has no diff range; falling back to full scan",
        event_name or "<unset>",
    )
    return None


def _is_audited_path(relpath: str) -> bool:
    """Return True if `relpath` matches an audited file pattern.

    `**/` patterns are matched both with and without the prefix so
    top-level files hit the same rules as nested ones.
    """
    norm = relpath.replace(os.sep, "/")
    for pattern in _AUDITED_PATTERNS:
        if fnmatch.fnmatchcase(norm, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(
            norm, pattern[len("**/"):]
        ):
            return True
    return False


def _determine_changed_audited_files(
    event_name: str,
    event: dict[str, Any],
    scan_path: Path,
) -> list[Path] | None:
    """Return the changed audited files inside `scan_path`, or `None`.

    Filters the git diff to `_AUDITED_PATTERNS`. The result drives the
    'changed' mode short-circuit but, unlike bandit/zizmor, the files
    themselves are NOT fed to trivy: its filesystem scanner needs the
    whole subtree to resolve transitive deps and cross-file IaC
    references, so we either scan everything or nothing.

    Semantics:

    * `None` — no usable diff range; caller should fall back to a full
      recursive scan of `scan_path`.
    * `[]` — diff range was usable but contained no audited files under
      `scan_path`; caller should treat this as a clean no-op.
    * `[paths…]` — at least one audited file changed; caller should run
      trivy against `scan_path` and log the trigger files.
    """
    diff = _diff_range(event_name, event)
    if diff is None:
        return None
    base_sha, head_sha = diff

    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", base_sha],
        check=False,
        capture_output=True,
    )

    try:
        result = subprocess.run(
            [
                "git", "diff",
                "--name-only",
                "--diff-filter=ACMR",
                f"{base_sha}..{head_sha}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        log.warning(
            "git diff %s..%s failed (rc=%s, stderr=%r); falling back to full scan",
            base_sha,
            head_sha,
            exc.returncode,
            exc.stderr.strip() if exc.stderr else "",
        )
        return None

    scan_root = scan_path.resolve()
    files: list[Path] = []
    for raw in result.stdout.splitlines():
        relpath = raw.strip()
        if not relpath or not _is_audited_path(relpath):
            continue
        candidate = Path(relpath)
        # Skip files resolving outside scan_path.
        try:
            candidate.resolve().relative_to(scan_root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        files.append(candidate)
    return files


def _run_trivy(
    binary: Path,
    user_targets: list[_ReportTarget],
    *,
    config_path: str,
    scanners: list[str],
    scan_path: Path,
) -> Path:
    """Run trivy for each user target plus an internal JSON tally pass
    if the user didn't already request one.

    Returns the path to a JSON report containing every finding; callers
    feed it to :func:`_tally_findings_by_severity` to take the
    threshold-based job-fail decision. Trivy always runs with every
    severity tier and `--exit-code 0` so reports stay complete and
    findings don't fail the run; the rc decision is decoupled from
    trivy's own exit code.

    Raises :class:`RuntimeError` for any non-zero trivy exit (an
    internal scan failure, not a finding).
    """
    severity_csv = ",".join(_SEVERITY_ORDER)
    scanners_csv = ",".join(scanners)
    base_args: list[str] = [
        str(binary),
        "fs",
        "--severity", severity_csv,
        "--scanners", scanners_csv,
        "--exit-code", "0",
        "--config", config_path,
        # Quiet the progress bar; the wrapper logs its own summary.
        "--quiet",
    ]

    # Reuse a user-requested JSON report for the tally, else add one.
    user_json = next((t for t in user_targets if t.fmt == "json"), None)
    if user_json is not None:
        tally_path = user_json.path
        runs: list[_ReportTarget] = list(user_targets)
    else:
        tally_path = Path(_INTERNAL_TALLY_PATH)
        runs = [*user_targets, _ReportTarget(fmt="json", path=tally_path)]

    subprocess_env = _trivy_subprocess_env()
    # trivy emits a single report per invocation, so we re-run per format.
    for tgt in runs:
        cmd = [
            *base_args,
            "--format", tgt.fmt,
            "--output", str(tgt.path),
            str(scan_path),
        ]
        log.info("Running: %s", " ".join(cmd))
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=subprocess_env,
            )
        except OSError as exc:
            raise RuntimeError(
                f"trivy invocation for format '{tgt.fmt}' failed to start: {exc}"
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"trivy exited unexpectedly with code {completed.returncode} "
                f"for format '{tgt.fmt}'; stderr: "
                f"{completed.stderr.strip() if completed.stderr else '<empty>'}"
            )

    return tally_path


def _tally_findings_by_severity(json_path: Path) -> dict[str, int]:
    """Read trivy's JSON output and tally findings by severity.

    Trivy's `--format json` emits a top-level object with a `Results`
    array; each result groups findings under `Vulnerabilities` /
    `Misconfigurations` / `Secrets` / `Licenses`, each carrying a
    `Severity` string. Returns a dict with at minimum the keys in
    `_SEVERITY_ORDER` (each `int >= 0`); any other `Severity` value
    (e.g. `UNKNOWN`) is preserved but doesn't participate in the
    threshold decision.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to read trivy JSON tally at {json_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"trivy JSON tally at {json_path} is not an object "
            f"(got {type(data).__name__})"
        )
    counts: dict[str, int] = {sev: 0 for sev in _SEVERITY_ORDER}
    finding_keys = ("Vulnerabilities", "Misconfigurations", "Secrets", "Licenses")
    for result in data.get("Results") or []:
        if not isinstance(result, dict):
            continue
        for key in finding_keys:
            for finding in result.get(key) or []:
                if not isinstance(finding, dict):
                    continue
                sev = str(finding.get("Severity") or "UNKNOWN").upper()
                counts[sev] = counts.get(sev, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-mode",
        default=os.environ.get("TRIVY_SCAN_MODE", "changed"),
        choices=("changed", "all"),
        help=(
            "'changed' (default) short-circuits with a no-op when no "
            "dependency-manifest / IaC / container file changed in the "
            "calling event, and otherwise scans the entire --source-dir "
            "(trivy's filesystem scanner needs the whole subtree to "
            "resolve transitive deps and cross-file IaC references, so "
            "we don't try to feed individual changed files). 'all' "
            "always scans --source-dir recursively."
        ),
    )
    p.add_argument(
        "--report-formats",
        default=os.environ.get("TRIVY_REPORT_FORMATS", "sarif"),
        help=(
            "Comma-separated list of trivy report formats. Allowed values: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))}."
        ),
    )
    p.add_argument(
        "--scanners",
        default=os.environ.get("TRIVY_SCANNERS", _DEFAULT_SCANNERS),
        help=(
            "Comma-separated list of trivy scanners to enable. Allowed "
            f"values: {', '.join(_SUPPORTED_SCANNERS)}. Default "
            f"'%(default)s' intentionally omits 'secret' because "
            "gitleaks already covers secret detection in this "
            "repository; enable it explicitly if you want both."
        ),
    )
    p.add_argument(
        "--source-dir",
        default=os.environ.get("TRIVY_SOURCE_DIR", "."),
        help=(
            "Path to scan (default %(default)s). Set to a subdirectory "
            "of the checkout to restrict the scan to that subtree. The "
            "path must exist."
        ),
    )
    p.add_argument(
        "--severity-threshold",
        default=os.environ.get(
            "TRIVY_SEVERITY_THRESHOLD", _DEFAULT_SEVERITY_THRESHOLD
        ),
        choices=_SEVERITY_CHOICES,
        help=(
            "Minimum trivy severity that fails the job. Trivy reports "
            "every finding regardless of threshold so the uploaded "
            "reports keep the full picture; after the scan the script "
            "tallies findings by severity and exits 1 only if any are "
            "at or above this threshold. Trivy categorises findings as "
            "LOW / MEDIUM / HIGH / CRITICAL. Default '%(default)s' "
            "fails on HIGH or CRITICAL findings; set 'low' to fail on "
            "any finding."
        ),
    )
    return p


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        targets = _parse_report_formats(args.report_formats)
        scanners = _parse_scanners(args.scanners)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    source_dir = Path(args.source_dir)
    if not source_dir.is_dir():
        log.error(
            "scan path '%s' does not exist or is not a directory "
            "(did the checkout step fetch it?)",
            source_dir,
        )
        return 2

    try:
        config_path = _resolve_config_path()
        event = gha_load_github_event()
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        log.error("%s", exc)
        return 2

    if args.scan_mode == "all":
        files: list[Path] | None = None
    else:
        files = _determine_changed_audited_files(
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
            event=event,
            scan_path=source_dir,
        )

    if files is None:
        log.info(
            "Trivy scope: recursive scan of %s (no usable diff range)",
            source_dir,
        )
    elif files:
        log.info(
            "Trivy scope: recursive scan of %s "
            "(%d audited file(s) changed in the event)",
            source_dir,
            len(files),
        )
        for f in files:
            log.info("  - %s", f)
    else:
        log.info(
            "Trivy scope: no audited files changed under %s; nothing to scan",
            source_dir,
        )

    log.info(
        "Trivy formats: %s",
        ", ".join(f"{t.fmt}->{t.path}" for t in targets),
    )
    log.info("Trivy scanners: %s", ", ".join(scanners))
    log.info(
        "Trivy fail threshold: severity >= %s (lower-severity findings "
        "still appear in reports)",
        args.severity_threshold.upper(),
    )

    sarif_target = next((t for t in targets if t.fmt == "sarif"), None)
    non_sarif = [t for t in targets if t.fmt != "sarif"]

    # 'changed' mode with nothing to scan: emit empty paths so uploads skip.
    if files is not None and not files:
        gha_set_output({"sarif_path": "", "non_sarif_paths": ""})
        return 0

    # Emit outputs up-front so uploads run even if trivy fails partway.
    gha_set_output(
        {
            "sarif_path": "" if sarif_target is None else str(sarif_target.path),
            "non_sarif_paths": "\n".join(str(t.path) for t in non_sarif),
        }
    )

    internal_tally_used = not any(t.fmt == "json" for t in targets)
    try:
        binary = _ensure_trivy()
        tally_path = _run_trivy(
            binary,
            targets,
            config_path=config_path,
            scanners=scanners,
            scan_path=source_dir,
        )
        counts = _tally_findings_by_severity(tally_path)
    except RuntimeError as exc:
        log.error("%s", exc)
        _emit_non_sarif_reports(non_sarif)
        return 2
    finally:
        if internal_tally_used:
            Path(_INTERNAL_TALLY_PATH).unlink(missing_ok=True)

    threshold = args.severity_threshold.upper()
    threshold_idx = _SEVERITY_ORDER.index(threshold)
    failing = sum(counts[sev] for sev in _SEVERITY_ORDER[threshold_idx:])
    summary = ", ".join(f"{sev}={counts[sev]}" for sev in _SEVERITY_ORDER)
    extra = {
        sev: n
        for sev, n in counts.items()
        if sev not in _SEVERITY_ORDER and n
    }
    if extra:
        summary += ", " + ", ".join(f"{sev}={n}" for sev, n in extra.items())
    log.info("Trivy findings: %s", summary)

    _emit_non_sarif_reports(non_sarif)

    if failing:
        log.error(
            "trivy reported %d finding(s) at severity >= %s; failing the job "
            "(see report artifacts for the full set, including lower-severity "
            "findings that didn't trigger the threshold)",
            failing,
            threshold,
        )
        return 1
    log.info(
        "All findings below threshold '%s'; passing (lower-severity findings "
        "are still listed in the uploaded reports)",
        threshold,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

