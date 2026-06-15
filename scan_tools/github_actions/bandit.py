#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Run bandit for the current checkout.

- Install the pinned `bandit[sarif]` release and verify it.
- Require `bandit.yaml` at the repo root (hard error if missing).
- Derive change sets from the GitHub event for changed/all scans; run per
  requested format and emit SARIF/non-SARIF paths plus severity tally.

Exit codes: 0 clean/empty changed set; 1 findings at/above threshold or
bad formats; 2 input/config/runtime errors.

Inputs come from CLI flags with `BANDIT_*` env var defaults.
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    "csv": "csv",
    "html": "html",
    "xml": "xml",
    "yaml": "yaml",
    "txt": "txt",
}
_BANDIT_VERSION = "1.9.4"
_CONFIG_PATH = "bandit.yaml"
# Bandit exits with this code when it finds any issue at/above the
# --severity-level; we always scan at low, so it just means "any finding".
_FINDING_EXIT_CODE = 1
# Ascending severity order; threshold comparisons rely on it.
_SEVERITY_ORDER: tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
_SEVERITY_CHOICES: tuple[str, ...] = tuple(s.lower() for s in _SEVERITY_ORDER)
_DEFAULT_SEVERITY_THRESHOLD = "high"
# Internal JSON tally pass output; cleaned up before returning.
_INTERNAL_TALLY_PATH = "bandit-tally.json"
# Extensions bandit treats as Python source; mirrors its --recursive walker.
_PYTHON_EXTENSIONS: tuple[str, ...] = (".py", ".pyw")
# Map bandit severities to GitHub security-severity values.
_BANDIT_SECURITY_SEVERITY: dict[str, str] = {
    "HIGH": "8.5",
    "MEDIUM": "5.0",
    "LOW": "1.0",
}


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
        print(f"::group::Bandit report: {path}")
        print(content)
        print("::endgroup::")
        fence = _md_code_fence(content)
        summary_chunks.append(
            f"### Bandit report: `{path}`\n\n{fence}\n{content}\n{fence}"
        )
    if summary_chunks:
        gha_append_step_summary("\n\n".join(summary_chunks))


def _ensure_bandit() -> Path:
    """Install the pinned bandit release (with SARIF) and return its CLI path.

    Always re-runs `pip install` so the version pin is enforced even
    when an older bandit is already on PATH; pip is fast when the
    requested version is already installed. Smoke-tests the binary
    afterwards so we fail fast if the install left a half-broken state.
    """
    spec = f"bandit[sarif]=={_BANDIT_VERSION}"
    log.info("Installing %s", spec)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", spec],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to install {spec}: {exc}") from exc

    # Console scripts land next to sys.executable, even for an unactivated venv.
    binary_path = Path(sys.executable).parent / "bandit"
    if not binary_path.is_file():
        found = shutil.which("bandit")
        if found is None:
            raise RuntimeError(
                f"bandit CLI not found at {binary_path} or on PATH after "
                f"installing {spec}; is the active Python environment writable?"
            )
        binary_path = Path(found)

    try:
        result = subprocess.run(
            [str(binary_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"bandit at {binary_path} failed to execute after install: {exc}"
        ) from exc
    log.info(
        "Installed %s at %s",
        result.stdout.strip().splitlines()[0] if result.stdout.strip() else "bandit",
        binary_path,
    )
    return binary_path


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
        targets.append(_ReportTarget(fmt=fmt, path=Path(f"bandit-report.{ext}")))
    if not targets:
        raise ValueError(
            "report_formats is empty (expected one or more of: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))})"
        )
    return targets


def _resolve_config_path() -> str:
    if not Path(_CONFIG_PATH).is_file():
        raise FileNotFoundError(
            f"bandit config not found at '{_CONFIG_PATH}'. "
            "Run from the repo root so the config is resolvable."
        )
    log.info("Using bandit config: %s", _CONFIG_PATH)
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


def _determine_changed_py_files(
    event_name: str,
    event: dict[str, Any],
    scan_path: Path,
) -> list[Path] | None:
    """Return the changed Python files inside `scan_path`, or `None`.

    Filters the git diff to `_PYTHON_EXTENSIONS` so both scan modes look
    at the same set of files.

    Semantics:

    * `None` — no usable diff range; caller should fall back to a full
      recursive scan of `scan_path`.
    * `[]` — diff range was usable but contained no Python files under
      `scan_path`; caller should treat this as a clean no-op.
    * `[paths…]` — exact set of files for bandit to scan.
    """
    diff = _diff_range(event_name, event)
    if diff is None:
        return None
    base_sha, head_sha = diff

    # Best-effort fetch so the base SHA is reachable for the diff.
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
        if not relpath.endswith(_PYTHON_EXTENSIONS):
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


def _run_bandit(
    binary: Path,
    user_targets: list[_ReportTarget],
    *,
    config_path: str,
    files: list[Path] | None,
    scan_path: Path,
) -> Path:
    """Run bandit at `--severity-level low` for each user target plus an
    internal JSON tally pass if the user didn't already request one.

    Returns the path to a JSON report containing every finding; callers
    feed it to :func:`_tally_findings_by_severity` to take the
    threshold-based job-fail decision. Bandit always scans at LOW so the
    reports keep every finding regardless of the configured threshold.

    Raises :class:`RuntimeError` for unexpected bandit exit codes
    (anything outside `{0, _FINDING_EXIT_CODE}`).
    """
    base_args: list[str] = [str(binary), "--severity-level", "low"]
    base_args.extend(["--configfile", config_path])
    if files is None:
        base_args.extend(["--recursive", str(scan_path)])
    else:
        base_args.extend(str(p) for p in files)

    # Reuse a user-requested JSON report for the tally, else add one.
    user_json = next((t for t in user_targets if t.fmt == "json"), None)
    if user_json is not None:
        tally_path = user_json.path
        runs: list[_ReportTarget] = list(user_targets)
    else:
        tally_path = Path(_INTERNAL_TALLY_PATH)
        runs = [*user_targets, _ReportTarget(fmt="json", path=tally_path)]

    # bandit emits a single report per invocation, so we re-run per format.
    for tgt in runs:
        cmd = [*base_args, "--format", tgt.fmt, "--output", str(tgt.path)]
        log.info("Running: %s", " ".join(cmd))
        rc = subprocess.run(cmd, check=False).returncode
        if rc in (0, _FINDING_EXIT_CODE):
            continue
        raise RuntimeError(
            f"bandit exited unexpectedly with code {rc} for format '{tgt.fmt}'"
        )

    # Enrich SARIF severity for the Security tab (tally pass is never SARIF).
    for tgt in user_targets:
        if tgt.fmt == "sarif" and tgt.path.is_file():
            _enrich_sarif_with_security_severity(tgt.path)

    return tally_path


def _enrich_sarif_with_security_severity(sarif_path: Path) -> None:
    """Inject `security-severity` into each SARIF result so the GitHub
    Security tab tiers bandit findings the same way it tiers CodeQL.

    Maps bandit's per-result `properties.issue_severity` through
    `_BANDIT_SECURITY_SEVERITY`. Pre-existing values are preserved.
    Enrichment failures are logged at WARNING and don't propagate:
    `level` still drives the Security tab tier on its own.
    """
    try:
        with open(sarif_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("SARIF severity enrichment skipped (%s): %s", sarif_path, exc)
        return
    if not isinstance(data, dict):
        log.warning(
            "SARIF severity enrichment skipped: %s is not a JSON object",
            sarif_path,
        )
        return

    enriched = 0
    preserved = 0
    unknown = 0
    for run in data.get("runs") or []:
        if not isinstance(run, dict):
            continue
        for result in run.get("results") or []:
            if not isinstance(result, dict):
                continue
            props = result.get("properties")
            if not isinstance(props, dict):
                props = {}
                result["properties"] = props
            if props.get("security-severity") is not None:
                preserved += 1
                continue
            sev = str(props.get("issue_severity") or "").upper()
            score = _BANDIT_SECURITY_SEVERITY.get(sev)
            if score is None:
                unknown += 1
                continue
            props["security-severity"] = score
            enriched += 1

    if enriched == 0:
        log.debug(
            "SARIF severity enrichment: nothing to add (%d preserved, %d unknown) in %s",
            preserved,
            unknown,
            sarif_path,
        )
        return

    try:
        with open(sarif_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        log.warning("Failed to write enriched SARIF to %s: %s", sarif_path, exc)
        return

    log.info(
        "SARIF severity enrichment: %d result(s) given security-severity "
        "(%d already had it, %d had unknown severity) in %s",
        enriched,
        preserved,
        unknown,
        sarif_path,
    )


def _tally_findings_by_severity(json_path: Path) -> dict[str, int]:
    """Returns a dict with at minimum `LOW`/`MEDIUM`/`HIGH` keys (each
    `int >= 0`); any other `issue_severity` value bandit emits (e.g.
    `UNDEFINED`) is preserved verbatim but doesn't participate in the
    threshold decision.
    """
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Failed to read bandit JSON tally at {json_path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"bandit JSON tally at {json_path} is not an object "
            f"(got {type(data).__name__})"
        )
    counts: dict[str, int] = {sev: 0 for sev in _SEVERITY_ORDER}
    for issue in data.get("results") or []:
        sev = str(issue.get("issue_severity") or "UNDEFINED").upper()
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scan-mode",
        default=os.environ.get("BANDIT_SCAN_MODE", "changed"),
        choices=("changed", "all"),
        help=(
            "'changed' (default) scans only Python source files modified "
            "by the calling event (PR commits or push range). 'all' "
            "recursively scans every Python source file under "
            "--source-dir; bandit's own --recursive walker filters to "
            ".py / .pyw automatically so non-Python files are skipped "
            "regardless of mode."
        ),
    )
    p.add_argument(
        "--report-formats",
        default=os.environ.get("BANDIT_REPORT_FORMATS", "sarif"),
        help=(
            "Comma-separated list of bandit report formats. Allowed values: "
            f"{', '.join(sorted(_SUPPORTED_FORMATS))}."
        ),
    )
    p.add_argument(
        "--source-dir",
        default=os.environ.get("BANDIT_SOURCE_DIR", "."),
        help=(
            "Path to scan (default %(default)s). Set to a subdirectory of "
            "the checkout to restrict the scan to that subtree; the "
            "'changed' scan mode further restricts to only the Python "
            "source files modified inside this path. The path must exist."
        ),
    )
    p.add_argument(
        "--severity-threshold",
        default=os.environ.get(
            "BANDIT_SEVERITY_THRESHOLD", _DEFAULT_SEVERITY_THRESHOLD
        ),
        choices=_SEVERITY_CHOICES,
        help=(
            "Minimum bandit severity that fails the job. Bandit always "
            "scans at LOW so the uploaded reports keep every finding; "
            "after the scan the script tallies findings by severity and "
            "exits 1 only if any are at or above this threshold. Bandit "
            "categorises findings as LOW / MEDIUM / HIGH (no 'critical' "
            "tier). Default '%(default)s' fails only on HIGH findings; "
            "set 'low' to fail on any finding (the legacy behaviour)."
        ),
    )
    return p


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")
    args = build_parser().parse_args(argv)

    try:
        targets = _parse_report_formats(args.report_formats)
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
        files = _determine_changed_py_files(
            event_name=os.environ.get("GITHUB_EVENT_NAME", ""),
            event=event,
            scan_path=source_dir,
        )

    if files is None:
        log.info(
            "Bandit scope: recursive scan of %s (bandit filters to %s itself)",
            source_dir,
            "/".join(_PYTHON_EXTENSIONS),
        )
    elif files:
        log.info(
            "Bandit scope: %d changed Python source file(s) under %s",
            len(files),
            source_dir,
        )
        for f in files:
            log.info("  - %s", f)
    else:
        log.info(
            "Bandit scope: no Python source files changed under %s; nothing to scan",
            source_dir,
        )

    log.info(
        "Bandit formats: %s",
        ", ".join(f"{t.fmt}->{t.path}" for t in targets),
    )
    log.info(
        "Bandit fail threshold: severity >= %s (lower-severity findings "
        "still appear in reports)",
        args.severity_threshold.upper(),
    )

    sarif_target = next((t for t in targets if t.fmt == "sarif"), None)
    non_sarif = [t for t in targets if t.fmt != "sarif"]

    # 'changed' mode with nothing to scan: emit empty paths so uploads skip.
    if files is not None and not files:
        gha_set_output({"sarif_path": "", "non_sarif_paths": ""})
        return 0

    # Emit outputs up-front so uploads run even if bandit fails partway.
    gha_set_output(
        {
            "sarif_path": "" if sarif_target is None else str(sarif_target.path),
            "non_sarif_paths": "\n".join(str(t.path) for t in non_sarif),
        }
    )

    internal_tally_used = not any(t.fmt == "json" for t in targets)
    try:
        binary = _ensure_bandit()
        tally_path = _run_bandit(
            binary,
            targets,
            config_path=config_path,
            files=files,
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
    log.info("Bandit findings: %s", summary)

    _emit_non_sarif_reports(non_sarif)

    if failing:
        log.error(
            "bandit reported %d finding(s) at severity >= %s; failing the job "
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
