# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Writes a Markdown test-result table to $GITHUB_STEP_SUMMARY.
#
# Two modes:
#
#   platform  — single-platform summary written from inside e2e-tests.yml.
#               Reads test-counts/counts.json, appends md_summary.md in a
#               <details> block, and optionally adds a download link.
#
#   aggregate — cross-platform summary written from inside e2e-nightly.yml.
#               Reads all counts.json files under a directory tree and
#               renders one row per platform plus a TOTAL footer.
#
# Environment variables:
#
#   MODE              — "platform" (default) or "aggregate"
#   GITHUB_STEP_SUMMARY — path to the step-summary file (injected by Actions)
#
#   platform mode:
#     PLATFORM_NAME   — display name of the platform (e.g. "linux-gfx94x")
#     COUNTS_FILE     — path to counts.json (default: test-counts/counts.json)
#     MD_SUMMARY_FILE — path to the markdown report (default: md_summary.md)
#     ARTIFACT_URL    — (optional) URL for the "Download test report" link
#
#   aggregate mode:
#     RUN_ID          — rockrel run id or artifact tag shown in the heading
#     COUNTS_DIR      — directory tree to search for counts.json files
#                       (default: all-counts)
#     SKIPPED_TARGETS — JSON array of platform names skipped by check_runners
#                       (default: "[]")

from __future__ import annotations

import json
import os
import pathlib
import sys


def _markdown_table_row(cells: list[str], *, bold: bool = False) -> str:
    if bold:
        cells = [f"**{c}**" for c in cells]
    return "| " + " | ".join(cells) + " |"


def _table_header(has_skip: bool) -> list[str]:
    if has_skip:
        return [
            "| Platform | Tests Run | Pass | Skip | Fail | Error | Status |",
            "|---|--:|--:|--:|--:|--:|---|",
        ]
    return [
        "| Platform | Tests Run | Pass | Fail | Error | Status |",
        "|---|--:|--:|--:|--:|---|",
    ]


def _result_row(r: dict, *, has_skip: bool) -> str:
    ok = r["failed"] == 0 and r["error"] == 0 and r.get("total", 0) > 0
    status = "✅ Passed" if ok else "❌ Failed"
    if has_skip:
        return _markdown_table_row(
            [
                r["platform"],
                str(r["total"]),
                str(r["passed"]),
                str(r.get("skipped", 0)),
                str(r["failed"]),
                str(r["error"]),
                status,
            ]
        )
    return _markdown_table_row(
        [
            r["platform"],
            str(r["total"]),
            str(r["passed"]),
            str(r["failed"]),
            str(r["error"]),
            status,
        ]
    )


def _timeout_row(r: dict, *, has_skip: bool) -> str:
    dash = "—"
    status = "⏭ Runner timeout"
    if has_skip:
        return _markdown_table_row([r["platform"], dash, dash, dash, dash, dash, status])
    return _markdown_table_row([r["platform"], dash, dash, dash, dash, status])


def _skipped_platform_row(name: str, *, has_skip: bool) -> str:
    dash = "—"
    status = "⏭ Skipped — platform not available"
    if has_skip:
        return _markdown_table_row([name, dash, dash, dash, dash, dash, status])
    return _markdown_table_row([name, dash, dash, dash, dash, status])


def _total_row(
    total_run: int, total_pass: int, total_skip: int, total_fail: int, total_err: int, *, has_skip: bool
) -> str:
    # Append "| |" directly (no space inside the empty status cell) to match the
    # original f-string format: "... | **{total_err}** | |".
    # Passing "" through _markdown_table_row would produce "|  |" (space-padded).
    if has_skip:
        cells = [
            "**TOTAL**",
            f"**{total_run}**",
            f"**{total_pass}**",
            f"**{total_skip}**",
            f"**{total_fail}**",
            f"**{total_err}**",
        ]
    else:
        cells = ["**TOTAL**", f"**{total_run}**", f"**{total_pass}**", f"**{total_fail}**", f"**{total_err}**"]
    return "| " + " | ".join(cells) + " | |"


# ─── Platform mode ────────────────────────────────────────────────────────────


def write_platform_summary() -> None:
    platform = os.environ.get("PLATFORM_NAME", "unknown")
    counts_path = pathlib.Path(os.environ.get("COUNTS_FILE", "test-counts/counts.json"))
    md_path = pathlib.Path(os.environ.get("MD_SUMMARY_FILE", "md_summary.md"))
    artifact_url = os.environ.get("ARTIFACT_URL", "").strip()
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")

    if not counts_path.exists():
        print(f"counts.json not found at {counts_path} — skipping step summary")
        sys.exit(0)

    r = json.loads(counts_path.read_text())
    platform = r.get("platform", platform)
    has_skip = r.get("skipped", 0) > 0

    lines: list[str] = []
    lines.append(f"## Test summary - {platform}")
    lines.append("")
    lines.extend(_table_header(has_skip))
    lines.append(_result_row(r, has_skip=has_skip))
    lines.append("")
    lines.append("---")
    lines.append("")

    if md_path.exists():
        lines.append(f"<details open><summary>Full Test Report for {platform}</summary>")
        lines.append("")
        lines.append(md_path.read_text())
        lines.append("")
        lines.append("</details>")
    else:
        lines.append("_No summary file generated._")

    # Two blank lines before the download link — matches the original output where
    # the Python summary step wrote one trailing blank and the subsequent bash step
    # opened with `echo "" >> $GITHUB_STEP_SUMMARY` adding a second.
    lines.append("")
    lines.append("")

    if artifact_url:
        lines.append(f"### ⬇ [Download test report — {platform}]({artifact_url})")
    else:
        lines.append(f"### ⬇ Download test report — {platform}")
    lines.append("")
    lines.append("> Zip archive — extract `nightly_report.html` and open locally.")

    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


# ─── Aggregate mode ───────────────────────────────────────────────────────────


def write_aggregate_summary() -> None:
    run_id = os.environ.get("RUN_ID", "unknown")
    counts_dir = pathlib.Path(os.environ.get("COUNTS_DIR", "all-counts"))
    raw = os.environ.get("SKIPPED_TARGETS", "").strip()
    skipped_names: list[str] = json.loads(raw) if raw else []
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")

    rows: list[dict] = []
    for counts_file in sorted(counts_dir.rglob("counts.json")):
        try:
            rows.append(json.loads(counts_file.read_text()))
        except Exception as exc:
            print(f"Skipping {counts_file}: {exc}")

    has_skip = any(r.get("skipped", 0) > 0 for r in rows)

    lines: list[str] = []
    lines.append(f"## Nightly Results — {run_id} — All Platforms")
    lines.append("")
    lines.extend(_table_header(has_skip))

    total_run = total_pass = total_skip = total_fail = total_err = 0

    for r in rows:
        if r.get("runner_timeout"):
            lines.append(_timeout_row(r, has_skip=has_skip))
        else:
            lines.append(_result_row(r, has_skip=has_skip))
            total_run += r["total"]
            total_pass += r["passed"]
            total_skip += r.get("skipped", 0)
            total_fail += r["failed"]
            total_err += r["error"]

    for name in skipped_names:
        lines.append(_skipped_platform_row(name, has_skip=has_skip))

    if rows:
        lines.append(_total_row(total_run, total_pass, total_skip, total_fail, total_err, has_skip=has_skip))
    else:
        lines.append("_No platform results collected._")

    lines.append("")

    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n".join(lines))


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mode = os.environ.get("MODE", "platform").strip().lower()
    if mode == "aggregate":
        write_aggregate_summary()
    elif mode == "platform":
        write_platform_summary()
    else:
        sys.exit(f"Unknown MODE={mode!r}. Expected 'platform' or 'aggregate'.")
