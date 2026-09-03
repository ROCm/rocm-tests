# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Polls the GitHub runner API for persistent (non-ARC) runner availability and
# updates the target matrix accordingly. ARC (ephemeral) runners bypass this
# check; queue_guard.py handles their post-dispatch timeout.
#
# Environment variables (set by the workflow step):
#   GITHUB_TOKEN       — GitHub App token with runner:read scope
#   GITHUB_REPOSITORY  — "owner/repo" (injected automatically by Actions)
#   TARGETS_JSON       — JSON array of target dicts from setup_matrix
#   MAX_WAIT_SECS      — total polling budget in seconds (default: 120)
#   POLL_INTERVAL_SECS — seconds between polls (default: 15)
#   GITHUB_OUTPUT      — path to the GitHub Actions output file

import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

token = os.environ["GITHUB_TOKEN"]
repo = os.environ["GITHUB_REPOSITORY"]  # "owner/repo"
org = repo.split("/")[0]
targets = json.loads(os.environ["TARGETS_JSON"])
max_wait = int(os.environ.get("MAX_WAIT_SECS", "120"))
interval = int(os.environ.get("POLL_INTERVAL_SECS", "15"))


def is_arc(label: str) -> bool:
    """ARC (Actions Runner Controller) runners are ephemeral Kubernetes pods.
    They register only when a job is dispatched and deregister on completion,
    so they are never visible in the runner API when idle. Detect them by the
    'ossci' substring present in all ARC scale-set labels in this repo.
    queue_guard.py handles the post-dispatch timeout for these.
    """
    return "ossci" in label.lower()


def get_registered_labels(wanted: set[str]) -> set[str]:
    """Return the subset of *wanted* label names that have at least one
    registered runner (online or in_use). Stops paginating an endpoint as
    soon as all wanted labels are found — avoids fetching the full fleet
    when only a handful of persistent runners need checking.
    Tries repo-level then org-level endpoints.
    """
    found = set()
    endpoints = [
        f"https://api.github.com/repos/{repo}/actions/runners",
        f"https://api.github.com/orgs/{org}/actions/runners",
    ]
    for base_url in endpoints:
        if found >= wanted:
            break  # all labels already located — skip remaining endpoint
        page = 1
        while True:
            url = f"{base_url}?per_page=100&page={page}"
            req = Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
            try:
                with urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
            except HTTPError as exc:
                level = "repo" if "repos" in base_url else "org"
                print(f"  [{level}] runners API: HTTP {exc.code} — skipping")
                break
            except URLError as exc:
                print(f"  runners API network error ({exc.reason}) — skipping")
                break

            runners = data.get("runners", [])
            if not runners:
                break
            for r in runners:
                if r.get("status") in ("online", "in_use"):
                    for lbl in r.get("labels", []):
                        if lbl["name"] in wanted:
                            found.add(lbl["name"])
            if len(runners) < 100 or found >= wanted:
                break
            page += 1
    return found


# ── Classify targets ──────────────────────────────────────────────────────────
# target_available=false  → final skip (testplan.ini decision)
# ARC runner (ossci label) → trust target_available=true; queue_guard handles post-dispatch
# persistent runner        → poll the runner API for online/in_use status
resolved: dict[str, bool] = {}  # platform name → True (run) | False (skip)
to_poll: list[dict] = []  # persistent runners needing a live API check

for t in targets:
    label = t.get("runs_on", "")
    if t.get("target_available", "true").strip().lower() == "false":
        resolved[t["name"]] = False
        print(f"  {t['name']}: testplan.ini target_available=false — skip")
    elif is_arc(label):
        resolved[t["name"]] = True
        print(f"  {t['name']} ({label}): ARC ephemeral runner — bypassing poll, queue_guard handles dispatch timeout")
    elif not label:
        resolved[t["name"]] = True
        print(f"  {t['name']}: no runs_on label configured — passing through")
    else:
        to_poll.append(t)

# ── Poll persistent runners until resolved or deadline reached ────────────────
if not to_poll:
    print("\nNo persistent runners to poll — done.")
else:
    wanted = {t["runs_on"] for t in to_poll}  # labels we actually care about
    deadline = time.monotonic() + max_wait
    attempt = 0
    print(f"\nPolling {len(to_poll)} persistent runner(s) (budget: {max_wait}s, interval: {interval}s) …\n")

    while to_poll:
        attempt += 1
        registered = get_registered_labels(wanted)
        print(f"[poll #{attempt}]")

        still_pending = []
        for t in to_poll:
            label = t["runs_on"]
            if label in registered:
                resolved[t["name"]] = True
                wanted.discard(label)
                print(f"  ✓ {t['name']} ({label}): online")
            else:
                print(f"  … {t['name']} ({label}): not yet registered")
                still_pending.append(t)
        to_poll = still_pending

        if not to_poll:
            print("\nAll persistent runners resolved — proceeding immediately.")
            break

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            for t in to_poll:
                resolved[t["name"]] = False
                print(f"  ✗ {t['name']} ({t['runs_on']}): not registered after {max_wait}s — will skip")
            to_poll = []
            break

        sleep_secs = min(interval, remaining)
        names_left = [t["name"] for t in to_poll]
        print(f"  Still pending: {names_left} — next check in {sleep_secs:.0f}s ({remaining:.0f}s budget remaining)\n")
        time.sleep(sleep_secs)

# ── Build updated targets matrix and skipped list ────────────────────────────
# resolved contains every platform that was explicitly classified above;
# targets not reached by the API check (no runs_on label) default to available.
print("\n── Runner availability summary ──")
updated_targets = []
skipped_names = []

for t in targets:
    avail = resolved.get(t["name"], True)
    status = "✓ will run" if avail else "✗ skipped"
    print(f"  {t['name']}: {status}")
    updated_targets.append({**t, "target_available": str(avail).lower()})
    if not avail:
        skipped_names.append(t["name"])

gh_output = os.environ.get("GITHUB_OUTPUT", "")
if gh_output:
    with open(gh_output, "a") as fh:
        fh.write(f"targets={json.dumps(updated_targets)}\n")
        fh.write(f"skipped_targets={json.dumps(skipped_names)}\n")
