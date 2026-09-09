# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Resolves a rockrel run_id from a workflow filename (or passes through a
# numeric run_id directly).
#
# Environment variables (set by the workflow step):
#   GITHUB_TOKEN    — GitHub API token
#   SOURCE_REPO     — e.g. "ROCm/rockrel"
#   ARTIFACT_SOURCE — workflow filename (e.g. "multi_arch_release.yml") OR
#                     an all-numeric run id to pin
#   GITHUB_OUTPUT   — path to the GitHub Actions output file

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

repo = os.environ["SOURCE_REPO"]
source = os.environ["ARTIFACT_SOURCE"].strip()

if source.isdigit():
    run_id = source
    print(f"artifact_source is a run id - pinning {repo} run {run_id}")
else:
    url = (
        f"https://api.github.com/repos/{repo}/actions/workflows/"
        f"{source}/runs?status=success&event=schedule&branch=main&per_page=1"
    )
    req = Request(
        url,
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
    except HTTPError as exc:
        sys.exit(
            f"GitHub API returned {exc.code} for {url}\n"
            f'artifact_source must be a workflow filename in {repo} (e.g. "multi_arch_release.yml")'
            " or an all-numeric run id."
        )
    except URLError as exc:
        sys.exit(f"Could not reach the GitHub API for {url}: {exc.reason}")
    runs = data.get("workflow_runs") or []
    if not runs:
        sys.exit(f"No successful scheduled run on main for {repo}/{source}")
    run_id = runs[0]["id"]
    print(f"Latest successful run of {repo}/{source}: {run_id}")

gh_output = os.environ.get("GITHUB_OUTPUT", "")
if gh_output:
    with open(gh_output, "a") as f:
        f.write(f"run_id={run_id}\n")
