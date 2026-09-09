# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Secondary safety net: checks that a runner was actually acquired within the
# allowed queue window. check_runners.py in e2e-nightly.yml already vetted
# runner availability before dispatch; this catches the rare edge case where a
# runner went offline between the pre-check and job pickup.
#
# Outputs (written to GITHUB_OUTPUT):
#   proceed     — "true" if the runner was acquired in time, "false" otherwise
#   skip_reason — human-readable message when proceed=false
#
# Environment variables (set by the workflow step):
#   WORKFLOW_QUEUED_AT  — ISO-8601 timestamp of workflow trigger (github.run_started_at)
#   RUNNER_NAME         — runner label for log context
#   MAX_QUEUE_WAIT_SECS — timeout in seconds (default: 900 = 15 min)
#   GITHUB_OUTPUT       — path to the GitHub Actions output file

from datetime import datetime, timezone
import os

queued_at = os.environ.get("WORKFLOW_QUEUED_AT", "").strip()
runner = os.environ.get("RUNNER_NAME", "")
max_wait = int(os.environ.get("MAX_QUEUE_WAIT_SECS", "900"))
gh_output = os.environ.get("GITHUB_OUTPUT", "")


def _set_outputs(**pairs: str) -> None:
    if gh_output:
        with open(gh_output, "a") as f:
            for key, value in pairs.items():
                f.write(f"{key}={value}\n")


if not queued_at:
    print(f"Runner queue guard: WORKFLOW_QUEUED_AT not set — skipping wait check for {runner!r}")
    _set_outputs(proceed="true")
else:
    triggered = datetime.fromisoformat(queued_at.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    waited = (now - triggered).total_seconds()
    print(f"Runner queue guard: runner={runner!r} waited={waited:.0f}s limit={max_wait}s", flush=True)

    if waited > max_wait:
        print(
            f"::warning::Runner {runner!r} not acquired within {max_wait // 60}min"
            f" (waited {waited:.0f}s) — platform skipped",
            flush=True,
        )
        _set_outputs(
            proceed="false",
            skip_reason=f"runner not acquired within {max_wait // 60}min (waited {waited:.0f}s)",
        )
    else:
        print(f"Runner acquired within limit ({waited:.0f}s ≤ {max_wait}s) — proceeding", flush=True)
        _set_outputs(proceed="true")
