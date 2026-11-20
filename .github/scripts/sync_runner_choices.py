# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate the manual E2E runner dropdown from enabled testplan.ini entries.

Run from any directory with Python. Use --check to preview the choices and
detect stale workflow options without writing files. This does not query live
runner status or change the reusable workflow_call inputs.
"""

import argparse
import configparser
import json
from pathlib import Path

BEGIN = "        # BEGIN GENERATED RUNNER OPTIONS"
END = "        # END GENERATED RUNNER OPTIONS"
ROOT = Path(__file__).resolve().parents[2]


def runner_labels(testplan: Path) -> list[str]:
    """Return unique, enabled runner labels in testplan order."""
    config = configparser.ConfigParser(interpolation=None)
    with testplan.open(encoding="utf-8") as source:
        config.read_file(source)

    labels: list[str] = []
    for section in config.sections():
        entry = config[section]
        if not entry.getboolean("target_available", fallback=False):
            continue
        label = entry["runs_on"].strip()
        if not label:
            raise ValueError(f"{section}: enabled target has an empty runs_on label")
        if label not in labels:
            labels.append(label)
    if not labels:
        raise ValueError("testplan.ini has no enabled runner labels")
    return labels


def update_options(workflow: str, labels: list[str]) -> str:
    """Replace only the explicitly marked runner options block."""
    if workflow.count(BEGIN) != 1 or workflow.count(END) != 1:
        raise ValueError("Expected exactly one generated runner options block")
    before, _, remainder = workflow.partition(BEGIN)
    if END not in remainder:
        raise ValueError("Runner options block markers are out of order")
    _, _, after = remainder.partition(END)
    options = "\n".join(f"          - {json.dumps(label)}" for label in labels)
    return f"{before}{BEGIN}\n        options:\n{options}\n{END}{after}"


def main() -> int:
    """Print the dropdown choices and update or check the workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check for stale options without modifying files")
    args = parser.parse_args()

    workflow = ROOT / ".github" / "workflows" / "e2e-tests.yml"
    labels = runner_labels(ROOT / "testplan.ini")
    original = workflow.read_text(encoding="utf-8")
    updated = update_options(original, labels)

    print("E2E Tests > Run workflow > runs_on")
    for index, label in enumerate(labels, start=1):
        print(f"  {index}. {label}")
    print("Configured runners only; live availability is not checked.")
    if original == updated:
        print("Dropdown options are up to date.")
    elif args.check:
        print("ERROR: Dropdown options are stale. Run sync_runner_choices.py without --check.")
        return 1
    else:
        workflow.write_text(updated, encoding="utf-8")
        print(f"Updated {workflow.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
