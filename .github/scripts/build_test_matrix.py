# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Parses testplan.ini and emits the test matrix JSON to GITHUB_OUTPUT.
#
# Output variables written to GITHUB_OUTPUT:
#   targets         — JSON array of all target dicts (one per [section])
#   skipped_targets — JSON array of section names whose target_available=false
#
# Usage:
#   python3 .github/scripts/build_test_matrix.py
#
# The script expects testplan.ini to be present in the current working directory
# (i.e. the repository root, which is the default for GitHub Actions steps).

import configparser
import json
import os

conf = configparser.ConfigParser()
conf.read("testplan.ini")

all_targets: list[dict] = []
skipped: list[str] = []

for section in conf.sections():
    entry = {"name": section, **dict(conf.items(section))}
    all_targets.append(entry)
    if entry.get("target_available", "true").strip().lower() == "false":
        skipped.append(section)

print(f"all targets: {all_targets!r}")
print(f"skipped:     {skipped!r}")

gh_output = os.environ["GITHUB_OUTPUT"]
with open(gh_output, "a") as f:
    f.write(f"targets={json.dumps(all_targets)}\n")
    f.write(f"skipped_targets={json.dumps(skipped)}\n")
