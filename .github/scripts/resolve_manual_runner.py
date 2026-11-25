# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve manual E2E inputs using the selected runner's test-plan entry."""

import configparser
import json
import os
from pathlib import Path


def resolve(testplan: Path, runner: str, overrides: dict[str, str]) -> dict[str, str]:
    """Resolve a dropdown choice or runner label and apply user overrides."""
    runner = runner.split(" | ")[1].strip() if " | " in runner else runner.strip()
    config = configparser.ConfigParser(interpolation=None)
    with testplan.open(encoding="utf-8") as source:
        config.read_file(source)
    matches = [section for section in config.sections() if config[section].get("runs_on", "").strip() == runner]
    if len(matches) != 1:
        raise ValueError(f"Expected one testplan.ini entry for runner {runner!r}; found {len(matches)}")

    section = matches[0]
    entry = config[section]
    if not entry.getboolean("target_available", fallback=True):
        raise ValueError(f"{section}: runner {runner!r} is disabled in testplan.ini")
    if not section.startswith("linux-"):
        raise ValueError(f"{section}: the E2E job requires a Linux runner")
    values = {"name": section, "runs_on": entry.get("runs_on", "").strip()}
    for key in ("artifact_group", "gpu_arch", "compiler_gpu_arch", "tests_filters", "extra_docker_args"):
        values[key] = entry.get(key, "").strip()
    # An architecture change must not retain a compiler architecture for the previous family.
    gpu_override = overrides.get("gpu_arch", "").strip()
    if gpu_override and gpu_override != values["gpu_arch"] and not overrides.get("compiler_gpu_arch", "").strip():
        values["compiler_gpu_arch"] = ""
    for key, value in overrides.items():
        if value.strip():
            values[key] = value
    for key in ("name", "runs_on", "artifact_group", "gpu_arch", "tests_filters"):
        if not values[key]:
            raise ValueError(f"{section}: missing {key}; supply it in the workflow input or testplan.ini")
    values["compiler_gpu_arch"] = values["compiler_gpu_arch"] or values["gpu_arch"]
    return values


def main() -> None:
    """Read workflow inputs and emit the resolved configuration."""
    keys = ("name", "artifact_group", "gpu_arch", "compiler_gpu_arch", "tests_filters", "extra_docker_args")
    overrides = {key: os.environ[f"INPUT_{key.upper()}"] for key in keys}
    values = resolve(Path("testplan.ini"), os.environ["INPUT_RUNS_ON"], overrides)
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"config={json.dumps(values)}\n")
    for key in ("name", "runs_on", "artifact_group", "gpu_arch", "compiler_gpu_arch", "tests_filters"):
        print(f"{key}: {values[key]}")


if __name__ == "__main__":
    main()
