# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate GPU mutation safeguards without GPU hardware or driver operations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

pytestmark = [
    pytest.mark.hw.cpu_only,  # type: ignore[attr-defined]
    pytest.mark.ci.pr,  # type: ignore[attr-defined]
    pytest.mark.layer.runtime,  # type: ignore[attr-defined]
    pytest.mark.runtime.fast,  # type: ignore[attr-defined]
]

_TEST_SOURCE = """
import pytest

mutating = getattr(pytest.mark, "gpu_state.mutating")
pytestmark = getattr(pytest.mark, "ci.nightly")

@pytest.fixture
def mutation_fixture(request):
    if request.config.getoption("--gpu-state-mutations") == "skip":
        raise AssertionError("Skip mode must prevent fixture setup for mutating tests")

@mutating
def test_mutating(mutation_fixture):
    pass

@mutating
class TestMutating:
    def test_inherited(self, mutation_fixture):
        pass

@pytest.mark.parametrize("value", [pytest.param(1, marks=mutating)])
def test_parameter(value, mutation_fixture):
    pass

def test_read_only():
    pass
"""


@pytest.mark.parametrize(
    ("mode", "selector", "passed", "skipped"),
    [
        (None, ["test_sample.py"], 1, 3),
        ("skip", ["test_sample.py"], 1, 3),
        ("skip", ["test_sample.py::test_mutating"], 0, 1),
        (None, ["test_sample.py::test_mutating"], 0, 1),
        ("skip", ["test_sample.py", "-k", "mutating"], 0, 3),
        ("skip", ["test_sample.py", "-m", "ci.nightly"], 1, 3),
        ("skip", ["test_sample.py", "-m", "gpu_state.mutating"], 0, 3),
        ("skip", ["test_module.py"], 0, 1),
        ("allow", ["test_sample.py"], 4, 0),
        ("allow", ["test_sample.py::test_mutating"], 1, 0),
        ("allow", ["test_module.py"], 1, 0),
    ],
)
def test_gpu_mutation_guard(tmp_path, mode, selector, passed, skipped):
    """Check selection, marker inheritance, fixture prevention, and opt-in."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "test_sample.py").write_text(_TEST_SOURCE, encoding="utf-8")
    (tmp_path / "test_module.py").write_text(
        "import pytest\n" 'pytestmark = getattr(pytest.mark, "gpu_state.mutating")\n' "def test_mutating(): pass\n",
        encoding="utf-8",
    )
    report = tmp_path / "results.xml"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(root / "pyproject.toml"),
        f"--rootdir={tmp_path}",
        "--noconftest",
        "-p",
        "framework.plugins.gpu_safety",
        "--strict-markers",
        f"--junitxml={report}",
        "-q",
        *selector,
    ]
    if mode:
        command.append(f"--gpu-state-mutations={mode}")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = ET.parse(report).findall(".//testcase")
    assert len(cases) == passed + skipped
    assert sum(case.find("skipped") is not None for case in cases) == skipped
    for case in cases:
        skip = case.find("skipped")
        if skip is not None:
            assert "GPU/driver state mutations are disabled" in skip.get("message", "")
    assert all(case.find("failure") is None and case.find("error") is None for case in cases)


def test_allow_mutations_rejects_xdist_workers(tmp_path):
    """Do not allow xdist parallelism when tests can mutate GPU state."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "test_mutating.py").write_text(
        "import pytest\n" 'pytestmark = getattr(pytest.mark, "gpu_state.mutating")\n' "def test_mutating(): pass\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(root / "pyproject.toml"),
        f"--rootdir={tmp_path}",
        "--noconftest",
        "-p",
        "xdist.plugin",
        "-p",
        "framework.plugins.gpu_safety",
        "--gpu-state-mutations=allow",
        "-n",
        "2",
        "test_mutating.py",
        "-q",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode != 0
    assert "--gpu-state-mutations=allow cannot be used with pytest-xdist workers" in result.stdout + result.stderr


def test_shared_workflow_relies_on_safe_mutation_default():
    """Keep common workflow unprivileged and use the plugin's safe default."""
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github" / "workflows" / "e2e-tests.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["platform"]
    options = job["container"]["options"]
    assert "--privileged" not in options
    assert "--cap-add SYS_MODULE" not in options
    step_run = next((step["run"] for step in job["steps"] if step["name"] == "Run e2e tests"), None)
    assert step_run is not None, "step 'Run e2e tests' not found in e2e-tests.yml"
    command = step_run
    assert "--gpu-state-mutations" not in command
    assert 'KEYWORD_ARGS=(-k "${KEYWORD_FILTER}")' in command


def test_safety_skips_keep_allure_labels(tmp_path):
    """Safety skips retain reporting labels without running test fixtures."""
    root = Path(__file__).resolve().parents[2]
    (tmp_path / "test_sample.py").write_text(_TEST_SOURCE, encoding="utf-8")
    results_dir = tmp_path / "allure-results"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(root / "pyproject.toml"),
        f"--rootdir={tmp_path}",
        "--noconftest",
        "-p",
        "allure_pytest",
        "-p",
        "framework.plugins.reports_plugin",
        "-p",
        "framework.plugins.gpu_safety",
        f"--alluredir={results_dir}",
        "test_sample.py::test_mutating",
        "test_sample.py::test_read_only",
        "-q",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    results = [json.loads(path.read_text(encoding="utf-8")) for path in results_dir.glob("*-result.json")]
    assert len(results) == 2
    by_name = {entry["name"]: entry for entry in results}
    skipped = by_name["test_mutating"]
    assert skipped["status"] == "skipped"
    assert "GPU/driver state mutations are disabled" in skipped["statusDetails"]["message"]
    assert {"name": "tag", "value": "gpu_state.mutating"} in skipped["labels"]
    for entry in results:
        assert {"name": "feature", "value": "ci.nightly"} in entry["labels"]
    assert by_name["test_read_only"]["status"] == "passed"
