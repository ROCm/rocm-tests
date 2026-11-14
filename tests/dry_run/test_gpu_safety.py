# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate GPU mutation safeguards without GPU hardware or driver operations."""

from __future__ import annotations

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
    if request.config.getoption("--gpu-execution-mode") == "shared":
        raise AssertionError("Shared execution must skip before fixture setup")

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
        ("shared", ["test_sample.py"], 1, 3),
        ("shared", ["test_sample.py::test_mutating"], 0, 1),
        (None, ["test_sample.py::test_mutating"], 0, 1),
        ("shared", ["test_sample.py", "-k", "mutating"], 0, 3),
        ("shared", ["test_sample.py", "-m", "ci.nightly"], 1, 3),
        ("shared", ["test_sample.py", "-m", "gpu_state.mutating"], 0, 3),
        ("shared", ["test_module.py"], 0, 1),
        ("exclusive", ["test_sample.py"], 4, 0),
        ("exclusive", ["test_sample.py::test_mutating"], 1, 0),
        ("exclusive", ["test_module.py"], 1, 0),
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
        command.append(f"--gpu-execution-mode={mode}")
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
            assert "GPU/driver state changes are disabled in shared suites" in skip.get("message", "")
    assert all(case.find("failure") is None and case.find("error") is None for case in cases)


def test_shared_workflow_uses_gpu_safety_guard():
    """Keep shared containers unprivileged and retain caller keyword selection."""
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github" / "workflows" / "e2e-tests.yml").read_text(encoding="utf-8"))
    job = workflow["jobs"]["platform"]
    options = job["container"]["options"]
    assert "--privileged" not in options
    assert "--cap-add SYS_MODULE" not in options
    command = next(step["run"] for step in job["steps"] if step["name"] == "Run e2e tests")
    assert "--gpu-execution-mode=shared" in command
    assert 'KEYWORD_ARGS=(-k "${KEYWORD_FILTER}")' in command
