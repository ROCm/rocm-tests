# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocminfo.py -- ROCr rocminfo agent enumeration validation.

Runs ``rocminfo`` on an AMD GPU node and validates the reported agent topology:

- rocminfo executes and produces output.
- Every agent reports a Device Type.
- Every GPU agent reports a gfx name.
- Every agent reports a Vendor Name.
- When no GPU is present, a readable error message is emitted.
- An L2 cache size is printed for at least every GPU agent.

hw.gpu, ci.nightly, layer.runtime, runtime.fast and os.linux are declared explicitly.
"""

from pathlib import PurePosixPath
import re

import pytest

# rocminfo prints one block per agent, each starting with an "Agent N" header
# followed by indented "Field: value" lines.
_AGENT_RE = re.compile(r"^Agent \d+$")
_FIELD_RES = {
    "name": re.compile(r"^Name:\s+(.+)$"),
    "marketing": re.compile(r"^Marketing Name:\s+(.+)$"),
    "vendor": re.compile(r"^Vendor Name:\s+(.+)$"),
    "device_type": re.compile(r"^Device Type:\s+(.+)$"),
}
_L2_RE = re.compile(r"^L2:\s+(.+)$")


def _parse_agents(output: str):
    """Parse rocminfo output into per-agent field dicts and the list of L2 cache sizes."""
    agents: list[dict] = []
    current: dict | None = None
    l2_sizes: list[str] = []
    for raw in output.splitlines():
        line = raw.strip()
        if _AGENT_RE.match(line):
            current = {"name": None, "marketing": None, "vendor": None, "device_type": None}
            agents.append(current)
            continue
        if current is None:
            continue
        matched = False
        for key, pattern in _FIELD_RES.items():
            field = pattern.match(line)
            if field and current[key] is None:
                current[key] = field.group(1).strip()
                matched = True
                break
        if matched:
            continue
        l2 = _L2_RE.match(line)
        if l2:
            l2_sizes.append(l2.group(1).strip())
    return agents, l2_sizes


def _evaluate(agents: list[dict], l2_sizes: list[str]):
    """Return a check-name -> bool mapping plus the per-agent device-type list."""
    n_agents = len(agents)
    gpu_type = [a["device_type"] for a in agents]
    vendor = [a["vendor"] for a in agents]
    gpu_agents = [a for a in agents if a["device_type"] == "GPU"]
    gpu_marketing_names = [a["marketing"] for a in gpu_agents]
    gpu_id = [a["name"] for a in gpu_agents if a["name"] and a["name"].startswith("gfx")]

    checks = {
        "Device type is GPU for vendor AMD": bool(gpu_type) and len(gpu_type) == n_agents and None not in gpu_type,
        "Name starts from gfx": bool(gpu_id) and len(gpu_id) == len(gpu_agents) and None not in gpu_id,
        "Vendor name is present": bool(vendor) and len(vendor) == n_agents and None not in vendor,
        "L2 is printed": bool(l2_sizes) and len(l2_sizes) >= len(gpu_marketing_names),
    }
    return checks, gpu_type


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
@pytest.mark.os.linux
def test_rocminfo(target_executor, rock_dir):
    """Validate rocminfo agent enumeration on an AMD GPU node."""
    # ``rocminfo`` is frequently absent from PATH — it lives under a versioned
    # ROCm install (e.g. /opt/rocm-7.15.0/bin/rocminfo). Invoke it by full path
    # off the resolved TheRock/ROCm bin dir.
    therock_bin_dir = PurePosixPath(rock_dir) / "bin"
    rocminfo = PurePosixPath(therock_bin_dir) / "rocminfo"
    result = target_executor.run(str(rocminfo))
    diag = f"(exit={result.exit_code})\nstdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    executed = bool(result.ok and result.stdout)
    assert executed, f"rocminfo did not execute {diag}"

    agents, l2_sizes = _parse_agents(result.stdout)
    checks, gpu_type = _evaluate(agents, l2_sizes)

    # A readable error message is required only when no GPU agent is reported.
    if "GPU" not in gpu_type:
        checks["No-GPU error message is readable"] = bool(result.stderr)

    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"rocminfo checks failed: {failed} {diag}"
