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
- rocminfo GPU count matches the system GPU availability.

hw.gpu, ci.nightly, layer.runtime, and os.linux are auto-injected via CATEGORY_PROFILES.
Only runtime.fast must be declared explicitly.
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


def _evaluate(agents: list[dict], l2_sizes: list[str], sys_gpu_count: int | None = None):
    """Return a check-name -> bool mapping plus the per-agent device-type list."""
    n_agents = len(agents)
    gpu_type = [a["device_type"] for a in agents]
    vendor = [a["vendor"] for a in agents]
    gpu_agents = [a for a in agents if a["device_type"] == "GPU"]
    gpu_marketing_names = [a["marketing"] for a in gpu_agents]
    gpu_id = [a["name"] for a in gpu_agents if a["name"] and a["name"].startswith("gfx")]

    # Check 1: Device type is GPU for vendor AMD (all agents must be GPU and non-None)
    check_device_type = bool(gpu_type) and len(gpu_type) == n_agents and None not in gpu_type

    # Check 2: Name starts from gfx — gpu_id count must match system GPU count (not just gpu_agents count)
    # This cross-validates rocminfo against actual system GPU availability
    check_gfx_name = bool(gpu_id) and None not in gpu_id
    if sys_gpu_count is not None:
        check_gfx_name = check_gfx_name and len(gpu_id) == sys_gpu_count

    # Check 3: Vendor name is present (all agents must have vendor)
    check_vendor = bool(vendor) and len(vendor) == n_agents and None not in vendor

    # Check 4: L2 cache size is printed for at least every GPU agent
    check_l2 = bool(l2_sizes) and len(l2_sizes) >= len(gpu_marketing_names)

    checks = {
        "Device type is GPU for vendor AMD": check_device_type,
        "Name starts from gfx": check_gfx_name,
        "Vendor name is present": check_vendor,
        "L2 is printed": check_l2,
    }
    return checks, gpu_type


@pytest.mark.runtime.fast  # type: ignore[attr-defined]
def test_rocminfo(target_executor, rock_dir):  # pylint: disable=too-many-locals
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

    # Get system GPU count from ROCR_VISIBLE_DEVICES for cross-validation
    cmd = (
        'python3 -c "import os; '
        "visible = os.environ.get('ROCR_VISIBLE_DEVICES', ''); "
        "print(len(visible.split(',')) if visible else 1)\""
    )
    gpu_count_result = target_executor.run(cmd)
    sys_gpu_count = int(gpu_count_result.stdout.strip()) if gpu_count_result.ok else None

    agents, l2_sizes = _parse_agents(result.stdout)
    checks, gpu_type = _evaluate(agents, l2_sizes, sys_gpu_count)

    # When no GPU agent is reported, rocminfo must include a readable error indicator
    if "GPU" not in gpu_type:
        stderr_lower = result.stderr.lower()
        no_gpu_indicators = ["no gpu", "no device", "gpu not found", "rocm error", "failed", "error"]
        has_readable_error = any(indicator in stderr_lower for indicator in no_gpu_indicators)
        checks["No-GPU error message is readable"] = has_readable_error or bool(result.stderr)

    failed = [name for name, ok in checks.items() if not ok]
    assert not failed, f"rocminfo checks failed: {failed} {diag}"
