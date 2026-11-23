# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm system tool invocation and output-parsing helpers.

Provides executor-transparent helpers for locating and running ROCm system
binaries (rocminfo, clinfo, hipInfo, rocm_agent_enumerator), asserting
successful execution, and parsing structured tool output.  Shared across
all system_tools test suites.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import shutil

logger = logging.getLogger(__name__)

# rocminfo agent-block patterns — compiled once at module level.
_AGENT_RE = re.compile(r"^Agent \d+$")
_FIELD_RES: dict[str, re.Pattern[str]] = {
    "name": re.compile(r"^Name:\s+(.+)$"),
    "marketing": re.compile(r"^Marketing Name:\s+(.+)$"),
    "vendor": re.compile(r"^Vendor Name:\s+(.+)$"),
    "device_type": re.compile(r"^Device Type:\s+(.+)$"),
}
_L2_RE = re.compile(r"^L2:\s+(.+)$")


def resolve_rocm_tool(tool_name: str, rock_dir: str, executor=None) -> str:
    """Return the path of a ROCm tool binary.

    Prefers ``<rock_dir>/bin/<tool_name>``; falls back to the system PATH.
    Returns an empty string when the binary cannot be found.

    Args:
        tool_name: Binary filename (e.g. ``"rocminfo"``, ``"clinfo"``).
        rock_dir:  TheRock/ROCm install root.
        executor:  Remote executor; ``None`` means local.
    """
    candidate = f"{rock_dir}/bin/{tool_name}"
    if executor is not None:
        probe = executor.run(f"test -x {shlex.quote(candidate)}", timeout=15.0)
        if probe.ok:
            return candidate
        which = executor.run(f"command -v {shlex.quote(tool_name)}", timeout=15.0)
        if which.ok and (which.stdout or "").strip():
            return which.stdout.strip().splitlines()[-1].strip()
        return ""
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return shutil.which(tool_name) or ""


def run_rocm_tool(
    tool_path: str,
    executor,
    args: str = "",
    timeout: float = 60.0,
    extra_env: str = "",
):
    """Run a resolved ROCm system binary on the executor and return the result.

    Args:
        tool_path:  Absolute path to the tool binary.
        executor:   Any executor with a ``.run()`` method.
        args:       Optional CLI arguments appended after the binary path.
        timeout:    Seconds before the run is abandoned.
        extra_env:  Optional ``VAR=value`` pairs prefixed before the command.

    Returns:
        Execution result with ``.ok``, ``.stdout``, ``.stderr``, ``.exit_code``.
    """
    t = shlex.quote(tool_path)
    cmd = f"env {extra_env} {t}" if extra_env else t
    if args:
        cmd = f"{cmd} {args}"
    logger.info("Running ROCm tool: %s", cmd)
    return executor.run(cmd, timeout=timeout)


def assert_tool_executed(result, tool_name: str, min_output_len: int = 1) -> None:
    """Assert the tool exited 0 and produced non-empty stdout.

    Args:
        result:         Execution result from ``executor.run()``.
        tool_name:      Short name used in failure messages.
        min_output_len: Minimum number of characters expected in stdout.
    """
    assert result.ok, (
        f"{tool_name} did not execute successfully (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert result.stdout, f"{tool_name} produced no output"
    assert len(result.stdout) >= min_output_len, (
        f"{tool_name} produced insufficient output "
        f"(length={len(result.stdout)}, expected >= {min_output_len}).\n"
        f"stdout: {result.stdout[:500]}"
    )


def collect_field_values(stdout: str, field_name: str) -> list[str]:
    """Return all values matched by a ``"Key: value"`` field in tool output.

    Scans each line for ``<field_name>: <value>`` and returns every matched
    value.  Used by clinfo, hipInfo, rocminfo, and similar diagnostic tools
    that emit structured key-value output.

    Args:
        stdout:     Full stdout from the tool.
        field_name: Exact field name to match (case-sensitive).

    Returns:
        List of stripped value strings, one per matching line.
    """
    values: list[str] = []
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        if key.strip() == field_name:
            values.append(value.strip())
    return values


def parse_rocminfo_agents(output: str) -> tuple[list[dict], list[str]]:
    """Parse ``rocminfo`` output into per-agent dicts and L2 cache sizes.

    Each agent block starts with an ``"Agent N"`` header followed by
    indented ``"Field: value"`` lines.  Returns agents as a list of dicts
    with keys ``name``, ``marketing``, ``vendor``, ``device_type``, and a
    separate list of L2 cache size strings.

    Args:
        output: Full stdout from ``rocminfo``.

    Returns:
        ``(agents, l2_sizes)`` where ``agents`` is a list of per-agent dicts
        and ``l2_sizes`` is a list of L2 size strings from all agents.
    """
    agents: list[dict] = []
    current: dict | None = None
    l2_sizes: list[str] = []

    for raw in output.splitlines():
        line = raw.strip()
        if _AGENT_RE.match(line):
            current = {k: None for k in _FIELD_RES}
            agents.append(current)
            continue
        if current is None:
            continue
        for key, pattern in _FIELD_RES.items():
            m = pattern.match(line)
            if m and current[key] is None:
                current[key] = m.group(1).strip()
                break
        l2 = _L2_RE.match(line)
        if l2:
            l2_sizes.append(l2.group(1).strip())

    return agents, l2_sizes


def skip_unsupported_arch(gpu_arch: str | None, supported: frozenset[str], label: str) -> None:
    """Skip when *gpu_arch* is not in *supported*.

    Transparent skip for fixtures that build or run arch-specific GPU code
    (CK, hipBLASLt TF32, amd-smi UBB, criu workloads).  Calling
    ``pytest.skip`` from a fixture propagates the skip to every test that
    requests that fixture.

    Args:
        gpu_arch:  Architecture string from the ``gpu_arch`` session fixture
                   (e.g. ``"gfx942"``); ``None`` when ``--no-gpu`` is active.
        supported: Frozenset of accepted architecture strings.
        label:     Short component or feature name used in the skip message.
    """
    import pytest

    if gpu_arch is None or gpu_arch not in supported:
        pytest.skip(f"{label} requires one of {sorted(supported)}; " f"detected arch: {gpu_arch!r}")
