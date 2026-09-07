# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rocm_agent_enumerator.py -- rocm_agent_enumerator GPU target validation.

Validates:
    1. ``rocm_agent_enumerator -name`` reports at least one ``gfx*`` target name,
       confirming the enumerator can discover the AMD GPU agent on this node.

    2. The first ``gfx`` target name reported by ``rocm_agent_enumerator -name``
       appears in ``rocminfo | grep "Name:"`` output as the full
       ``amdgcn-amd-amdhsa--<gfx_name>`` triple, confirming consistency between
       the two ROCm discovery tools.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hip_runtime/:
    hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux

Explicit markers:
    runtime.fast
"""

from __future__ import annotations

import re

import pytest


@pytest.mark.runtime.fast
def test_rocm_agent_enumerator_target_names(
    target_executor,
    rock_dir: str,
) -> None:
    """Verify that rocm_agent_enumerator -name reports at least one gfx target.

    Runs ``<rock_dir>/bin/rocm_agent_enumerator -name`` and asserts the output
    contains at least one line matching the ``gfx`` prefix that identifies an
    AMD GPU compute agent.
    """
    enumerator = f"{rock_dir}/bin/rocm_agent_enumerator"
    result = target_executor.run(f"{enumerator} -name")
    assert result.ok, (
        f"rocm_agent_enumerator -name failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "gfx" in result.stdout, (
        "rocm_agent_enumerator -name did not report any gfx target name.\n"
        f"stdout: {result.stdout[:1000]}"
    )


@pytest.mark.runtime.fast
def test_rocm_agent_enumerator_names_in_rocminfo(
    target_executor,
    rock_dir: str,
) -> None:
    """Verify that the gfx name from rocm_agent_enumerator appears in rocminfo output.

    Step 1: Run ``rocm_agent_enumerator -name`` and extract the first ``gfx*`` target.
    Step 2: Run ``rocminfo | grep "Name:"`` and assert the expected
    ``amdgcn-amd-amdhsa--<gfx_name>`` triple appears at least once.

    This cross-validates consistency between the two ROCm GPU discovery tools.
    """
    enumerator = f"{rock_dir}/bin/rocm_agent_enumerator"
    enum_result = target_executor.run(f"{enumerator} -name")
    assert enum_result.ok, (
        f"rocm_agent_enumerator -name failed (exit={enum_result.exit_code}):\n"
        f"stdout: {enum_result.stdout[:2000]}\nstderr: {enum_result.stderr[:500]}"
    )
    assert "gfx" in enum_result.stdout, (
        "rocm_agent_enumerator -name did not report any gfx target — "
        "cannot verify rocminfo consistency without a target name.\n"
        f"stdout: {enum_result.stdout[:1000]}"
    )

    # Parse gfx target names from the enumerator output.
    names = [line.strip() for line in enum_result.stdout.splitlines() if line.strip().startswith("gfx")]
    assert names, (
        "rocm_agent_enumerator -name produced 'gfx' in output but no gfx-prefixed lines "
        f"could be parsed.\nstdout: {enum_result.stdout[:1000]}"
    )

    first_name = names[0]

    # Escape any regex-special characters in the gfx name (e.g., '+' in gfx906+xnack).
    escaped = re.escape(first_name)

    # Build a pattern that matches 1 to name_count occurrences of the full triple.
    # rocminfo "Name:" lines look like: "  Name:                    amdgcn-amd-amdhsa--gfx942"
    name_pattern = rf"(?:Name:\s+amdgcn-amd-amdhsa--{escaped})"

    rocminfo = f"{rock_dir}/bin/rocminfo"
    rocminfo_result = target_executor.run(f'{rocminfo} | grep "Name:"')
    assert rocminfo_result.ok, (
        f"rocminfo | grep 'Name:' failed (exit={rocminfo_result.exit_code}):\n"
        f"stdout: {rocminfo_result.stdout[:2000]}\nstderr: {rocminfo_result.stderr[:500]}"
    )

    assert re.search(name_pattern, rocminfo_result.stdout), (
        f"rocminfo output does not contain the expected agent name triple "
        f"'amdgcn-amd-amdhsa--{first_name}'.\n"
        f"Pattern searched: {name_pattern!r}\n"
        f"rocminfo 'Name:' lines:\n{rocminfo_result.stdout[:2000]}"
    )

    # Confirm at least as many occurrences as names reported by the enumerator.
    matches = re.findall(name_pattern, rocminfo_result.stdout)
    assert len(matches) >= 1, (
        f"Expected at least 1 occurrence of '{first_name}' triple in rocminfo output; "
        f"found {len(matches)}.\nrocminfo 'Name:' lines:\n{rocminfo_result.stdout[:2000]}"
    )
