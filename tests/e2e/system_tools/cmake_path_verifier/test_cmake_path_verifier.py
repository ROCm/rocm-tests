# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_cmake_path_verifier.py -- ROCm cmake packaging path compliance check.

Validates:
    1. cmake config directories exist for hip, amd_comgr, and amd-dbgapi (mandatory).
    2. All cmake config directories found under {rocm_dir}/lib/cmake/ are scanned
       and must not contain any hardcoded /opt/rocm path outside the permitted HIP
       fallback pattern (HINTS ${ROCM_PATH} PATHS "/opt/rocm").

Packages are installed by the session fixture in conftest.py before scanning.
The scan is dynamic — whatever cmake subdirectories exist after installation
are all verified, with no hardcoded expected list beyond the three mandatory ones.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/system_tools/cmake_path_verifier/:
    hw.cpu_only, layer.runtime, ci.nightly, os.linux

Explicit markers (not auto-injected):
    runtime.fast
"""

from __future__ import annotations

import logging
import re

import pytest

from tests.e2e.system_tools.cmake_path_verifier._constants import (
    HARDCODE_PATTERN,
    HIP_ALLOWED_PATTERN,
    PACKAGES_ALWAYS_VERIFY,
)

logger = logging.getLogger("rocm.test")

_RE_HARDCODE = re.compile(HARDCODE_PATTERN)
_RE_HIP_ALLOWED = re.compile(HIP_ALLOWED_PATTERN)


def _check_file_for_hardcode(content: str, is_hip: bool) -> list[str]:
    """Return lines that violate the hardcode rule."""
    violations: list[str] = []
    for line in content.splitlines():
        if is_hip and _RE_HIP_ALLOWED.search(line):
            continue
        if _RE_HARDCODE.search(line):
            violations.append(line)
    return violations


def _verify_package(executor, package_dir: str, package: str) -> list[str]:
    """Check all cmake files under *package_dir* for hardcoded /opt/rocm paths.

    Returns a list of human-readable violation strings (empty means clean).
    """
    find_result = executor.run(f"find {package_dir} -type f")
    if not find_result.ok:
        return [f"find failed (exit={find_result.exit_code}): {(find_result.stderr or '')[:300]}"]

    cmake_files = [p.strip() for p in (find_result.stdout or "").splitlines() if p.strip()]
    if not cmake_files:
        return [f"cmake directory exists but contains no files: {package_dir}"]

    is_hip = package == "hip"
    violations: list[str] = []
    for filepath in cmake_files:
        cat_result = executor.run(f"cat {filepath}")
        if not cat_result.ok:
            violations.append(f"{filepath}: <unreadable: exit={cat_result.exit_code}>")
            continue
        for line in _check_file_for_hardcode(cat_result.stdout or "", is_hip):
            violations.append(f"{filepath}: {line}")
    return violations


@pytest.mark.runtime.fast
def test_cmake_mandatory_packages_present(target_executor, rock_dir: str) -> None:
    """Fail if hip, amd_comgr, or amd-dbgapi are missing their cmake config directories."""
    rocm_root = rock_dir or "/opt/rocm"
    cmake_root = f"{rocm_root}/lib/cmake"
    missing = []
    for package in PACKAGES_ALWAYS_VERIFY:
        result = target_executor.run(f"test -d {cmake_root}/{package} && echo EXISTS || echo MISSING")
        if "MISSING" in (result.stdout or ""):
            missing.append(f"{cmake_root}/{package}")
    if missing:
        pytest.fail(
            "Mandatory cmake config directories missing:\n"
            + "\n".join(f"  {p}" for p in missing)
        )


@pytest.mark.runtime.fast
def test_cmake_no_hardcoded_paths(target_executor, rock_dir: str) -> None:
    """Scan all cmake config directories under {rocm_dir}/lib/cmake/ and fail on hardcoded /opt/rocm paths.

    The list of packages is discovered dynamically from the filesystem — no hardcoded expected list.
    """
    rocm_root = rock_dir or "/opt/rocm"
    cmake_root = f"{rocm_root}/lib/cmake"

    # Discover all cmake package subdirectories present after installation.
    list_result = target_executor.run(f"find {cmake_root} -maxdepth 1 -mindepth 1 -type d")
    if not list_result.ok:
        pytest.fail(
            f"Failed to list cmake directories under {cmake_root} "
            f"(exit={list_result.exit_code}):\n{(list_result.stderr or '')[:300]}"
        )

    package_dirs = [p.strip() for p in (list_result.stdout or "").splitlines() if p.strip()]
    if not package_dirs:
        pytest.fail(f"No cmake package directories found under {cmake_root}")

    logger.info("found %d cmake package(s) to verify under %s", len(package_dirs), cmake_root)

    all_violations: dict[str, list[str]] = {}
    for package_dir in sorted(package_dirs):
        package = package_dir.rsplit("/", 1)[-1]
        logger.info("checking package=%s", package)
        violations = _verify_package(target_executor, package_dir, package)
        if violations:
            all_violations[package] = violations

    if all_violations:
        lines = [
            f"{len(all_violations)} package(s) contain hardcoded /opt/rocm paths.",
            "Packages must use CMAKE_PREFIX_PATH + find_package() — never assume /opt/rocm.",
            "",
        ]
        for pkg, viols in all_violations.items():
            lines.append(f"  {pkg}:")
            for v in viols[:5]:
                lines.append(f"    {v}")
            if len(viols) > 5:
                lines.append(f"    ... and {len(viols) - 5} more")
        pytest.fail("\n".join(lines))

    logger.info("all %d cmake package(s) passed — no hardcoded /opt/rocm paths found", len(package_dirs))
