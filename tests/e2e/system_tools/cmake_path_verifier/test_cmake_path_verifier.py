# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_cmake_path_verifier.py -- ROCm cmake packaging path compliance check.

Ported from: cmake_path_verifier.py (ROCr test area, AMD internal framework).

Validates:
    For each ROCm package under {rock_dir}/lib/cmake/<package>/:
    1. The cmake config directory exists (mandatory — missing dir is a fail).
    2. No cmake file in the tree contains a hardcoded /opt/rocm path outside
       the permitted HIP fallback pattern (HINTS ${ROCM_PATH} PATHS "/opt/rocm").

Packages must resolve their build-time dependencies via CMAKE_PREFIX_PATH and
find_package() rather than assuming /opt/rocm is the install prefix. A
hardcoded prefix causes silent build failures when ROCm is installed elsewhere.

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
    HIP_ALLOWED_PATTERN,
    HARDCODE_PATTERN,
    PACKAGES_TO_VERIFY,
)

logger = logging.getLogger("rocm.test")

_RE_HARDCODE = re.compile(HARDCODE_PATTERN)
_RE_HIP_ALLOWED = re.compile(HIP_ALLOWED_PATTERN)


def _check_file_for_hardcode(content: str, is_hip: bool) -> list[str]:
    """Return lines that violate the hardcode rule.

    For hip cmake files the HIP-permitted fallback line is skipped before
    the general hardcode check is applied.
    """
    violations: list[str] = []
    for line in content.splitlines():
        if is_hip and _RE_HIP_ALLOWED.search(line):
            continue
        if _RE_HARDCODE.search(line):
            violations.append(line)
    return violations


@pytest.mark.runtime.fast
@pytest.mark.parametrize("package", PACKAGES_TO_VERIFY)
def test_cmake_package_path_verifier(
    target_executor,
    rock_dir: str,
    package: str,
) -> None:
    """Assert that the cmake config directory for *package* exists and contains no hardcoded /opt/rocm paths.

    Fails hard when the cmake directory is absent (it must be present in a correctly
    installed ROCm build). Fails hard when any cmake file contains a line that hardcodes
    /opt/rocm outside the permitted HIP fallback pattern.
    """
    rocm_root = rock_dir or "/opt/rocm"
    cmake_root = f"{rocm_root}/lib/cmake"
    package_dir = f"{cmake_root}/{package}"

    logger.info("verifying cmake config dir for package=%s at %s", package, package_dir)

    # Rule 27: missing cmake dir is a hard fail — cmake config is mandatory.
    dir_check = target_executor.run(f"test -d {package_dir} && echo EXISTS || echo MISSING")
    if "MISSING" in (dir_check.stdout or ""):
        pytest.fail(
            f"cmake config directory missing for package '{package}': {package_dir}\n"
            f"Expected the package to be installed with its cmake config files under "
            f"{cmake_root}/<package>/. Install the package (or its -devel variant) and retry."
        )

    logger.info("cmake dir found for package=%s — listing cmake files", package)

    # List all files recursively; use find to avoid shell glob limitations on large trees.
    find_result = target_executor.run(f"find {package_dir} -type f")
    if not find_result.ok:
        pytest.fail(
            f"'find {package_dir} -type f' failed (exit={find_result.exit_code}):\n"
            f"stderr: {find_result.stderr[:500]}"
        )

    cmake_files = [p.strip() for p in (find_result.stdout or "").splitlines() if p.strip()]
    if not cmake_files:
        pytest.fail(
            f"cmake config directory exists but contains no files: {package_dir}\n"
            f"A correctly packaged ROCm component must ship at least one "
            f"*-config.cmake or *Targets.cmake file."
        )

    logger.info("package=%s — found %d cmake file(s); scanning for hardcoded /opt/rocm", package, len(cmake_files))

    is_hip = package == "hip"
    all_violations: dict[str, list[str]] = {}

    for filepath in cmake_files:
        cat_result = target_executor.run(f"cat {filepath}")
        if not cat_result.ok:
            # Non-readable file (e.g. permissions) — report as a violation so the
            # suite does not silently pass on unreadable cmake files.
            all_violations[filepath] = [f"<unreadable: exit={cat_result.exit_code}>"]
            continue

        violations = _check_file_for_hardcode(cat_result.stdout or "", is_hip)
        if violations:
            all_violations[filepath] = violations

    if all_violations:
        summary_lines = [
            f"package '{package}': hardcoded /opt/rocm found in {len(all_violations)} file(s).",
            "Packages must use CMAKE_PREFIX_PATH + find_package() — never assume /opt/rocm.",
            "",
        ]
        for filepath, lines in all_violations.items():
            summary_lines.append(f"  {filepath}:")
            for line in lines[:5]:  # cap per-file output to keep assertion readable
                summary_lines.append(f"    {line}")
            if len(lines) > 5:
                summary_lines.append(f"    ... and {len(lines) - 5} more line(s)")
        pytest.fail("\n".join(summary_lines))

    logger.info(
        "package=%s PASSED — %d cmake file(s) verified, no hardcoded /opt/rocm paths found",
        package,
        len(cmake_files),
    )
