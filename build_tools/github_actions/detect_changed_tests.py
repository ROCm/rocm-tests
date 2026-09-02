#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Detect which test files are impacted by a PR's changed file list.

Reads CHANGED_FILES from the environment (newline-separated relative paths as
produced by ``git diff --name-only``), applies impact rules, and writes two
GitHub Actions outputs:

    test_paths   Space-separated list of paths to pass to pytest.
                 Empty when nothing is impacted.
    has_tests    'true' or 'false'.

Impact rules (evaluated in order; first matching rule for each changed file wins):

1. ``tests/e2e/<area>/test_*.py``
   → include that file directly.

2. ``tests/e2e/<area>/conftest.py``
   ``tests/e2e/<area>/_workload.py``
   ``tests/e2e/<area>/src/**``
   → include the entire ``tests/e2e/<area>/`` directory.

3. ``tests/common/_cmake_build.py``
   → include all test directories that use CMake builds:
     compiler, hipblaslt, hwq_heuristic, rocprim, rocm_libs, hpc/quda, recovery/criu.

4. ``tests/common/**`` (any other shared factory/util)
   → include the full test suite.

5. ``conftest.py`` (repo root)
   → include the full test suite.

6. ``framework/**``
   → include the full test suite (framework change may affect any test).

7. Any other change outside ``tests/`` and ``framework/`` is ignored — no tests run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath

# Allow imports from both build_tools/github_actions/ and the repo root.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT))

from framework.markers.taxonomy import CATEGORY_PROFILES  # noqa: E402
from github_actions_api import gha_set_output  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

# Derived from CATEGORY_PROFILES — the single source of truth for e2e dirs.
# Adding a new area to taxonomy.py automatically includes it here.
_ALL_E2E_DIRS: list[str] = sorted(f"{k}/" for k in CATEGORY_PROFILES)

# Directories whose tests depend on the shared CMake build helper.
_CMAKE_DIRS: list[str] = [
    "tests/e2e/compiler/",
    "tests/e2e/hipblaslt/",
    "tests/e2e/hwq_heuristic/",
    "tests/e2e/rocprim/",
    "tests/e2e/rocm_libs/",
    "tests/e2e/hpc/quda/",
    "tests/e2e/recovery/criu/",
]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _e2e_area(path: PurePosixPath) -> str | None:
    """Return the e2e sub-area prefix if path lives under tests/e2e/, else None.

    For ``tests/e2e/hip_runtime/test_foo.py`` returns ``tests/e2e/hip_runtime``.
    For ``tests/e2e/hpc/quda/test_foo.py`` returns ``tests/e2e/hpc/quda``.
    """
    parts = path.parts
    if len(parts) < 3 or parts[0] != "tests" or parts[1] != "e2e":
        return None
    # Walk back from the file to find the deepest directory that has a known
    # profile entry (or just use the immediate parent for test files).
    depth = len(parts) - 1  # index of the filename
    # For paths like tests/e2e/hpc/quda/... we want tests/e2e/hpc/quda.
    # We use all directory components after tests/e2e/ as the area.
    area_parts = parts[2:depth]
    if not area_parts:
        return None
    return "tests/e2e/" + "/".join(area_parts)


def _is_test_file(path: PurePosixPath) -> bool:
    return path.name.startswith("test_") and path.suffix == ".py"


def _is_conftest(path: PurePosixPath) -> bool:
    return path.name == "conftest.py"


def _is_workload_or_src(path: PurePosixPath) -> bool:
    """True for underscore-prefixed files (e.g. _workload.py) or files under a src/ subtree."""
    return path.name.startswith("_") or "src" in path.parts


# ── Core detection logic ──────────────────────────────────────────────────────


def detect_impacted_paths(changed_files: list[str]) -> list[str]:
    """Map a list of changed file paths to the pytest paths that should run.

    Returns a deduplicated, sorted list of paths (files or directories).
    """
    impacted: set[str] = set()
    full_suite = False

    for raw in changed_files:
        path = PurePosixPath(raw.strip())
        parts = path.parts
        if not parts:
            continue

        # ── Rule 6: framework/** → full suite ────────────────────────────────
        if parts[0] == "framework":
            full_suite = True
            break

        # ── Rule 5: any conftest.py at or above the e2e area level → full suite
        # Covers: conftest.py (root), tests/conftest.py, tests/e2e/conftest.py.
        # Per-area conftest.py files (tests/e2e/<area>/conftest.py) are handled
        # by Rule 2 below via _is_conftest().
        if path.name == "conftest.py" and (
            len(parts) == 1  # repo root
            or (len(parts) == 2 and parts[0] == "tests")  # tests/conftest.py
            or (len(parts) == 3 and parts[0] == "tests" and parts[1] == "e2e")  # tests/e2e/conftest.py
        ):
            full_suite = True
            break

        # ── Rules 3 & 4: tests/common/** ─────────────────────────────────────
        if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "common":
            if path.name == "_cmake_build.py":
                # Rule 3: only CMake-based dirs
                for d in _CMAKE_DIRS:
                    impacted.add(d)
            else:
                # Rule 4: any other shared utility → full suite
                full_suite = True
                break
            continue

        # ── Guard: ignore everything outside tests/e2e/ ──────────────────────
        # Reaches here only for paths that start with "tests/" but are not
        # tests/common/ and not caught by the conftest rule above.
        # Only tests/e2e/** is relevant to the remaining rules.
        if parts[0] != "tests" or (len(parts) > 1 and parts[1] != "e2e"):
            continue

        # ── Rules 1 & 2: tests/e2e/** ────────────────────────────────────────
        area = _e2e_area(path)
        if area is None:
            continue

        area_prefix = area + "/"

        if _is_test_file(path):
            # Rule 1: include just this test file.
            impacted.add(raw.strip())
        elif _is_conftest(path) or _is_workload_or_src(path):
            # Rule 2: whole directory — conftest, underscore helpers, src/ trees.
            impacted.add(area_prefix)
        elif path.suffix == ".py":
            # Any other .py in the area (helper module, __init__, etc.) → whole dir.
            # Note: this branch is intentionally separate from _is_workload_or_src;
            # removing it would silently drop non-underscore helper modules.
            impacted.add(area_prefix)
        elif path.suffix in (".cpp", ".hip", ".h", ".hpp", ".cu", ".cmake") or \
                path.name == "CMakeLists.txt":
            # Non-python sources in e2e — whole dir.
            # path.suffix for "CMakeLists.txt" is ".txt", so check name directly.
            impacted.add(area_prefix)

    if full_suite:
        return sorted(_ALL_E2E_DIRS)

    # Deduplicate: if a directory is included, drop individual files under it.
    dirs = {p for p in impacted if p.endswith("/")}
    files = {p for p in impacted if not p.endswith("/")}
    filtered_files = {f for f in files if not any(f.startswith(d) for d in dirs)}
    return sorted(dirs | filtered_files)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    raw = os.environ.get("CHANGED_FILES", "").strip()
    changed_files = [line for line in raw.splitlines() if line.strip()]

    if not changed_files:
        print("No changed files provided — nothing to run.", flush=True)
        gha_set_output({"test_paths": "", "has_tests": "false"})
        return 0

    print(f"Changed files ({len(changed_files)}):", flush=True)
    for f in changed_files:
        print(f"  {f}", flush=True)

    paths = detect_impacted_paths(changed_files)

    if paths:
        test_paths_str = " ".join(paths)
        print(f"\nImpacted test paths ({len(paths)}):", flush=True)
        for p in paths:
            print(f"  {p}", flush=True)
        gha_set_output({"test_paths": test_paths_str, "has_tests": "true"})
    else:
        print("\nNo test files impacted by these changes.", flush=True)
        gha_set_output({"test_paths": "", "has_tests": "false"})

    return 0


if __name__ == "__main__":
    sys.exit(main())
