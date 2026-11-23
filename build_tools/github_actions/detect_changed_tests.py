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

6. ``framework/**`` — subdirectory-aware:
   - ``framework/executors/**``, ``framework/builder/**``, ``framework/common/**``,
     ``framework/plugins/**``, or ``framework/nodes/**``
     → include the full test suite (core executor/plugin change may affect any test).
   - ``framework/markers/taxonomy.py`` (diff parsed):
     New ``CATEGORY_PROFILES`` keys → those specific e2e area directories only.
     No new profile keys (e.g. ``MARKER_SCHEMA`` edit) → falls to smoke run.
   - Any other ``framework/`` file (``config/``, ``gpu/``, ``logging/``, ``markers/``
     excl. ``taxonomy.py``, ``os_adapter/``, ``reporting/``, ``results/``, ``rocm/``,
     ``scheduling/``):
     → smoke run: alphabetically first ``test_*.py`` in each ``tests/e2e/`` area.

7. Any other change outside ``tests/`` and ``framework/`` is ignored — no tests run.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import sys

# Allow imports from both build_tools/github_actions/ and the repo root.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_REPO_ROOT))

from github_actions_api import gha_set_output  # noqa: E402

from framework.markers.taxonomy import CATEGORY_PROFILES  # noqa: E402

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

# Framework subdirs whose changes warrant running the full test suite.
# Changes to core executor/plugin infrastructure can affect every test.
_CRITICAL_FW_DIRS: frozenset[str] = frozenset(
    {
        "framework/executors",
        "framework/builder",
        "framework/common",
        "framework/plugins",
        "framework/nodes",
    }
)

# Regex to find newly added CATEGORY_PROFILES keys in a taxonomy.py unified diff.
# Matches lines like: +    "tests/e2e/rocwmma": [
_CATEGORY_PROFILE_KEY_RE = re.compile(r'^\+\s{4}"(tests/e2e/[^"]+)"\s*:\s*\[')


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse_taxonomy_diff(diff_text: str) -> list[str]:
    """Return ``'tests/e2e/<area>/'`` for each new ``CATEGORY_PROFILES`` key in the diff.

    Parses a unified diff of ``framework/markers/taxonomy.py`` and extracts the path
    component of any added dict keys.  Returns an empty list for ``MARKER_SCHEMA``-only
    edits or when ``diff_text`` is empty.
    """
    return [match.group(1) + "/" for line in diff_text.splitlines() if (match := _CATEGORY_PROFILE_KEY_RE.match(line))]


def _smoke_paths(repo_root: Path) -> list[str]:
    """Return one test file per e2e area: the alphabetically first ``test_*.py``.

    Used for non-critical ``framework/`` changes that do not warrant the full suite.
    Areas with no ``test_*.py`` files are silently skipped.
    """
    smoke: list[str] = []
    for area_key in sorted(CATEGORY_PROFILES):
        area_dir = repo_root / area_key
        test_files = sorted(area_dir.glob("test_*.py"))
        if test_files:
            smoke.append(test_files[0].relative_to(repo_root).as_posix())
    return smoke


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


def _classify_e2e_path(path: PurePosixPath, raw: str) -> str | None:
    """Return the pytest path to add for a tests/e2e/** changed file, or None to skip.

    Rules 1 & 2 from the module-level docstring:
    - test_*.py → the file itself (Rule 1)
    - conftest, underscore helpers, src/ trees, other .py, C/HIP/CMake sources → whole area dir (Rule 2)
    - anything else (e.g. .md, .txt non-CMake) → None (no test impact)
    """
    area = _e2e_area(path)
    if area is None:
        return None
    area_prefix = area + "/"
    if _is_test_file(path):
        return raw.strip()
    if (
        _is_conftest(path)
        or _is_workload_or_src(path)
        or path.suffix == ".py"
        or path.suffix in (".cpp", ".hip", ".h", ".hpp", ".cu", ".cmake")
        or path.name == "CMakeLists.txt"
    ):
        return area_prefix
    return None


# ── Core detection logic ──────────────────────────────────────────────────────


def _is_global_conftest(path: PurePosixPath) -> bool:
    """Return True for conftest.py files that trigger a full-suite run (Rules 5).

    Covers the repo root, ``tests/``, and ``tests/e2e/`` levels only.
    Per-area ``tests/e2e/<area>/conftest.py`` files are handled by Rule 2.
    """
    parts = path.parts
    return path.name == "conftest.py" and (
        len(parts) == 1
        or (len(parts) == 2 and parts[0] == "tests")
        or (len(parts) == 3 and parts[0] == "tests" and parts[1] == "e2e")
    )


def _classify_single_file(
    raw: str,
    critical_fw_hit: bool,
    other_fw_files: set[str],
    impacted: set[str],
) -> tuple[bool, bool]:
    """Classify one changed file path and update the mutable collections in-place.

    Returns ``(full_suite_triggered, critical_fw_hit)`` after processing this file.
    ``full_suite_triggered`` being True signals the caller to stop the loop early.
    """
    path = PurePosixPath(raw.strip())
    parts = path.parts
    if not parts:
        return False, critical_fw_hit

    # Rule 6: framework/**
    if parts[0] == "framework":
        fw_subdir = f"framework/{parts[1]}" if len(parts) > 1 else "framework"
        if fw_subdir in _CRITICAL_FW_DIRS:
            return False, True
        other_fw_files.add(raw.strip())
        return False, critical_fw_hit

    # Rule 5: global conftest.py → full suite
    if _is_global_conftest(path):
        return True, critical_fw_hit

    # Rules 3 & 4: tests/common/**
    if len(parts) >= 2 and parts[0] == "tests" and parts[1] == "common":
        if path.name == "_cmake_build.py":
            impacted.update(_CMAKE_DIRS)
        else:
            return True, critical_fw_hit  # Rule 4: full suite
        return False, critical_fw_hit

    # Guard: only tests/e2e/** remains relevant
    if parts[0] != "tests" or (len(parts) > 1 and parts[1] != "e2e"):
        return False, critical_fw_hit

    # Rules 1 & 2: tests/e2e/**
    result = _classify_e2e_path(path, raw)
    if result is not None:
        impacted.add(result)
    return False, critical_fw_hit


def _resolve_framework_impacted(
    other_fw_files: set[str],
    taxonomy_diff: str | None,
    repo_root: Path,
) -> set[str]:
    """Resolve impacted paths for non-critical ``framework/`` file changes.

    Parses newly added ``CATEGORY_PROFILES`` keys from ``taxonomy.py`` when present;
    falls back to a smoke run (one test file per e2e area) for remaining files.
    """
    impacted: set[str] = set()
    remaining = other_fw_files
    taxonomy_path = "framework/markers/taxonomy.py"
    if taxonomy_path in other_fw_files and taxonomy_diff is not None:
        new_dirs = _parse_taxonomy_diff(taxonomy_diff)
        impacted.update(new_dirs)
        if new_dirs:
            remaining = other_fw_files - {taxonomy_path}
    if remaining:
        impacted.update(_smoke_paths(repo_root))
    return impacted


def _deduplicate(impacted: set[str]) -> list[str]:
    """Drop individual file paths that are already covered by a directory entry."""
    dirs = {p for p in impacted if p.endswith("/")}
    files = {p for p in impacted if not p.endswith("/")}
    filtered = {f for f in files if not any(f.startswith(d) for d in dirs)}
    return sorted(dirs | filtered)


def detect_impacted_paths(
    changed_files: list[str],
    taxonomy_diff: str | None = None,
    repo_root: Path | None = None,
) -> list[str]:
    """Map a list of changed file paths to the pytest paths that should run.

    Args:
        changed_files: Newline-free paths as produced by ``git diff --name-only``.
        taxonomy_diff: Raw unified diff text for ``framework/markers/taxonomy.py``
            when that file is among the changed files.  Used to detect newly added
            ``CATEGORY_PROFILES`` keys.  Pass ``None`` when unavailable.
        repo_root: Absolute path to the repository root.  Used by ``_smoke_paths``
            to locate ``test_*.py`` files.  Defaults to ``_REPO_ROOT``.

    Returns:
        A deduplicated, sorted list of paths (files or directories) to pass to pytest.
    """
    impacted: set[str] = set()
    full_suite = False
    critical_fw_hit = False
    other_fw_files: set[str] = set()

    for raw in changed_files:
        triggered, critical_fw_hit = _classify_single_file(raw, critical_fw_hit, other_fw_files, impacted)
        if triggered:
            full_suite = True
            break

    # ── Resolve framework change buckets ─────────────────────────────────────
    if critical_fw_hit:
        full_suite = True
    elif other_fw_files and not full_suite:
        root = repo_root if repo_root is not None else _REPO_ROOT
        impacted.update(_resolve_framework_impacted(other_fw_files, taxonomy_diff, root))

    if full_suite:
        return sorted(_ALL_E2E_DIRS)

    return _deduplicate(impacted)


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

    # Fetch the unified diff for taxonomy.py when it appears in the changed set.
    # BASE_SHA and HEAD_SHA are exported by the detect_changes job in pr-test-run.yml.
    taxonomy_diff: str | None = None
    fw_files = [f.strip() for f in changed_files if f.strip().startswith("framework/")]
    if "framework/markers/taxonomy.py" in fw_files:
        base_sha = os.environ.get("BASE_SHA", "").strip()
        head_sha = os.environ.get("HEAD_SHA", "HEAD").strip()
        if base_sha:
            import subprocess

            result = subprocess.run(
                ["git", "diff", base_sha, head_sha, "--", "framework/markers/taxonomy.py"],
                capture_output=True,
                text=True,
                check=False,
            )
            taxonomy_diff = result.stdout

    paths = detect_impacted_paths(changed_files, taxonomy_diff=taxonomy_diff, repo_root=_REPO_ROOT)

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
