# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
taxonomy.py -- Marker schema and category profile definitions.

MARKER_SCHEMA: authoritative set of valid marker values per dimension.
CATEGORY_PROFILES: per-directory default marker injection (markers_plugin reads this).
PARAMETRIC_MARKERS: markers that take arguments (gpu_vram, gpu_count, etc.).
Add new values to MARKER_SCHEMA before using them in test files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Marker taxonomy
# ---------------------------------------------------------------------------

MARKER_SCHEMA: dict[str, set[str]] = {
    # Hardware requirement (REQUIRED)
    "hw": {"gpu", "multi_gpu", "cpu_only"},
    # CI gate membership (REQUIRED)
    "ci": {"pr", "nightly", "weekly"},
    # ROCm stack layer under test (REQUIRED)
    "layer": {"runtime", "math_lib"},
    # Expected duration (optional but strongly recommended)
    "runtime": {"fast", "medium", "soak"},
    # Target platform (optional)
    "os": {"linux"},
    # Scenario type (optional)
    "e2e": {"stack", "multinode"},
}

REQUIRED_DIMENSIONS: set[str] = {"hw", "ci", "layer"}

# Parametric markers — accept arguments; not dimensions; no linting enforcement.
# gpu_count(N): minimum GPUs required (used by multi_gpu_fixture and NodePool)
PARAMETRIC_MARKERS: dict[str, str] = {
    "gpu_vram": "Minimum VRAM in GB (@pytest.mark.gpu_vram(16))",
    "gpu_count": "Minimum GPU count per node (@pytest.mark.gpu_count(4))",
    "container_image": "Override container image (@pytest.mark.container_image('rocm/pytorch:6.3'))",
    "gpu_indices": "Exact GPU indices to acquire, bypassing NUMA selection (@pytest.mark.gpu_indices([0, 2]))",
}

# Duration guidance (informational — not enforced programmatically)
DURATION_GUIDANCE: dict[str, str] = {
    "fast": "< 5 minutes    — use with ci.pr",
    "medium": "< 30 minutes   — use with ci.nightly",
    "soak": "hours          — use with ci.weekly",
}

# ---------------------------------------------------------------------------
# Allure label mapping (single source of truth for reports_plugin.py)
# ---------------------------------------------------------------------------

# Maps each marker dimension to the Allure label type it should populate.
# reports_plugin.py reads this instead of hard-coding "if dim == 'hw'" chains.
# To wire a new dimension into Allure: add it here only.
ALLURE_DIMENSION_MAP: dict[str, str] = {
    "hw": "severity",  # allure.dynamic.severity(...)
    "ci": "feature",  # allure.dynamic.feature(...)
    "layer": "story",  # allure.dynamic.story(...)
    "e2e": "epic",  # allure.dynamic.epic(...)
    "os": "tag",  # allure.dynamic.tag(...)
    "runtime": "tag",
}

# Allure severity level for each hw.* value.
# Used by reports_plugin when ALLURE_DIMENSION_MAP[dim] == "severity".
HW_SEVERITY_MAP: dict[str, str] = {
    "gpu": "critical",
    "multi_gpu": "critical",
    "cpu_only": "minor",
}

# ---------------------------------------------------------------------------
# Per-directory category profiles
# ---------------------------------------------------------------------------

# Maps path prefixes (relative to repo root, POSIX forward-slash) to a list of
# "dim.val" marker strings that apply to every test under that path.
#
# Rules:
#   - Profiles are ADDITIVE: a profile marker is injected only when the test
#     function has no existing marker in that dimension (function always wins).
#   - The longest matching prefix wins (no overlap in practice, but safe).
#   - runtime.* is intentionally absent from all profiles — duration varies
#     per test and must be declared explicitly on each function.
#
# Used by:
#   - framework/plugins/markers_plugin.py  (collection-time injection)
#   - framework/markers/linter.py          (profile-aware lint checks)
CATEGORY_PROFILES: dict[str, list[str]] = {
    "tests/e2e/compiler": [
        "hw.gpu",
        "layer.runtime",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
    "tests/e2e/hwq_heuristic": [
        "hw.gpu",
        "layer.runtime",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
    "tests/e2e/hipblaslt": [
        "hw.gpu",
        "layer.math_lib",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
    "tests/e2e/hip_runtime": [
        "hw.gpu",
        "layer.runtime",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
    "tests/e2e/rocm_libs": [
        "hw.gpu",
        "layer.math_lib",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
    "tests/e2e/rocprim": [
        "hw.gpu",
        "layer.math_lib",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
    "tests/e2e/concurrent_collectives": [
        "hw.multi_gpu",
        "layer.math_lib",
        "ci.nightly",
        "e2e.stack",
        "os.linux",
    ],
}
