# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
markers_plugin.py -- Apply per-directory category profiles at collection time.

Reads ``CATEGORY_PROFILES`` from ``framework.markers.taxonomy`` and injects
profile markers onto collected test items.  The injection is purely additive:
a profile marker is only added when the test function has no existing marker
in that dimension, so function-level markers always win (overrides are silent).

Category profiles centralise the "what hardware / layer / CI gate" answer for
an entire test directory.  Individual test functions then only need to declare
their test-specific markers (e.g. ``runtime.medium``).

The longest matching path prefix in ``CATEGORY_PROFILES`` wins when a test
file lives under a nested sub-path.

This plugin is registered in the root ``conftest.py`` ``pytest_plugins`` list
and runs before the sharding / remote-node plugins sort the collected items.
"""

from __future__ import annotations

import pathlib

import pytest

from framework.markers.taxonomy import CATEGORY_PROFILES


def pytest_collection_modifyitems(  # pylint: disable=unused-argument
    session: pytest.Session,
    config: pytest.Config,
    items: list,
) -> None:
    """Inject category-profile markers onto collected test items.

    Args:
        session: Active pytest session (unused; required by hook spec).
        config:  Active pytest config (unused; required by hook spec).
        items:   Collected test items, modified in place.
    """
    _apply_profiles(items)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_profiles(items: list) -> None:
    """Add profile markers to each item whose path matches a category profile."""
    cwd = pathlib.Path.cwd()
    for item in items:
        profile = _matching_profile(item, cwd)
        if not profile:
            continue
        # Collect dimensions already covered by existing markers on this item.
        covered = {m.name.split(".", 1)[0] for m in item.iter_markers() if m.name and "." in m.name}
        for marker_str in profile:
            dim = marker_str.split(".", 1)[0]
            if dim not in covered:
                item.add_marker(getattr(pytest.mark, marker_str))


def _matching_profile(item: pytest.Item, cwd: pathlib.Path) -> list[str]:
    """Return the profile marker list for the longest matching path prefix.

    Args:
        item: A collected pytest test item.
        cwd:  Repository root (current working directory when pytest was invoked).

    Returns:
        List of ``"dim.val"`` strings, or empty list if no profile matches.
    """
    try:
        rel = pathlib.Path(str(item.fspath)).relative_to(cwd)
    except ValueError:
        return []
    rel_str = rel.as_posix()
    # Choose the longest matching prefix (more specific directories win).
    match = ""
    for prefix in CATEGORY_PROFILES:
        if rel_str.startswith(prefix) and len(prefix) > len(match):
            match = prefix
    return CATEGORY_PROFILES.get(match, [])
