# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
markers_plugin.py -- CATEGORY_PROFILES marker injection at collection time.

Reads CATEGORY_PROFILES from taxonomy.py. For each test item, injects missing
required-dimension markers based on the item's file path prefix. Function-level
markers always win; this plugin only fills gaps. Must be loaded FIRST in
pytest_plugins (markers_plugin.py → all other plugins).
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
