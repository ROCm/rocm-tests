# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""gpu_count.py -- parse and resolve the ``@pytest.mark.gpu_count(...)`` argument.

``gpu_count`` accepts three argument forms:

- ``int``            — an explicit GPU count, e.g. ``gpu_count(4)``.
- ``"ALL"`` / ``"all"`` — reserve *every* GPU available on the target node
  (case-insensitive sentinel).  Resolved against the node's real capacity at
  acquisition time, so the test scales to whatever the host provides.
- ``list[int]``      — reserve ``max(list)`` GPUs in a single run, e.g.
  ``gpu_count([2, 4, 8])`` reserves 8.  (The workload can still sweep the
  smaller widths internally.)

Centralising the parsing here keeps the marker contract identical across the
collection-time scheduler, the NodePool acquisition paths, and the runtime
``requested_gpu_count`` fixture.
"""

from __future__ import annotations

from collections.abc import Sequence

# Case-insensitive sentinel meaning "reserve all GPUs available on the node".
GPU_COUNT_ALL = "all"


def parse_gpu_count(raw: object) -> int | str:
    """Normalise a raw ``gpu_count`` marker argument to an int or the ALL sentinel.

    Args:
        raw: The first positional arg of ``@pytest.mark.gpu_count(...)`` — an int,
             the string ``"ALL"``/``"all"``, a numeric string, or a list/tuple of ints.

    Returns:
        ``GPU_COUNT_ALL`` for the ALL sentinel, otherwise a positive ``int``
        (``max(...)`` for a list).

    Raises:
        ValueError: For an empty list or an unparseable value.
    """
    if isinstance(raw, str):
        token = raw.strip().lower()
        if token == GPU_COUNT_ALL:
            return GPU_COUNT_ALL
        return int(token)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [int(x) for x in raw]
        if not values:
            raise ValueError("@pytest.mark.gpu_count([...]) requires a non-empty list")
        return max(values)
    if not isinstance(raw, (int, float)):
        raise ValueError(f"@pytest.mark.gpu_count expects int, list, or 'all'; got {type(raw).__name__!r}: {raw!r}")
    return int(raw)


def gpu_count_from_marker(marker: object, *, default: int) -> int | str:
    """Return the parsed ``gpu_count`` value for a marker, or *default* when absent.

    Args:
        marker:  The marker object from ``get_closest_marker("gpu_count")`` (or None).
        default: Value to return when the marker is missing or argument-less.

    Returns:
        ``GPU_COUNT_ALL`` or a positive ``int``.
    """
    args = getattr(marker, "args", None)
    if marker is not None and args:
        return parse_gpu_count(args[0])
    return default


def resolve_gpu_count(value: int | str, *, available: int, default: int) -> int:
    """Resolve a parsed ``gpu_count`` to a concrete positive count.

    Args:
        value:     Output of :func:`parse_gpu_count` / :func:`gpu_count_from_marker`
                   (an int or ``GPU_COUNT_ALL``).
        available: GPUs available on the target node (used to resolve ALL).
        default:   Fallback when ALL is requested but capacity is unknown (``available <= 0``).

    Returns:
        A positive integer GPU count.
    """
    if value == GPU_COUNT_ALL:
        return available if available and available > 0 else default
    count = int(value)
    return count if count > 0 else default
