# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-GPU RVS module qualification matrix.

``rvs_config_mapping.csv`` carries one row per ``<device>_<revision>`` key and
one column per RVS module. A cell says two things at once: whether the module is
qualified on that GPU at all, and which config directory it should read.

    ""            module is not qualified on this GPU
    "./"          module uses the config at the top of the conf tree
    "./MI350X"    module uses the config under that subdirectory

Directory cells are written inconsistently in the source data (``./MI210`` next
to ``./MI300X/``), so they are normalised on load.
"""

from __future__ import annotations

import csv
import functools
import logging
import pathlib

logger = logging.getLogger(__name__)

_MAPPING_CSV = pathlib.Path(__file__).with_name("rvs_config_mapping.csv")

# The first two columns identify the device; every column after them is a module.
_KEY_COLUMN = "DID_RID"
_NAME_COLUMN = "Device"

#: Returned for a module that is qualified on a GPU but reads the top-level conf.
TOP_LEVEL = ""


@functools.lru_cache(maxsize=1)
def _mapping() -> dict[str, dict[str, str | None]]:
    """Load the matrix as ``{device key: {module: conf dir or None}}``."""
    table: dict[str, dict[str, str | None]] = {}
    with _MAPPING_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get(_KEY_COLUMN) or "").strip().lower()
            if not key:
                continue
            table[key] = {
                module: _normalise_dir(cell)
                for module, cell in row.items()
                if module not in (_KEY_COLUMN, _NAME_COLUMN)
            }
    logger.debug("Loaded RVS config mapping for %d devices from %s", len(table), _MAPPING_CSV)
    return table


def _normalise_dir(cell: str | None) -> str | None:
    """Turn a raw cell into a conf directory, or ``None`` when not qualified."""
    value = (cell or "").strip().rstrip("/")
    if not value:
        return None
    if value in (".", "./"):
        return TOP_LEVEL
    return value[2:] if value.startswith("./") else value


def is_known_device(device_key: str) -> bool:
    """Whether the matrix has a row for this ``<device>_<revision>`` key."""
    return (device_key or "").lower() in _mapping()


@functools.lru_cache(maxsize=1)
def _names() -> dict[str, str]:
    """Load ``{device key: human-readable name}``."""
    with _MAPPING_CSV.open(newline="") as handle:
        return {
            key: (row.get(_NAME_COLUMN) or "").strip()
            for row in csv.DictReader(handle)
            if (key := (row.get(_KEY_COLUMN) or "").strip().lower())
        }


def device_name(device_key: str) -> str:
    """Human-readable device name for a key, or the key itself when unknown."""
    return _names().get((device_key or "").lower()) or device_key


def conf_dir_for_module(device_key: str, module: str) -> str | None:
    """Return the conf directory *module* reads on this GPU.

    Returns ``TOP_LEVEL`` for a module that reads the top of the conf tree, the
    subdirectory name when it has its own config, and ``None`` when the module
    is not qualified on this GPU or the matrix does not cover it.
    """
    return _mapping().get((device_key or "").lower(), {}).get(module)


def covers_module(module: str) -> bool:
    """Whether *module* is a column in the matrix at all.

    Lets callers tell "this module is not qualified here" apart from "the matrix
    says nothing about this module", which must not be read as a skip.
    """
    return any(module in modules for modules in _mapping().values())
