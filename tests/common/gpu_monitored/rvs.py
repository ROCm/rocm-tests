# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS (ROCm Validation Suite) detection helpers.

RVS is shipped separately from the ROCm tarball, but is expected to be
preinstalled (its ``rvs`` binary and config tree land under ``rocm_root``)
before the suite runs. This module only *locates* that preinstalled RVS --
it does NOT build RVS from source or install a package. The stack under
test is treated as immutable (see AGENTS.md container conventions); if RVS
is absent the ``rvs_*`` tests surface ``BUILD_FAILED`` so CI flags the
missing component instead of silently source-building a mismatched RVS.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
import re

from tests.common.gpu_monitored.config import Config

from .shared_builder import SharedToolBuilder

# Vendored copy of ROCmTest's device->config mapping
# (ROCmTest/tests/TOOLS/RVS/rvs_config_mapping.csv). Keep in sync when
# upstream RVS adds new silicon / config dirs.
_CONFIG_MAP_PATH = Path(__file__).with_name("rvs_config_mapping.csv")
_config_map_cache: dict[str, dict[str, str]] | None = None


def is_installed(rocm_root: Path) -> bool:
    rvs = rocm_root / "bin" / "rvs"
    conf = rocm_root / "share" / "rocm-validation-suite" / "conf"
    return rvs.is_file() and os.access(rvs, os.X_OK) and conf.is_dir()


def find_bin(config: Config) -> Path | None:
    override = os.environ.get("ROCM_TEST_RVS_BIN", "").strip()
    if override:
        p = Path(override)
        if p.is_file() and os.access(p, os.X_OK):
            return p
    if is_installed(config.rocm_root):
        return config.rocm_root / "bin" / "rvs"
    return None


# ---------------------------------------------------------------------------
# ROCmTest-compatible config resolution (rvs_config_mapping.csv)
# ---------------------------------------------------------------------------
# This mirrors ROCmTest's ``RocmValidationSuite`` config lookup exactly:
#   * key on the PCI ``<device_id>_<revision>`` (DID_RID);
#   * per-test cell in the CSV names the relative conf subdir:
#       ``./``        -> generic top-level ``conf/<test>.conf``
#       ``./MI300A/`` -> ``conf/MI300A/<test>.conf``
#       (empty)       -> test not applicable for this device -> UNSUPPORTED
#   * device not present in the CSV at all               -> UNSUPPORTED
#   * mapped config file missing on disk                 -> FAIL
# Path components are resolved case-insensitively, same as ROCmTest.


class ConfigMapUnavailableError(Exception):
    """The vendored ``rvs_config_mapping.csv`` is missing or unusable.

    Distinct from "this device has no config": the CSV ships inside this
    package, so anything wrong with the *file* -- absent, unreadable,
    truncated, mis-parsed, or missing the columns the shipped tests read --
    is a harness packaging fault: a bad checkout, an incomplete copy to the
    test host, a botched re-sync. Reporting that as ``UNSUPPORTED`` gave a
    zero exit and a green CI run for every RVS test on every device, which
    is exactly the "absence of evidence read as success" pattern this suite
    is meant to remove. Callers turn this into a FAIL instead.

    An *empty cell* in a well-formed mapping is the opposite case: the file
    is intact and is telling us this component does not apply to this
    device. That stays ``UNSUPPORTED``.
    """


class RvsInstallIncompleteError(Exception):
    """The preinstalled RVS binary and configuration tree disagree."""


# The first two columns are positional -- the loader keys on row[0] and
# ignores row[1] -- so their header names are the only evidence that the
# file's column order still matches what this parser assumes. A re-sync that
# moved Device into column 0 would key every row on a model name, miss every
# device-id lookup, and degrade the whole fleet to UNSUPPORTED.
_CONFIG_MAP_POSITIONAL_HEADERS = ("did_rid", "device")

# Every device id in the canonical CSV is a PCI device id and revision as
# ``<4 hex>_<2 hex>``, matched after lower-casing. Enforced so a corrupted
# id cannot be accepted as a dict key -- which would leave the device it was
# meant to describe unmatched, and therefore UNSUPPORTED.
_DID_RID_RE = re.compile(r"^[0-9a-f]{4}_[0-9a-f]{2}$")


def _load_config_map() -> dict[str, dict[str, str]]:  # noqa: C901
    """Parse the vendored ``rvs_config_mapping.csv`` into
    ``{device_id_lower: {component_lower: cell}}`` (cached).

    Raises ``ConfigMapUnavailableError`` for any structural problem with the
    file; see that exception for why this is not ``UNSUPPORTED``.
    """
    global _config_map_cache
    if _config_map_cache is not None:
        return _config_map_cache
    mapping: dict[str, dict[str, str]] = {}
    try:
        with open(_CONFIG_MAP_PATH, newline="") as fh:
            # strict=True so unbalanced quoting raises instead of silently
            # shifting cell boundaries, which would misalign every component
            # column on that row and could select another part's config.
            reader = csv.reader(fh, strict=True)
            headers = next(reader)
            actual = tuple(h.strip().lower() for h in headers[:2])
            if actual != _CONFIG_MAP_POSITIONAL_HEADERS:
                raise ConfigMapUnavailableError(
                    f"config mapping {_CONFIG_MAP_PATH} starts with columns "
                    f"{actual!r}; expected "
                    f"{_CONFIG_MAP_POSITIONAL_HEADERS!r}. The first two "
                    f"columns are positional, so a different order would key "
                    f"every row on the wrong field"
                )
            components = [h.strip().lower() for h in headers[2:]]
            if not components:
                raise ConfigMapUnavailableError(
                    f"config mapping {_CONFIG_MAP_PATH} has no component "
                    f"columns after DID_RID,Device; no test could resolve a "
                    f"config from it"
                )
            # Rows are folded into a dict keyed on these names, so a repeated
            # column would let the rightmost cell silently overwrite the
            # others -- a device could then be pointed at a different config
            # than the mapping appears to specify, with nothing logged. A
            # blank name would likewise create an unaddressable ``''``
            # component. Neither can be resolved sensibly, so reject both
            # rather than pick a winner.
            blank_count = sum(1 for c in components if not c)
            if blank_count:
                raise ConfigMapUnavailableError(
                    f"config mapping {_CONFIG_MAP_PATH} has {blank_count} "
                    f"unnamed component column(s); every column after "
                    f"DID_RID,Device must carry the component name it maps"
                )
            duplicates = sorted({c for c in components if components.count(c) > 1})
            if duplicates:
                raise ConfigMapUnavailableError(
                    f"config mapping {_CONFIG_MAP_PATH} repeats component "
                    f"column(s) {duplicates!r}; the rightmost cell would "
                    f"silently win and could select a different config than "
                    f"the mapping appears to specify"
                )
            width = len(headers)
            for line_no, row in enumerate(reader, start=2):
                # A wholly blank line is a formatting artifact (trailing
                # newline, spacer) and is skipped. A row with *content* but no
                # device id is not: it used to be skipped just as quietly, so
                # if the corrupted row happened to be the one for the GPU
                # under test, the mapping still loaded, that device looked
                # absent, and the test reported a zero-exit UNSUPPORTED for a
                # broken vendored file.
                if not row or not any(cell.strip() for cell in row):
                    continue
                if not row[0].strip():
                    raise ConfigMapUnavailableError(
                        f"config mapping {_CONFIG_MAP_PATH} line {line_no} has "
                        f"no device id but is not empty; a row missing its "
                        f"DID_RID would make that device look absent and "
                        f"report UNSUPPORTED"
                    )
                if len(row) != width:
                    raise ConfigMapUnavailableError(
                        f"config mapping {_CONFIG_MAP_PATH} line {line_no} has "
                        f"{len(row)} field(s), expected {width}; a truncated "
                        f"row would read as empty cells and report the "
                        f"affected tests UNSUPPORTED"
                    )
                did = row[0].strip().lower()
                if not _DID_RID_RE.match(did):
                    # An unrecognisable device id is accepted as a dict key
                    # otherwise, so the real device it was meant to describe
                    # goes missing and resolves to UNSUPPORTED. Every row in
                    # the canonical CSV is <4 hex>_<2 hex>.
                    raise ConfigMapUnavailableError(
                        f"config mapping {_CONFIG_MAP_PATH} line {line_no} has "
                        f"device id {row[0].strip()!r}, which is not the "
                        f"expected <4 hex>_<2 hex> form; the device it "
                        f"describes would look absent and report UNSUPPORTED"
                    )
                if did in mapping:
                    # Two rows for one device means the vendored copy needs
                    # re-syncing, and picking either one risks running a config
                    # intended for a different part -- the wrong-config match
                    # this selector exists to prevent. This used to keep the
                    # first row and print a warning, which is the one CSV
                    # integrity fault here that degraded instead of failing:
                    # every sibling check (blank or repeated component column,
                    # malformed device id, wrong header, bad row width) raises.
                    # A warning in a CI log is exactly the signal this file
                    # refuses to rely on elsewhere, so raise here too.
                    raise ConfigMapUnavailableError(
                        f"config mapping {_CONFIG_MAP_PATH} repeats device id "
                        f"{did!r} (line {line_no}); one of the two rows would "
                        f"decide which config this device runs, so the vendored "
                        f"copy needs re-syncing"
                    )
                cells = row[2:]
                mapping[did] = {components[i]: cells[i].strip() for i in range(len(components))}
    except FileNotFoundError as e:
        # The CSV is vendored in this package, so its absence says nothing
        # about the host. Refuse to resolve rather than degrade the whole
        # fleet to a zero-exit UNSUPPORTED.
        raise ConfigMapUnavailableError(
            f"config mapping not found at {_CONFIG_MAP_PATH}; the vendored "
            f"rvs_config_mapping.csv is missing from the harness"
        ) from e
    except StopIteration as e:
        raise ConfigMapUnavailableError(
            f"config mapping {_CONFIG_MAP_PATH} is empty (no header row); "
            f"the vendored rvs_config_mapping.csv is truncated"
        ) from e
    except csv.Error as e:
        raise ConfigMapUnavailableError(
            f"config mapping {_CONFIG_MAP_PATH} could not be parsed ({e}); "
            f"the vendored rvs_config_mapping.csv is malformed"
        ) from e
    except OSError as e:
        raise ConfigMapUnavailableError(f"config mapping {_CONFIG_MAP_PATH} could not be read ({e})") from e
    if not mapping:
        raise ConfigMapUnavailableError(
            f"config mapping {_CONFIG_MAP_PATH} has a header but no device "
            f"rows; the vendored rvs_config_mapping.csv is truncated"
        )
    _config_map_cache = mapping
    return mapping


def ensure_config_map(component: str) -> None:
    """Check the mapping is usable for ``component`` before resolving.

    Called up front by the RVS tests so a packaging fault is reported even
    on the legacy GPU-name resolution path, which never consults the
    mapping and would otherwise run a generic config while the shipped
    mapping was missing entirely.
    """
    mapping = _load_config_map()
    if not any(component in row for row in mapping.values()):
        raise ConfigMapUnavailableError(
            f"config mapping {_CONFIG_MAP_PATH} has no {component!r} column; "
            f"a test shipped with this harness cannot resolve a config from it"
        )


def _ci_child(directory: Path, name: str, *, want_dir: bool) -> Path | None:
    """Case-insensitive lookup of a single child entry."""
    try:
        for entry in directory.iterdir():
            if entry.name.lower() != name.lower():
                continue
            if entry.is_dir() if want_dir else entry.is_file():
                return entry
    except OSError:
        return None
    return None


class RvsConfigEscapesRootError(ConfigMapUnavailableError):
    """A mapped config resolves outside the RVS ``conf`` tree.

    Separated from "the file is not there" because the two need different
    messages: a path that escapes containment -- via ``..`` or a symlink --
    means the vendored mapping is corrupt or hostile, whereas an absent file
    means the install is incomplete. Both FAIL, but pointing an operator at
    the wrong one of those wastes the triage.
    """


def _resolve_ci_path(conf_root: Path, subdir: str, conf_name: str) -> Path | None:
    """Resolve ``conf_root/<subdir>/<conf_name>``, exact first then
    case-insensitively per path component (mirrors ROCmTest).

    Returns ``None`` when the file simply is not there. Raises
    ``RvsConfigEscapesRootError`` when a candidate exists but resolves outside
    ``conf_root``, so the caller can say which of the two happened.
    """
    subdir_path = Path(subdir)
    conf_path = Path(conf_name)
    if (
        subdir_path.is_absolute()
        or any(part in {".", ".."} for part in subdir_path.parts)
        or conf_path.is_absolute()
        or len(conf_path.parts) != 1
        or conf_path.name in {".", ".."}
    ):
        return None

    try:
        resolved_root = conf_root.resolve()
    except OSError:
        return None

    def _inside_root(candidate: Path) -> bool:
        try:
            candidate.resolve().relative_to(resolved_root)
            return True
        except (OSError, RuntimeError, ValueError):
            return False

    def _accept(candidate: Path) -> Path:
        """Return ``candidate``, or refuse it if it leaves ``conf_root``."""
        if _inside_root(candidate):
            return candidate
        raise RvsConfigEscapesRootError(
            f"mapped RVS config {candidate} resolves outside {resolved_root}; "
            f"the vendored mapping points out of the RVS conf tree"
        )

    exact = (conf_root / subdir / conf_name) if subdir else (conf_root / conf_name)
    if exact.is_file():
        return _accept(exact)
    cur = conf_root
    for part in [p for p in subdir.split("/") if p]:
        nxt = _ci_child(cur, part, want_dir=True)
        if nxt is None:
            return None
        cur = nxt
    candidate = _ci_child(cur, conf_name, want_dir=False)
    return _accept(candidate) if candidate is not None else None


def _resolve_conf_root(rocm_root: Path) -> Path:
    """Return the RVS ``conf`` tree root, preferring a framework-built install."""
    extra = os.environ.get("ROCM_TEST_RVS_CONF_ROOT", "").strip()
    if extra:
        p = Path(extra)
        if p.is_dir():
            return p
    return rocm_root / "share" / "rocm-validation-suite" / "conf"


def resolve_conf_for_device(
    conf_name: str,
    device_id: str,
    rocm_root: Path,
) -> tuple[Path | None, str | None]:
    """Resolve an RVS config for ``device_id`` via the vendored CSV.

    Returns ``(path, None)`` on success or ``(None, reason)`` when the test
    is UNSUPPORTED for this device (device not mapped or its cell is empty).
    ``conf_name`` is the ``<component>.conf`` file (e.g.
    ``iet_stress.conf``); the CSV column is its ``<component>`` stem.

    Raises ``RvsInstallIncompleteError`` when RVS's conf tree or a non-empty
    mapped config is absent, and ``ConfigMapUnavailableError`` for a damaged
    vendored mapping. These are installation/packaging faults rather than
    statements about this device, so callers turn them into FAIL.
    """
    component = conf_name[:-5] if conf_name.endswith(".conf") else conf_name
    conf_root = _resolve_conf_root(rocm_root)
    if not conf_root.is_dir():
        # The conf tree ships with RVS itself, and callers only get here once
        # find_bin() has located the rvs binary. Binary present but configs
        # absent is a broken or partial install, which is operator-fixable --
        # so it must FAIL like a missing binary already does, not report a
        # zero-exit UNSUPPORTED and green CI. Only statements the *intact*
        # mapping makes about this device stay UNSUPPORTED.
        raise RvsInstallIncompleteError(
            f"RVS conf tree not found at {conf_root} even though the rvs "
            f"binary is present; the RVS install is incomplete"
        )

    mapping = _load_config_map()
    row = mapping.get((device_id or "").lower())
    if row is None:
        return None, (f"device {device_id or 'unknown'} not in RVS config " f"mapping (rvs_config_mapping.csv)")
    if component not in row:
        # A shipped test whose column is absent means the file is wrong, not
        # that the device is unsupported -- see ConfigMapUnavailableError.
        raise ConfigMapUnavailableError(
            f"RVS component {component!r} is not a column in "
            f"{_CONFIG_MAP_PATH.name}; the vendored mapping does not match "
            f"the tests shipped with this harness"
        )
    cell = row[component].strip()
    if not cell:
        return None, (f"{component} not applicable for device {device_id} " f"(empty mapping cell)")

    subdir = cell[2:] if cell.startswith("./") else cell
    subdir = subdir.strip("/")
    subdir_path = Path(subdir)
    if subdir_path.is_absolute() or any(part in {".", ".."} for part in subdir_path.parts):
        raise ConfigMapUnavailableError(
            f"RVS component {component!r} for device {device_id!r} maps to "
            f"unsafe relative path {cell!r} in {_CONFIG_MAP_PATH.name}"
        )
    conf = _resolve_ci_path(conf_root, subdir, conf_name)
    if conf is None:
        loc = f"{subdir}/{conf_name}" if subdir else conf_name
        raise RvsInstallIncompleteError(
            f"mapped RVS config {loc} not found under {conf_root}; the "
            f"non-empty {component!r} mapping requires this installed file"
        )
    return conf, None


def find_conf(config_name: str, *, gpu_only: bool, rocm_root: Path, gpu_conf_dir: str) -> Path | None:
    """Locate an RVS config file in the ROCm installation.

    Prefers the GPU-model-specific config
    (``<conf>/<gpu_conf_dir>/<name>``); when ``gpu_only`` is False, falls
    back to the generic top-level ``<conf>/<name>`` so a GPU without a
    dedicated config still runs (see ``_RvsBased._gpu_only``).
    """
    conf_root = _resolve_conf_root(rocm_root)
    if not conf_root.is_dir():
        return None

    if gpu_conf_dir:
        candidate = conf_root / gpu_conf_dir / config_name
        if candidate.is_file():
            return candidate

    if gpu_only:
        return None

    candidate = conf_root / config_name
    if candidate.is_file():
        return candidate
    return None


def _install_check(rocm_root: Path) -> bool:
    """True when RVS is preinstalled under ``rocm_root`` or exported by pytest."""
    override = os.environ.get("ROCM_TEST_RVS_BIN", "").strip()
    if override:
        p = Path(override)
        if p.is_file() and os.access(p, os.X_OK):
            return True
    return is_installed(rocm_root)


def _missing_install(config: Config) -> bool:
    """``SharedToolBuilder`` build_fn. RVS is shipped separately from the
    ROCm tarball but is expected to be preinstalled, so there is nothing to
    build: when the install check fails we report the missing component and
    let the caller surface ``BUILD_FAILED``. We deliberately do NOT
    clone/cmake a source tree or install a package -- the stack under test
    is immutable.
    """
    print(
        f"  [build] rvs: not found under {config.rocm_root} "
        f"(expected {config.rocm_root}/bin/rvs and "
        f"{config.rocm_root}/share/rocm-validation-suite/conf, or "
        f"ROCM_TEST_RVS_BIN from the RVS pytest fixtures). RVS is "
        f"shipped separately from the ROCm tarball; preinstall it under "
        f"the ROCm root or let tests/e2e/rvs/conftest.py build it."
    )
    return False


_builder = SharedToolBuilder(
    label="rvs",
    install_check=_install_check,
    build_fn=_missing_install,
)


def build(config: Config) -> bool:
    """Verify RVS is present in the ROCm installation (shared between
    ``rvs_iet_stress`` and ``rvs_tst``). Does NOT build from source."""
    return _builder.build(config)
