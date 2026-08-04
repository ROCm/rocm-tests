# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse and normalize PyTorch provisioning specifications.

``mode=auto`` uses the production wheel path: ``multiarch -> family``. PyTorch
installation is wheel-only; ROCm OS package installation remains separate under
``--pre-install rocm=...`` / ``--pre-install pkg=...``. Index URLs are supplied
by the ``[frameworks]`` section of ``rocm-test.toml`` (see
:class:`ChannelConfig`). The literals below are last-resort code defaults only —
never the source of truth for a real run.

Staging channel
---------------
``mode=staging`` installs from the pre-promotion multi-arch index
(``[frameworks].staging_index`` in rocm-test.toml).  It behaves identically to
``mode=multiarch`` except that it points at the staging index URL instead of the
production one.  Wheels on the staging index are pre-release candidates; use this
mode only in environments that explicitly opt in to pre-release validation.
``mode=staging`` is not included in ``AUTO_PIP_ORDER`` so ``mode=auto`` never
selects it automatically — an explicit ``--pre-install pytorch=mode=staging`` is
required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Last-resort code defaults. rocm-test.toml [frameworks] overrides all of these;
# they exist only so the provisioner still works with no config file present.
DEFAULT_MULTIARCH_INDEX = "https://rocm.nightlies.amd.com/whl-multi-arch/"
DEFAULT_STAGING_MULTIARCH_INDEX = "https://rocm.nightlies.amd.com/whl-staging-multi-arch/"
DEFAULT_FAMILY_INDEX_BASE = "https://rocm.nightlies.amd.com/v2"

# Production wheel preference order for ``mode=auto``.
AUTO_PIP_ORDER: tuple[str, ...] = ("multiarch", "family")

# Modes accepted on the CLI / in config.
VALID_MODES = {"auto", "multiarch", "staging", "family"}
# Modes that install pip wheels into a fingerprinted venv.
PIP_MODES = {"multiarch", "staging", "family"}


@dataclass(frozen=True)
class ChannelConfig:
    """Resolved install-channel defaults, sourced from ``[frameworks]`` config.

    Callers build this from ``framework_config.frameworks`` so that no index URL
    or channel order is hardcoded in test or provisioner code.
    """

    multiarch_index: str = DEFAULT_MULTIARCH_INDEX
    family_index_base: str = DEFAULT_FAMILY_INDEX_BASE
    staging_index: str = DEFAULT_STAGING_MULTIARCH_INDEX


def default_channels() -> ChannelConfig:
    """Return the code-default channel configuration (no config file present)."""
    return ChannelConfig()


@dataclass(frozen=True)
class FrameworkSpec:
    """Normalized PyTorch install request."""

    framework: str = "pytorch"
    mode: str = "auto"
    index_url: str = ""
    find_links_url: str = ""
    device: str = ""
    gfx_family: str = ""
    torch: str = ""
    torchvision: str = ""
    torchaudio: str = ""
    version: str = ""  # generic version pin used by non-torch frameworks
    requirements: tuple[str, ...] = field(default_factory=tuple)
    pre: bool = True
    raw: str = ""

    @property
    def is_cli(self) -> bool:
        """Return True when the spec came from ``--pre-install <framework>=...``."""
        return bool(self.raw)


# Back-compat alias for the original PyTorch-only name.
PyTorchInstallSpec = FrameworkSpec


def parse_framework_spec(value: str, framework: str = "pytorch") -> FrameworkSpec:
    """Parse comma-delimited ``key=value`` options from ``--pre-install <framework>=...``.

    Preserves the original ``--pre-install pytorch=mode=multiarch,device=gfx942,torch=...``
    syntax verbatim. PyTorch installation modes are wheel-only.
    """
    fields: dict[str, str] = {}
    for segment in [part.strip() for part in value.split(",") if part.strip()]:
        if "=" not in segment:
            raise ValueError(f"Malformed {framework} install segment {segment!r}; expected key=value")
        key, _, val = segment.partition("=")
        fields[key.strip().lower().replace("-", "_")] = val.strip()

    mode = fields.get("mode", "auto").lower()
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported install mode {mode!r}; expected one of {sorted(VALID_MODES)}")

    pre_raw = fields.get("pre", "true").lower()
    pre = pre_raw not in {"0", "false", "no", "off"}
    requirements = tuple(p.strip() for p in fields.get("requirements", "").split(":") if p.strip())

    return FrameworkSpec(
        framework=framework,
        mode=mode,
        index_url=fields.get("index", fields.get("index_url", "")),
        find_links_url=fields.get("find_links", fields.get("find_links_url", "")),
        device=normalize_device_extra(fields.get("device", fields.get("gfx", ""))),
        gfx_family=fields.get("gfx_family", fields.get("family", "")),
        torch=fields.get("torch", ""),
        torchvision=fields.get("torchvision", ""),
        torchaudio=fields.get("torchaudio", ""),
        version=fields.get("version", ""),
        requirements=requirements,
        pre=pre,
        raw=value,
    )


# Back-compat alias for the original parser name.
parse_pytorch_spec = parse_framework_spec


def auto_spec(framework: str = "pytorch") -> FrameworkSpec:
    """Return the default lazy auto-mode spec for *framework*."""
    return FrameworkSpec(framework=framework, mode="auto")


def normalize_device_extra(value: str) -> str:
    """Normalize ``device-gfx942`` or ``gfx942`` to ``gfx942``."""
    value = value.strip()
    if value.startswith("device-"):
        return value[len("device-") :]
    return value


def device_extra(device: str) -> str:
    """Return the pip extra name for a GFX device target."""
    dev = normalize_device_extra(device)
    return f"device-{dev}" if dev else ""


def gfx_family_for_arch(gfx_arch: str) -> str:
    """Map a concrete GFX arch to TheRock's per-family v2 index suffix.

    Unknown arches fall back to the raw arch string so that explicit
    ``mode=multiarch`` (which does not use the family suffix) still works
    correctly on unlisted hardware.
    """
    arch = normalize_device_extra(gfx_arch).lower()
    # CDNA3: MI300X / MI325X
    if arch == "gfx942":
        return "gfx94X-dcgpu"
    # CDNA3.5: MI350X
    if arch == "gfx950":
        return "gfx950-dcgpu"
    # CDNA2: MI250X / MI250 / MI210
    if arch == "gfx90a":
        return "gfx90a-dcgpu"
    # CDNA1: MI100
    if arch == "gfx908":
        return "gfx908-dcgpu"
    if arch in {"gfx1100", "gfx1101", "gfx1102", "gfx1103"}:
        return "gfx110X-all"
    if arch in {"gfx1200", "gfx1201"}:
        return "gfx120X-all"
    if arch == "gfx1151":
        return "gfx1151"
    return arch
