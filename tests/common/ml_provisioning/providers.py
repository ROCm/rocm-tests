# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PyTorch package and validation provider.

The provisioner core owns node discovery, venv management, channel selection,
caching, and validation plumbing. This module supplies the PyTorch-specific
parts:

* ``packages(spec)``       - the wheel names (with pip extras / pins) to install.
* ``primary_package``      - package whose versions drive candidate selection.
* ``companion_packages``   - packages version-matched to the primary wheel.
* ``sanity_snippet(spec)`` - a Python one-shot that prints JSON metadata and
  exits 0 (healthy) / 3 (unhealthy), run on the target node after install.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from .spec import FrameworkSpec, device_extra


@runtime_checkable
class FrameworkProvider(Protocol):
    """Per-framework install + validation strategy."""

    name: str
    primary_package: str
    companion_packages: tuple[str, ...]

    def packages(self, spec: FrameworkSpec) -> list[str]:
        """Return the pip package specifiers to install for *spec*."""

    def sanity_snippet(self, spec: FrameworkSpec) -> str:
        """Return a Python snippet that prints JSON metadata + exits 0/3."""


def _pin(version: str) -> str:
    return f"=={version}" if version else ""


class PyTorchProvider:
    """PyTorch (torch/torchvision/torchaudio) on ROCm."""

    name = "pytorch"
    primary_package = "torch"
    companion_packages: tuple[str, ...] = ("torchvision", "torchaudio")

    def packages(self, spec: FrameworkSpec) -> list[str]:
        """Return torch/vision/audio specifiers.

        ``family`` mode installs plain names against the per-arch v2 index;
        ``multiarch`` mode uses ``[device-gfxNNN]`` pip extras.
        """
        if spec.mode == "family":
            return [
                f"torch{_pin(spec.torch)}",
                f"torchvision{_pin(spec.torchvision)}",
                f"torchaudio{_pin(spec.torchaudio)}",
            ]
        extra = device_extra(spec.device)
        torch_name = f"torch[{extra}]" if extra else "torch"
        vision_name = f"torchvision[{extra}]" if extra else "torchvision"
        return [
            f"{torch_name}{_pin(spec.torch)}",
            f"{vision_name}{_pin(spec.torchvision)}",
            f"torchaudio{_pin(spec.torchaudio)}",
        ]

    def sanity_snippet(self, spec: FrameworkSpec) -> str:
        """Hardened post-install check.

        Beyond a version-string match this asserts ``torch.version.hip`` (a real
        ROCm build), ``torch.cuda.is_available()``, and — when ``transformers``
        is present — that ``transformers.image_utils`` imports (catches the
        torchvision/torchaudio ABI-mismatch footgun).
        """
        expected = json.dumps(spec.torch)
        return (
            "import json\n"
            "meta = {}\n"
            "ok = False\n"
            "try:\n"
            "    import torch\n"
            "    meta['torch_version'] = torch.__version__\n"
            "    meta['torch_hip'] = getattr(torch.version, 'hip', None)\n"
            "    meta['cuda_available'] = bool(torch.cuda.is_available())\n"
            "    if meta['cuda_available']:\n"
            "        meta['device_count'] = torch.cuda.device_count()\n"
            "        meta['device_name'] = torch.cuda.get_device_name(0)\n"
            "        x = torch.empty((1,), device='cuda')\n"
            "        x.fill_(16)\n"
            "        meta['device_smoke'] = float(x.cpu().item())\n"
            "    try:\n"
            "        import transformers  # noqa: F401\n"
            "        import transformers.image_utils  # noqa: F401\n"
            "        meta['transformers_image_utils'] = True\n"
            "    except ImportError:\n"
            "        pass\n"  # transformers optional; only a hard failure if present-but-broken
            "    except Exception as _te:\n"
            "        meta['transformers_error'] = repr(_te)\n"
            "        raise\n"
            f"    _expected = {expected}\n"
            "    meta['version_matches'] = (not _expected) or meta['torch_version'] == _expected\n"
            "    ok = bool(meta['torch_hip']) and meta['cuda_available'] and meta['version_matches']\n"
            "except Exception as _e:\n"
            "    meta['error'] = repr(_e)\n"
            "    ok = False\n"
            "print(json.dumps(meta))\n"
            "raise SystemExit(0 if ok else 3)\n"
        )


def get_provider(framework: str) -> FrameworkProvider:
    """Return the provider for *framework* or raise a clear error."""
    if framework == "pytorch":
        return PyTorchProvider()
    raise ValueError("Only 'pytorch' framework provisioning is currently supported")
