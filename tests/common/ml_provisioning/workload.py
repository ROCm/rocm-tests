# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Helpers for reporting ML workload outcomes."""

from __future__ import annotations

from framework.common.helpers import ExecutionResult

_INVALID_KERNEL_IMAGE_MARKERS = (
    "device kernel image is invalid",
    "hipErrorInvalidImage",
)


def workload_failure_detail(result: ExecutionResult, workload_name: str, metadata: dict | None = None) -> str:
    """Return extra diagnostic text for a failed ROCm ML workload.

    Provisioning validates that the ML framework is importable and ROCm-enabled.
    Workload tests exercise deeper library-specific GPU paths, such as PyTorch
    operators dispatching into hipBLASLt.
    """
    combined_output = f"{result.stdout}\n{result.stderr}"
    if not any(marker in combined_output for marker in _INVALID_KERNEL_IMAGE_MARKERS):
        return ""

    details = _format_metadata(metadata or {})
    return (
        f"\nDiagnostic: {workload_name} reached a real HIP kernel launch and failed with "
        "hipErrorInvalidImage. Framework installation/version validation succeeded, but the installed "
        "ROCm device wheel or loaded ROCm runtime libraries produced an invalid GPU code object "
        "for this GPU. This is expected to FAIL, not skip, because a selected gfx-specific ROCm "
        f"wheel should run on the matching GPU family.{details}"
    )


def _format_metadata(metadata: dict) -> str:
    validation = metadata.get("validation", metadata)
    spec = metadata.get("spec", {})
    pieces = [
        ("torch", validation.get("torch_version") or spec.get("torch")),
        ("torch_hip", validation.get("torch_hip")),
        ("device_count", validation.get("device_count")),
        ("torchvision", spec.get("torchvision")),
        ("torchaudio", spec.get("torchaudio")),
    ]
    rendered = [f"{name}={value}" for name, value in pieces if value not in (None, "")]
    return f" ({', '.join(rendered)})" if rendered else ""
