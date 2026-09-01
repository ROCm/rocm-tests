# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Node-local PyTorch provisioning.

Public surface:
    - ``provision_framework`` / ``provision_pytorch`` — core provisioning entry.
    - ``ensure_framework_env`` / ``ensure_pytorch_env`` / ``torch_python`` — fixtures helpers.
    - ``parse_framework_spec`` / ``parse_pytorch_spec`` — CLI ``--pre-install`` parser.
"""

from __future__ import annotations

from .fixtures import ensure_framework_env, ensure_pytorch_env, torch_python
from .provisioner import (
    FrameworkProvisionResult,
    PyTorchProvisionResult,
    provision_framework,
    provision_pytorch,
    result_from_dict,
    result_to_dict,
)
from .spec import (
    ChannelConfig,
    FrameworkSpec,
    PyTorchInstallSpec,
    auto_spec,
    parse_framework_spec,
    parse_pytorch_spec,
)

__all__ = [
    "ChannelConfig",
    "FrameworkProvisionResult",
    "FrameworkSpec",
    "PyTorchInstallSpec",
    "PyTorchProvisionResult",
    "auto_spec",
    "ensure_framework_env",
    "ensure_pytorch_env",
    "parse_framework_spec",
    "parse_pytorch_spec",
    "provision_framework",
    "provision_pytorch",
    "result_from_dict",
    "result_to_dict",
    "torch_python",
]
