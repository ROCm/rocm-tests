# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.rocm.libs -- Version-agnostic ROCm CLI wrappers.

Contains the concrete module implementations:

    amd_smi  -- AMD SMI query helpers with multi-schema JSON support.
    hip      -- HIP runtime version and device probing.
    rccl     -- RCCL collective benchmark wrappers.
    stack    -- Full ROCm stack summary and version gating.
"""
