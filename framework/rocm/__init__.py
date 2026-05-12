# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
framework.rocm -- Version-agnostic wrappers for ROCm CLI tools.

Abstracts away CLI flag changes and JSON schema differences across ROCm versions
(5.x → 6.x → 7.x) and platforms.  Test code imports from here; it never calls
amd-smi, hipconfig, or rccl-tests directly.

Sub-package:
    libs/  -- Concrete module implementations (amd_smi, hip, rccl, stack).

Usage::

    from framework.rocm.libs.amd_smi import list_devices, query_gpu_temp, require_amd_smi_version
    from framework.rocm.libs.hip import hip_version, require_rocm_version
    from framework.rocm.libs.rccl import check_rccl_available, run_allreduce
    from framework.rocm.libs.stack import stack_summary
"""
