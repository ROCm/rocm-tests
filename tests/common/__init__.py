# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
tests.common -- Shared test data factories and C++ test headers.

All modules here are importable by any test file via:
    from tests.common.<module> import <symbol>

Modules:
    factories -- Test data factories: fake GpuInfo, fake ExecutionResult.

Headers (include/):
    utility.hpp -- C++ helpers shared by HIP kernel sources under tests/e2e/.

Note: ROCm library wrappers (hip, rccl, amd_smi, stack) live in
    framework/rocm/libs/ — import them from there:
        from framework.rocm.libs.hip import get_device_count
        from framework.rocm.libs.amd_smi import query_gpu_temp

Naming rules (IMPORTANT — enforced by norecursedirs in pyproject.toml):
    - Module files:  MUST NOT start with ``test_``. Use descriptive names.
                     pytest skips this directory entirely (norecursedirs).
    - Functions:     MUST NOT start with ``test_``. Use verb_noun form.
                     This prevents accidental collection if norecursedirs is relaxed.
    - Classes:       PascalCase, no ``Test`` prefix: GpuDeviceInfo, RcclResult.

Design rules:
    1. Factories return typed dataclasses or built-ins; no raw dict returns.
    2. Test-layer utilities only — framework-level code (plugins, allocator,
       config) must not import from tests.common.
"""
