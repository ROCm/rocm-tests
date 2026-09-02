# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, these scripts may clone, build, and run
external projects that carry their own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the Apex source tree or built kernels; the test
fixture obtains them from upstream at runtime. The upstream checkout retains its
own `LICENSE` file. If any downstream packaging flow redistributes the cloned
source or built binaries, that redistribution must retain the upstream copyright
notices, license terms, and disclaimers.

---

## Third-Party Runtime Dependencies

### 1. Apex (ROCm)

This test suite clones the ROCm Apex fork at runtime, builds its fused HIP kernel
extensions, and runs the L0 unit-test suite against them.

`ROCmSoftwarePlatform/apex` (now [ROCm/apex](https://github.com/ROCm/apex)) is the
ROCm variant of NVIDIA's Apex. It is a fork of
[NVIDIA/apex](https://github.com/NVIDIA/apex); the AMD modifications and the
original NVIDIA source are both governed by the same BSD 3-Clause license.

- **Fork parent:** https://github.com/NVIDIA/apex
- **Upstream repository:** https://github.com/ROCm/apex
  (formerly https://github.com/ROCmSoftwarePlatform/apex)
- **Upstream license file:** https://github.com/ROCm/apex/blob/master/LICENSE
- **Original Work:** Copyright (c) NVIDIA CORPORATION. All rights reserved.
- **Modifications:** Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **License:** BSD 3-Clause License

#### License — BSD 3-Clause

```
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

---

### 2. PyTorch and the ROCm Compute Stack

The Apex fused kernels are a PyTorch extension. The L0 suite is compiled and run
against the PyTorch build and the ROCm compute libraries (HIP runtime, hipcc, and
the math libraries the fused MLP / dense / transformer layers depend on) supplied
by the container image or host install used at runtime.

These are separate runtime dependencies with their own licenses and are **not**
distributed within this repository. Refer to each project's own license for
compliance requirements:

- [PyTorch](https://github.com/pytorch/pytorch) — BSD 3-Clause License
- ROCm compute libraries — see the license shipped with the installed ROCm stack

---

## Redistribution Guidance

The `rocm-tests` source files in this directory are MIT-licensed first-party test
code. The Apex checkout and built kernel extensions are runtime artifacts, not
vendored source in this repository. If a release, container image, cache, or test
artifact bundle includes the cloned Apex source or its build outputs, include the
corresponding upstream `LICENSE` file with that distributed material.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. For questions about licensing, consult the
upstream repositories linked above.*
