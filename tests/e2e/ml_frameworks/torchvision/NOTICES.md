# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, these scripts may clone, build, and run
external projects that carry their own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the torchvision source tree or built ops; the test
fixture obtains them from upstream at runtime. The upstream checkout retains its own
`LICENSE`. If any downstream packaging flow redistributes the cloned source or built
binaries, that redistribution must retain the upstream copyright notices, license
terms, and disclaimers.

---

## Third-Party Runtime Dependencies

### 1. TorchVision (ROCm)

This test suite clones the ROCm torchvision fork at runtime, builds its in-tree
C++/HIP operators (`setup.py build_ext --inplace`), and runs the cuda-tagged
functional/transforms tensor UT suites against them.

`ROCm/vision` is the ROCm variant of PyTorch's torchvision. It is a fork of
[pytorch/vision](https://github.com/pytorch/vision); the AMD modifications and the
original PyTorch source are both governed by the same BSD 3-Clause license.

- **Fork parent:** https://github.com/pytorch/vision
- **Upstream repository:** https://github.com/ROCm/vision
- **Upstream license file:** https://github.com/ROCm/vision/blob/main/LICENSE
- **Original Work:** Copyright (c) Soumith Chintala 2016. All rights reserved.
- **Modifications:** Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
- **License:** BSD 3-Clause License

#### License — BSD 3-Clause

```
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

The torchvision ops are built as a PyTorch extension and run against the PyTorch
build and ROCm compute libraries (HIP runtime, hipcc, and the compute the
resize/affine/warp kernels rely on) supplied by the container image or host install
used at runtime.

These are separate runtime dependencies with their own licenses and are **not**
distributed within this repository:

- [PyTorch](https://github.com/pytorch/pytorch) — BSD 3-Clause License
- ROCm compute libraries — see the license shipped with the installed ROCm stack

---

## Redistribution Guidance

The `rocm-tests` source files in this directory are MIT-licensed first-party test
code. The torchvision checkout and built ops are runtime artifacts, not vendored
source in this repository. If a release, container image, cache, or test artifact
bundle includes the cloned torchvision source or its build outputs, include the
corresponding upstream `LICENSE` file with that distributed material.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. For questions about licensing, consult the
upstream repositories linked above.*
