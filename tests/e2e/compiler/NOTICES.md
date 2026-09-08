# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, some of these scripts clone, build, and run
external projects that carry their own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the `HIP-Examples` or `rocm-examples` source trees
or any built binaries; the test fixtures obtain them from upstream at runtime into
the gitignored `output/` build tree. If any downstream packaging flow redistributes
the cloned source or built binaries, that redistribution must retain the upstream
copyright notices, license terms, and disclaimers.

---

## Third-Party Runtime Dependencies

### 1. HIP-Examples

The `hip_examples_repo` fixture clones
[ROCm/HIP-Examples](https://github.com/ROCm/HIP-Examples) at runtime (ref defaults
to `master`, overridable via `ROCM_TEST_HIP_EXAMPLES_REF`). Three sources from that
checkout are compiled with `hipcc`:

| Source built | Copyright | License |
|---|---|---|
| `vectorAdd/vectoradd_hip.cpp` | Copyright (c) 2015-2016 Advanced Micro Devices, Inc. | MIT |
| `openmp-helloworld/openmp_helloworld.cpp` | Copyright (c) 2020-present Advanced Micro Devices, Inc. | MIT |
| `HIP-Examples-Applications/MatrixMultiplication/MatrixMultiplication.cpp` | Copyright ©2015 Advanced Micro Devices, Inc. | BSD 2-Clause |

**Note on repository-level licensing.** `ROCm/HIP-Examples` carries no top-level
`LICENSE` file, so `assert_license_present()` logs a provenance warning for this
checkout by design rather than failing the session. License terms are instead
declared in per-file headers; the three files this area builds are AMD-copyrighted
and carry the permissive terms listed above, reproduced from their own headers.

**Note on submodules.** `HIP-Examples` declares two third-party submodules in
`.gitmodules` — [mixbench](https://github.com/ekondis/mixbench) and
[GPU-STREAM](https://github.com/UoB-HPC/GPU-STREAM). The `clone_repo` helper does
not initialise submodules, so neither is fetched, built, or executed by this
repository, and neither imposes obligations here. Any future fixture that enables
submodule checkout must add the corresponding notices.

- **Upstream repository:** https://github.com/ROCm/HIP-Examples

#### License — BSD 2-Clause (MatrixMultiplication.cpp, reproduced from its header)

```
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

   Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, this
   list of conditions and the following disclaimer in the documentation and/or
   other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

### 2. rocm-examples

The `rocm_examples_repo` fixture clones
[ROCm/rocm-examples](https://github.com/ROCm/rocm-examples) at runtime (ref
defaults to `amd-mainline`, overridable via `ROCM_TEST_ROCM_EXAMPLES_REF`). The
`hip_basic_spirv_build` factory then builds individual `HIP-Basic/` samples as
self-contained CMake projects targeting SPIR-V
(`-DCMAKE_HIP_ARCHITECTURES=amdgcnspirv`).

- **Copyright:** Copyright (c) Advanced Micro Devices, Inc.
- **License:** MIT License
- **Upstream repository:** https://github.com/ROCm/rocm-examples
- **Upstream license file:** https://github.com/ROCm/rocm-examples/blob/amd-staging/LICENSE.md

---

### 3. ROCm toolchain and runtime

Binaries in this directory are compiled by `hipcc` from the ROCm installation on
the target host and link against that install's HIP runtime. The ROCm stack is not
distributed by this repository and retains the license terms shipped with the
installation.

---

## Redistribution Guidance

The `rocm-tests` files in this directory are MIT-licensed first-party test code.
The `HIP-Examples` and `rocm-examples` checkouts and everything built from them are
runtime artifacts under `output/`, not vendored source in this repository. If a
release, container image, build cache, or test artifact bundle includes that cloned
source or the binaries built from it, include the corresponding upstream license
terms with the distributed material — the per-file headers for `HIP-Examples`
(which has no repository-level license file) and `LICENSE.md` for `rocm-examples`.

---

## First-Party Test Code

`conftest.py`, the `test_*.py` files, and the sources under
`tests/e2e/compiler/src/` in this repository are original AMD-authored code,
copyright Advanced Micro Devices, Inc., licensed under the MIT License (the same
license as the parent repository). They do not derive from `HIP-Examples` or
`rocm-examples` source code.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. It is an engineering-compliance summary, not
legal advice; final sign-off for any product distribution should come from AMD
OSS/legal review. For questions about licensing, consult the upstream repositories
linked above.*
