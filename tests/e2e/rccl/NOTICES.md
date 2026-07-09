# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, these scripts may clone, build, and run
external projects that carry their own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the `rccl-tests` source tree or built binaries;
the test fixture obtains them from upstream at runtime. The upstream checkout
retains its own `LICENSE.txt` and `NOTICES.txt`. If any downstream packaging flow
redistributes the cloned source or built binaries, that redistribution must retain
the upstream copyright notices, license terms, and disclaimers.

---

## Third-Party Runtime Dependencies

### 1. rccl-tests

This test suite may clone `projects/rccl-tests` from the
[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems) monorepo at runtime,
build it using its own Makefile, and execute the resulting `*_perf` binaries to
benchmark RCCL collective communications.

`rccl-tests` is a fork of [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests)
(`nvidia-nccl-tests v2.0.0`). The original NVIDIA source is BSD-licensed, and AMD
modifications are covered by the same BSD-style license. The non-endorsement
clause applies.

- **Original Work:** Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
- **Modifications:** Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
- **License:** BSD 3-Clause License
- **Upstream repository:** https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl-tests
- **Upstream license file:** https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl-tests/LICENSE.txt
- **Upstream notices file:** https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl-tests/NOTICES.txt

#### License — BSD 3-Clause

```
Copyright (c) 2016-2017, NVIDIA CORPORATION. All rights reserved.
Modifications Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
 * Neither the name of NVIDIA CORPORATION, nor the names of their
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#### Upstream origin notice (nccl-tests v2.0.0)

The rccl-tests project documents its own dependency on the original NVIDIA
nccl-tests in its `NOTICES.txt`. Because this repository does not redistribute
`rccl-tests`, the canonical notice remains in the upstream runtime checkout. For
convenience, the upstream notice also includes a DOE funding acknowledgement:

> The U.S. Department of Energy funded the development of this software
> under subcontract 7078610 with Lawrence Berkeley National Laboratory.

---

### 2. RCCL (ROCm Communication Collectives Library)

The `rccl-tests` binaries built above link against `librccl.so` from the installed
ROCm stack. RCCL is itself a fork of NVIDIA's NCCL library and is governed by a
BSD-3-Clause license with NVIDIA, AMD, and Microsoft copyright holders.
Additionally, RCCL incorporates files from the
[NVIDIA Tools Extension SDK (NVTX)](https://github.com/NVIDIA/NVTX).

- **Upstream repository:** https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl
- **Upstream license file:** https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl/LICENSE.txt

The tests do not distribute `librccl.so`; they use the RCCL library supplied by
the ROCm installation on the target host. The upstream RCCL license is referenced
here for attribution and to document the license terms of that runtime dependency.

#### License — BSD 3-Clause

```
Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
Modifications Copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
Modifications Copyright (c) Microsoft Corporation. Licensed under the MIT License.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

*  Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
*  Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
*  Neither the name of NVIDIA CORPORATION, Lawrence Berkeley National
   Laboratory, the U.S. Department of Energy, nor the names of their
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

#### DOE funding acknowledgement (reproduced from RCCL LICENSE.txt)

> The U.S. Department of Energy funded the development of this software
> under subcontract 7078610 with Lawrence Berkeley National Laboratory.

#### Microsoft modifications

Modifications contributed by Microsoft Corporation are licensed under the MIT
License (as noted in the RCCL license header). The MIT License is compatible with
BSD-3-Clause; no additional restrictions apply to downstream users.

---

### 3. MPI (Message Passing Interface) Runtime Environment

Multi-node execution of `rccl-tests` requires an external MPI implementation
installed on the host system. Common implementations include:

- [OpenMPI](https://www.open-mpi.org/) — licensed under the BSD 3-Clause License
- [MPICH](https://www.mpich.org/) — licensed under a permissive MIT-like license

These are separate host-level runtime dependencies and are **not** distributed
within this repository. Refer to each implementation's own license for compliance
requirements.

---

## Redistribution Guidance

The `rocm-tests` source files in this directory are MIT-licensed first-party test
code. The `rccl-tests` checkout and built `*_perf` binaries are runtime artifacts,
not vendored source in this repository. If a release, container image, cache, or
test artifact bundle includes the cloned `rccl-tests` source, `rccl-tests` build
outputs, or RCCL binaries, include the corresponding upstream `LICENSE.txt` and
`NOTICES.txt` files with that distributed material.

---

## First-Party Test Code

The source files under `tests/e2e/rccl/src/` in this repository are original
AMD-authored code.

These files are copyright Advanced Micro Devices, Inc. and are licensed under the
MIT License (the same license as the parent repository). They do not derive from
`nccl-tests` or `rccl-tests` source code.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. For questions about licensing, consult the
upstream repositories linked above.*
