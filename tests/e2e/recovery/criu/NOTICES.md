# Third-Party Notices — CRIU checkpoint/restore tests

The tests under `tests/e2e/recovery/criu/` **fetch, build, and run** third-party
open-source projects **at test time**. None of these projects' source or binaries are vendored,
committed, or redistributed as part of the rocm-tests repository — clones land in the
gitignored `output/` build directory (cuda_memtest, RAJAPerf) or on the test node itself (CRIU).

This file documents each component, exactly how it is used, and the resulting license
obligations. It is an engineering-compliance summary, not legal advice; final sign-off for
any product distribution should come from AMD OSS/legal review.

---

## Components and how they are used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| cuda_memtest | https://github.com/ComputationalRadiationPhysics/cuda_memtest | commit `0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d` | University of Illinois/NCSA Open Source License (permissive) |
| RAJAPerf | https://github.com/LLNL/rajaperf | default branch (pinnable via `ROCM_TEST_RAJAPERF_REF`) | BSD-3-Clause (permissive) |
| CRIU | https://github.com/checkpoint-restore/criu | tag `v4.1` | GPL-2.0-only (LGPL-2.1 for `lib/`) |

### cuda_memtest
1. `git clone` at the pinned commit into `output/test-binaries/recovery/cuda_memtest`.
2. **Modify** the sources: `hipify-perl` (CUDA → HIP) plus a one-line `sed` patch to
   `hipHostGetDevicePointer`.
3. Build a standalone binary with `hipcc`.
4. Run the binary as a subprocess.

### RAJAPerf
1. `git clone --recursive` into `output/test-binaries/recovery/rajaperf` (default branch, or the
   ref pinned by `ROCM_TEST_RAJAPERF_REF`). The recursive clone also fetches RAJAPerf's own
   submodules (RAJA, BLT, camp, desul, kokkos), each under its own permissive license
   (predominantly BSD-3-Clause; see each submodule's `LICENSE`).
2. Build **unmodified** with CMake + `make` (HIP enabled, static libraries). The sources are
   not patched.
3. Run the built `raja-perf.exe` binary as a subprocess, checkpoint/restore it with CRIU.

### CRIU
1. `git clone` tag `v4.1` onto the **test node** (`~/criu_src`, outside the repo) — see
   `scripts/install_criu.py`.
2. Build (`make`) and **install system-wide** (`sudo make install` → `/usr/local/sbin`);
   build the `amdgpu_plugin.so` and copy it to `/usr/lib/criu`.
3. Invoke the `criu` command-line tool (`criu dump` / `criu restore` / `criu check`) as a
   **separate process**. CRIU is **not modified** and **not linked** into rocm-tests code.

---

## Obligations assessment

- **No redistribution.** rocm-tests does not ship any of these projects' code or binaries; all are
  obtained at runtime from their upstream repositories. GPL-2.0, NCSA, and BSD-3-Clause obligations
  attach to *distribution*, which does not occur here.
- **CRIU (GPL-2.0)** is used only via arm's-length CLI invocation. Per GPL-2.0 §0, *"The act of
  running the Program is not restricted."* Running a separate `criu` process is aggregation, not
  a derivative work, so GPL copyleft does not extend to rocm-tests. rocm-tests source remains
  under its own license (`SPDX-License-Identifier: MIT`).
- **cuda_memtest (NCSA)** permits use, modification, and redistribution with attribution. The
  hipify/`sed` modifications are allowed; the built binary is not redistributed.
- **RAJAPerf (BSD-3-Clause)** permits use, modification, and redistribution. It is built
  **unmodified** and its binary is not redistributed, so no obligation is triggered.
- **Modification imposes no obligation.** NCSA and BSD-3-Clause are permissive: modifying sources
  (cuda_memtest's hipify + `sed`) requires nothing on its own — it does **not** require marking
  changed files or disclosing the diff (unlike GPL-2.0 §2(a)). Their retain-the-notice conditions
  trigger only on *redistribution*, which does not occur. RAJAPerf and CRIU are not modified, so
  GPL-2.0 §2(a) never applies either.
- **If distribution is ever added** (e.g. bundling the sources or built binaries): retain the
  NCSA copyright notices and disclaimer for cuda_memtest; retain the BSD-3-Clause copyright notice,
  conditions, and disclaimer for RAJAPerf (and its submodules), and honor the no-endorsement clause;
  and comply with GPL-2.0 source-offer requirements for CRIU. This is out of scope for the current
  runtime-fetch model.

---

## Attribution

### cuda_memtest — University of Illinois/NCSA Open Source License
```
Copyright 2009-2012, University of Illinois. All rights reserved.
Copyright 2013-2019, The developers of PIConGPU at Helmholtz-Zentrum Dresden-Rossendorf

Developed by:
  Innovative Systems Lab, National Center for Supercomputing Applications
Forked and maintained since 2013 by:
  Axel Huebl and Rene Widera, Computational Radiation Physics Group,
  Helmholtz-Zentrum Dresden-Rossendorf
```
Full license text: `LICENSE` in the cuda_memtest repository.

### RAJAPerf — BSD-3-Clause License
```
BSD 3-Clause License

Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
Full license text: `LICENSE` in the RAJAPerf repository. RAJAPerf's submodules (RAJA, BLT, camp,
desul, kokkos) carry their own licenses (predominantly BSD-3-Clause); see each submodule's `LICENSE`.

### CRIU — GNU General Public License, version 2 (LGPL-2.1 for `lib/`)
```
Copyright the CRIU project contributors (checkpoint-restore/criu).
Licensed under GPL-2.0-only; software under lib/ is licensed under LGPL-2.1.
Only version 2 of the GPL applies unless explicitly stated otherwise.
```
Full license text: `COPYING` in the CRIU repository.
