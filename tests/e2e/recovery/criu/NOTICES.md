# Third-Party Notices — CRIU recovery tests

The tests under `tests/e2e/recovery/criu/` **fetch, build, and run** third-party open-source
projects **at test run time**. No project's source or binaries are vendored, committed, or
redistributed as part of the rocm-tests repository — clones land in the gitignored `output/`
build directory (cuda_memtest, pytorch/examples, RAJAPerf, Kokkos). (CRIU, the checkpoint/restore
tool these tests drive, is covered separately by `tests/common/criu/NOTICES.md`.)

This file documents each component, exactly how it is used, and the resulting license
obligations. It is an engineering-compliance summary, not legal advice; final sign-off for any
product distribution should come from AMD OSS/legal review.

---

## Components and how they are used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| cuda_memtest | https://github.com/ComputationalRadiationPhysics/cuda_memtest | commit `0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d` | University of Illinois/NCSA Open Source License (permissive) |
| pytorch/examples (MNIST) | https://github.com/pytorch/examples | latest `main` (shallow clone) | BSD-3-Clause (permissive) |
| RAJAPerf | https://github.com/LLNL/rajaperf | default branch (pinnable via `ROCM_TEST_RAJAPERF_REF`) | BSD-3-Clause (permissive) |
| Kokkos | https://github.com/kokkos/kokkos | tag `4.2.01` (pinnable via `ROCM_TEST_KOKKOS_REF`) | Apache-2.0 WITH LLVM-exception (permissive) |

### cuda_memtest
1. `git clone` at the pinned commit into `output/test-binaries/recovery/cuda_memtest`.
2. **Modify** the sources: `hipify-perl` (CUDA → HIP) plus a one-line `sed` patch to
   `hipHostGetDevicePointer`.
3. Build a standalone binary with `hipcc`.
4. Run the binary as a subprocess.

### pytorch/examples (MNIST)
1. Shallow `git clone` of the default branch into
   `output/test-binaries/recovery/pyt_examples/examples`.
2. **Not modified** and **not built.** `examples/mnist/main.py` is run as a subprocess using
   the test container's **ambient** ROCm PyTorch (no `pip install`, no `requirements.txt`).
3. Used only to produce a live training process that CRIU checkpoints and restores.

### RAJAPerf
1. `git clone --recursive` into `output/test-binaries/recovery/rajaperf` (default branch, or the
   ref pinned by `ROCM_TEST_RAJAPERF_REF`). The recursive clone also fetches RAJAPerf's own
   submodules (RAJA, BLT, camp, desul, kokkos), each under its own permissive license
   (predominantly BSD-3-Clause; see each submodule's `LICENSE`).
2. Build **unmodified** with CMake + `make` (HIP enabled, static libraries). The sources are
   not patched.
3. Run the built `raja-perf.exe` binary as a subprocess, checkpoint/restore it with CRIU.

### Kokkos
1. `git clone` tag `4.2.01` into `output/external/recovery/kokkos` (pin with
   `ROCM_TEST_KOKKOS_REF`; override the URL with `ROCM_TEST_KOKKOS_URL`).
2. Build only the performance-benchmark target with CMake (`hipcc`, HIP backend, benchmarks
   enabled) into a `build/` tree inside the checkout. The Kokkos sources are **not modified**.
3. Run the built `Kokkos_PerformanceTest_Benchmark` binary as a subprocess and
   checkpoint/restore it with CRIU.

---

## Obligations assessment

- **No redistribution.** rocm-tests does not ship any of these projects' code or binaries; all are
  obtained at runtime from their upstream repositories. NCSA, BSD-3-Clause, and Apache-2.0
  obligations attach to *distribution*, which does not occur here.
- **cuda_memtest (NCSA)** permits use, modification, and redistribution with attribution. The
  hipify/`sed` modifications are allowed; the built binary is not redistributed.
- **pytorch/examples (BSD-3-Clause)** is used unmodified via arm's-length subprocess invocation
  and is not redistributed. BSD-3-Clause's retain-the-notice conditions trigger only on
  *redistribution*, which does not occur here.
- **RAJAPerf (BSD-3-Clause)** permits use, modification, and redistribution. It is built
  **unmodified** and its binary is not redistributed, so no obligation is triggered.
- **Kokkos (Apache-2.0 WITH LLVM-exception)** permits use, modification, and redistribution with
  attribution. Kokkos is built **unmodified** and its binary is not redistributed, so the only
  attribution/notice obligations would attach on *distribution*, which does not occur here.
- **Modification imposes no obligation.** NCSA, BSD-3-Clause, and Apache-2.0 are permissive:
  modifying sources (cuda_memtest's hipify + `sed`) requires nothing on its own — it does **not**
  require marking changed files or disclosing the diff. Their retain-the-notice conditions trigger
  only on *redistribution*, which does not occur. pytorch/examples, RAJAPerf, and Kokkos are not
  modified.
- **If distribution is ever added** (e.g. bundling the sources or built binaries): retain the NCSA
  copyright notices and disclaimer for cuda_memtest; retain the BSD-3-Clause notice for
  pytorch/examples and for RAJAPerf (and its submodules) with the no-endorsement clause; and retain
  the Apache-2.0 license, `NOTICE` file, and attribution for Kokkos. This is out of scope for the
  current runtime-fetch model.

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

### pytorch/examples — BSD 3-Clause License
```
Copyright (c) 2017, Pytorch contributors. All rights reserved.
Licensed under the BSD 3-Clause License.
```
Full license text: `LICENSE` in the pytorch/examples repository.

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

### Kokkos — Apache License 2.0 WITH LLVM-exception
```
Copyright the Kokkos authors / National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights.
Licensed under the Apache License, Version 2.0 with LLVM exceptions.
```
Full license text: `LICENSE` in the Kokkos repository.
