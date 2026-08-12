# Third-Party Notices — CRIU recovery tests

The tests under `tests/e2e/recovery/criu/` **fetch, build, and run** third-party open-source
projects **at test run time**. No project's source or binaries are vendored, committed, or
redistributed as part of the rocm-tests repository — clones land in the gitignored `output/`
build directory (cuda_memtest, pytorch/examples). (CRIU, the checkpoint/restore tool these tests
drive, is covered separately by `tests/common/criu/NOTICES.md`.)

This file documents each component, exactly how it is used, and the resulting license
obligations. It is an engineering-compliance summary, not legal advice; final sign-off for any
product distribution should come from AMD OSS/legal review.

---

## Components and how they are used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| cuda_memtest | https://github.com/ComputationalRadiationPhysics/cuda_memtest | commit `0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d` | University of Illinois/NCSA Open Source License (permissive) |
| pytorch/examples (MNIST) | https://github.com/pytorch/examples | latest `main` (shallow clone) | BSD-3-Clause (permissive) |

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

---

## Obligations assessment

- **No redistribution.** rocm-tests does not ship either project's code or binaries; both are
  obtained at runtime from their upstream repositories. NCSA and BSD-3-Clause obligations attach
  to *distribution*, which does not occur here.
- **cuda_memtest (NCSA)** permits use, modification, and redistribution with attribution. The
  hipify/`sed` modifications are allowed; the built binary is not redistributed.
- **pytorch/examples (BSD-3-Clause)** is used unmodified via arm's-length subprocess invocation
  and is not redistributed. BSD-3-Clause's retain-the-notice conditions trigger only on
  *redistribution*, which does not occur here.
- **Modification imposes no obligation.** NCSA is permissive: modifying the sources (hipify +
  `sed`) requires nothing on its own — it does **not** require marking changed files or disclosing
  the diff. NCSA's retain-the-notice conditions trigger only on *redistribution*, which does not
  occur.
- **If distribution is ever added** (e.g. bundling the sources or built binaries): retain the NCSA
  copyright notices and disclaimer for cuda_memtest and the BSD-3-Clause notice for
  pytorch/examples. This is out of scope for the current runtime-fetch model.

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
