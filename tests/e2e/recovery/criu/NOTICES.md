# Third-Party Notices — cuda_memtest

The tests under `tests/e2e/recovery/criu/` **fetch, modify, build, and run** the third-party
cuda_memtest project **at test time**. Its source and binaries are not vendored, committed, or
redistributed as part of the rocm-tests repository — the clone lands in the gitignored `output/`
build directory. (CRIU, the checkpoint/restore tool these tests drive, is covered separately by
`tests/common/criu/NOTICES.md`.)

This file documents the component, exactly how it is used, and the resulting license obligations.
It is an engineering-compliance summary, not legal advice; final sign-off for any product
distribution should come from AMD OSS/legal review.

---

## Component and how it is used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| cuda_memtest | https://github.com/ComputationalRadiationPhysics/cuda_memtest | commit `0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d` | University of Illinois/NCSA Open Source License (permissive) |

1. `git clone` at the pinned commit into `output/test-binaries/recovery/cuda_memtest`.
2. **Modify** the sources: `hipify-perl` (CUDA → HIP) plus a one-line `sed` patch to
   `hipHostGetDevicePointer`.
3. Build a standalone binary with `hipcc`.
4. Run the binary as a subprocess.

---

## Obligations assessment

- **No redistribution.** rocm-tests does not ship cuda_memtest's code or binaries; it is obtained
  at runtime from upstream. NCSA obligations attach to *redistribution*, which does not occur here.
- **cuda_memtest (NCSA)** permits use, modification, and redistribution with attribution. The
  hipify/`sed` modifications are allowed; the built binary is not redistributed.
- **Modification imposes no obligation.** NCSA is permissive: modifying the sources (hipify +
  `sed`) requires nothing on its own — it does **not** require marking changed files or disclosing
  the diff. NCSA's retain-the-notice conditions trigger only on *redistribution*, which does not
  occur.
- **If distribution is ever added** (e.g. bundling the sources or built binary): retain the NCSA
  copyright notices and disclaimer. This is out of scope for the current runtime-fetch model.

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
