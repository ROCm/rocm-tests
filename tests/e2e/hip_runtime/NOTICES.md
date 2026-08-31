# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, some of these scripts clone, build, and run
an external project that carries its own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the `hip-tests` source tree or any built binaries;
the test fixture obtains them from upstream at runtime into the gitignored
`output/` build tree. The upstream checkout retains its own license file. If any
downstream packaging flow redistributes the cloned source or built binaries, that
redistribution must retain the upstream copyright notices, license terms, and
disclaimers.

---

## Third-Party Runtime Dependencies

### 1. hip-tests (HIP samples suite)

The `hip_samples_repo` fixture performs a blob-filtered sparse clone of the
`samples/` subtree from [ROCm/hip-tests](https://github.com/ROCm/hip-tests) at
runtime. The `hip_sample_build` and `hip_sample_spirv_build` factories then
configure and build individual samples as self-contained CMake projects (the
SPIR-V variant builds the same sources with `-DCMAKE_HIP_ARCHITECTURES=amdgcnspirv`).
The clone ref defaults to `develop` and is overridable via
`ROCM_TEST_HIP_TESTS_REF`.

Only the `samples/` subtree is fetched; the upstream `catch/` unit-test suite and
its own third-party dependencies are not downloaded by this module.

- **Copyright:** Copyright (C) Advanced Micro Devices, Inc.
- **License:** MIT License
- **Upstream repository:** https://github.com/ROCm/hip-tests
- **Upstream license file:** https://github.com/ROCm/hip-tests/blob/develop/LICENSE.md

---

### 2. ROCm runtime libraries

Binaries built in this directory link against the HIP runtime from the ROCm
installation on the target host. The `rock_mps_test` build additionally links
hipBLASLt, hipRTC, and amd-smi when those components are present in the install.
None of these libraries are distributed by this repository; each retains the
license terms shipped with the ROCm installation.

---

## Redistribution Guidance

The `rocm-tests` files in this directory are MIT-licensed first-party test code.
The `hip-tests` checkout and the built sample binaries are runtime artifacts under
`output/`, not vendored source in this repository. If a release, container image,
build cache, or test artifact bundle includes the cloned `hip-tests` source or the
binaries built from it, include the upstream `LICENSE.md` with that distributed
material.

---

## First-Party Test Code

All sources under `tests/e2e/hip_runtime/src/` — including the self-contained
`split_barrier_stress`, `mps`, `ipc_module_load`, and `partition_isolation`
projects — are original AMD-authored code, copyright Advanced Micro Devices, Inc.,
licensed under the MIT License (the same license as the parent repository). They
are committed to this repository rather than fetched at runtime, and they do not
derive from `hip-tests` source code.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. It is an engineering-compliance summary, not
legal advice; final sign-off for any product distribution should come from AMD
OSS/legal review. For questions about licensing, consult the upstream repository
linked above.*
