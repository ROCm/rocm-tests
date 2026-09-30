# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, install, and execute the external Kokkos project at
runtime. Kokkos and its build-time dependencies are third-party software and
retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## Kokkos

This test suite may clone Kokkos from:

https://github.com/kokkos/kokkos

Used by: `tests/e2e/hpc/kokkos/` (HIP Relocatable Device Code / `-fgpu-rdc`
validation).

Kokkos is a C++ performance-portability programming ecosystem. The `rocm-tests`
repository does not vendor or redistribute Kokkos source code or Kokkos build
outputs as part of its source tree. The cloned Kokkos checkout retains its
upstream `LICENSE` file.

Kokkos is licensed under the Apache License v2.0 with LLVM Exceptions
(`Apache-2.0 WITH LLVM-exception`), a permissive license.

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes the cloned Kokkos source tree, Kokkos build
directory, Kokkos install directory, or Kokkos binaries, it must retain the
corresponding upstream Kokkos license, copyright notices, disclaimers, and any
third-party notices included by Kokkos.

Upstream license:

https://github.com/kokkos/kokkos/blob/master/LICENSE

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The Kokkos checkout, its dependencies, and all generated third-party
build/install artifacts are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
