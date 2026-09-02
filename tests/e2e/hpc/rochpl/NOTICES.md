# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, install, and execute the external rocHPL project at
runtime. rocHPL and its build-time/runtime dependencies are third-party software
and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## rocHPL

This test suite may clone rocHPL from:

https://github.com/ROCm/rocHPL

rocHPL is AMD's GPU-accelerated High-Performance Linpack benchmark. The
`rocm-tests` repository does not vendor or redistribute rocHPL source code or
rocHPL build outputs as part of its source tree. The cloned rocHPL checkout
retains its upstream `LICENSE` file (BSD-3-Clause, incorporating the original
Netlib HPL license terms).

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes the cloned rocHPL source tree, rocHPL build
directory, rocHPL install directory, or rocHPL binaries, it must retain the
corresponding upstream rocHPL license, copyright notices, disclaimers, and any
third-party notices included by rocHPL.

Upstream license:

https://github.com/ROCm/rocHPL/blob/main/LICENSE

## rocBLAS

rocHPL links against rocBLAS for its GPU DGEMM kernels. rocBLAS is provided by
the ROCm installation supplied via `--rock-dir` / `ROCK_DIR`; it is a third-party
dependency and not part of the `rocm-tests` source tree.

rocBLAS license information:

https://github.com/ROCm/rocBLAS/blob/develop/LICENSE.md

## MPI Runtime

The rocHPL test requires an MPI runtime. The framework may use an MPI
installation already present on the host, or it may provision OpenMPI into the
test build workspace when no suitable MPI runtime is found.

MPI implementations are third-party dependencies and are not part of the
`rocm-tests` source tree. If MPI source, binaries, build outputs, containers, or
cached artifacts are redistributed, their upstream license files and notices must
be preserved.

Common MPI implementations include:

- OpenMPI, BSD-style or Apache-2.0 license (depending on the version used)
- MPICH, permissive open-source license

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The rocHPL checkout, rocBLAS, MPI runtime, and all generated third-party
build/install artifacts are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
