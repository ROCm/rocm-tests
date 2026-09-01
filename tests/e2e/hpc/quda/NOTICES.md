# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, install, and execute the external QUDA project at
runtime. QUDA and its downloaded/build-time dependencies are third-party
software and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## QUDA

This test suite may clone QUDA from:

https://github.com/lattice/quda

QUDA is a lattice-QCD GPU library. The `rocm-tests` repository does not vendor or
redistribute QUDA source code or QUDA build outputs as part of its source tree.
The cloned QUDA checkout retains its upstream `LICENSE` file.

QUDA's upstream `LICENSE` states the QUDA project license (MIT) and additionally
compiles notices for bundled or referenced third-party components, including
BSD-3-Clause code (e.g. NVIDIA cub/NVTX, Google Test), Apache-2.0 code, further
MIT-licensed code, and CC-BY-SA material, among others.

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes the cloned QUDA source tree, QUDA build
directory, QUDA install directory, or QUDA binaries, it must retain the
corresponding upstream QUDA license, copyright notices, disclaimers, and any
third-party notices included by QUDA.

Upstream license:

https://github.com/lattice/quda/blob/develop/LICENSE

## USQCD / QMP

The QUDA build may download, build, or link USQCD/QMP components when configured
with options such as `QUDA_DOWNLOAD_USQCD=ON` and `QUDA_QMP=ON`.

These components are third-party dependencies. They are not part of the
`rocm-tests` source tree. If downloaded source, build outputs, install outputs,
containers, or cached artifacts containing these components are redistributed,
their upstream license files and notices must be preserved.

USQCD QMP source:

https://github.com/usqcd-software/qmp

## Eigen

The QUDA build may download or use Eigen when configured with
`QUDA_DOWNLOAD_EIGEN=ON`.

Eigen is a third-party dependency and is primarily licensed under MPL-2.0, with
some files under other licenses depending on what is used. If Eigen source,
headers, build outputs, containers, or cached artifacts are redistributed, the
corresponding Eigen license files and notices must be preserved.

Eigen license information:

https://gitlab.com/libeigen/eigen

## MPI Runtime

The QUDA test requires an MPI runtime. The framework may use an MPI installation
already present on the host, or it may provision OpenMPI into the test build
workspace when no suitable MPI runtime is found.

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

The QUDA checkout, QUDA dependencies, MPI runtime, and all generated third-party
build/install artifacts are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
