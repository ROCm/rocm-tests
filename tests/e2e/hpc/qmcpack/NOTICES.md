# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, and execute the external QMCPACK project at
runtime. QMCPACK and its build-time dependencies are third-party software and
retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## QMCPACK

This test suite may clone QMCPACK from:

https://github.com/QMCPACK/qmcpack

Used by: `tests/e2e/hpc/qmcpack/` (functional HIP deterministic ctest).

QMCPACK is a quantum Monte Carlo application. The `rocm-tests` repository
does not vendor or redistribute QMCPACK source code or QMCPACK build outputs
as part of its source tree. The cloned QMCPACK checkout retains its upstream
`LICENSE` file.

QMCPACK is licensed under the University of Illinois/NCSA Open Source License.

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes the cloned QMCPACK source tree, QMCPACK build
directory, or QMCPACK binaries, it must retain the corresponding upstream QMCPACK
license, copyright notices, disclaimers, and any third-party notices included
by QMCPACK.

Upstream license:

https://github.com/QMCPACK/qmcpack/blob/develop/LICENSE

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The QMCPACK checkout, its dependencies, and all generated third-party
build artifacts are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
