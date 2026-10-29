# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, and execute the external RAJA project at
runtime. RAJA and its build-time dependencies are third-party software and
retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## RAJA

This test suite may clone RAJA from:

https://github.com/LLNL/RAJA

Used by: `tests/e2e/hpc/raja/` (functional HIP unit ctest, `raja_ut`).

RAJA is a C++ performance-portability library. The `rocm-tests` repository
does not vendor or redistribute RAJA source code or RAJA build outputs as
part of its source tree. The cloned RAJA checkout retains its upstream
`LICENSE` file.

RAJA is licensed under the BSD 3-Clause License, a permissive license.

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes the cloned RAJA source tree, RAJA build
directory, or RAJA binaries, it must retain the corresponding upstream RAJA
license, copyright notices, disclaimers, and any third-party notices included
by RAJA.

Upstream license:

https://github.com/LLNL/RAJA/blob/develop/LICENSE

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The RAJA checkout, its dependencies, and all generated third-party
build artifacts are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
