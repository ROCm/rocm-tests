# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests clone, build, install, and execute the external UCX project at
runtime. UCX and its cloned/build-time dependencies are third-party software
and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## UCX

This test suite clones the UCX source at a configurable ref from:

https://github.com/openucx/ucx

UCX (Unified Communication X) is an open-source HPC communication framework. The
`rocm-tests` repository does not vendor or redistribute UCX source code or UCX
build outputs as part of its source tree. The cloned UCX source retains its
upstream `LICENSE` file.

UCX is licensed under the BSD 3-Clause license.

Upstream license:

https://github.com/openucx/ucx/blob/master/LICENSE

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes the cloned UCX source tree, UCX build
directory, UCX install directory, or UCX binaries (including the built
GoogleTest binary), it must retain the corresponding upstream UCX license,
copyright notices, disclaimers, and any third-party notices included by UCX.

## GoogleTest

The UCX build embeds GoogleTest when configured with `--enable-gtest`.

GoogleTest is a third-party dependency licensed under BSD 3-Clause. It is not part
of the `rocm-tests` source tree. If GoogleTest source, headers, build outputs,
containers, or cached artifacts are redistributed, the corresponding license files
and notices must be preserved.

GoogleTest license information:

https://github.com/google/googletest/blob/main/LICENSE

## OS build prerequisites

The UCX configure line used here auto-disables optional features when their
development packages are absent. For NUMA-optimised transports, provision
`numactl-devel` (and, for docs, `doxygen`) on fleet nodes ahead of time with
`--pre-install pkg=numactl-devel`. These OS packages are third-party software with
their own upstream licenses and are not part of the `rocm-tests` source tree.

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The UCX clone, UCX dependencies, GoogleTest, and all generated third-party
build/install artifacts are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
