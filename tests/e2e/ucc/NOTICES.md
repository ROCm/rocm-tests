# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests clone, build, install, and execute the external UCX, OpenMPI and UCC
projects at runtime. Those projects and their cloned/build-time dependencies are
third-party software and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## UCC

This test suite clones the UCC source at a configurable ref from:

https://github.com/ROCm/ucc

UCC (Unified Collective Communication) is an open-source collective
communication library; this is the ROCm fork of the upstream
[openucx/ucc](https://github.com/openucx/ucc) project. It supplies the
`ucc_test_mpi` collective driver this suite runs. The `rocm-tests` repository
does not vendor or redistribute UCC source code or UCC build outputs as part of
its source tree. The cloned UCC source retains its upstream `LICENSE` file.

UCC is licensed under the BSD 3-Clause license.

Upstream license:

https://github.com/ROCm/ucc/blob/develop/LICENSE

## UCX

This test suite clones the UCX source at a configurable ref from:

https://github.com/ROCm/ucx

UCX (Unified Communication X) is an open-source HPC communication framework;
this is the ROCm fork of the upstream
[openucx/ucx](https://github.com/openucx/ucx) project. It is built here with
`--with-rocm` to provide the GPU-aware transport that OpenMPI and UCC are then
configured against. The `rocm-tests` repository does not vendor or redistribute
UCX source code or UCX build outputs as part of its source tree. The cloned UCX
source retains its upstream `LICENSE` file.

UCX is licensed under the BSD 3-Clause license.

Upstream license:

https://github.com/ROCm/ucx/blob/master/LICENSE

## OpenMPI

This test suite clones OpenMPI, with its recursive submodules, at a configurable
ref from:

https://github.com/open-mpi/ompi

OpenMPI supplies the `mpirun` used to launch the collective driver across ranks.
It is built here with `--with-ucx` against the UCX above, because no ROCm package
ships a UCX-enabled MPI. The `rocm-tests` repository does not vendor or
redistribute OpenMPI source code, its submodules, or OpenMPI build outputs as
part of its source tree. The cloned checkout retains its upstream `LICENSE`
file.

OpenMPI is licensed under the BSD 3-Clause license.

The recursive submodule checkout brings in further third-party dependencies
(e.g. PMIx, PRRTE, hwloc, libevent), each under its own upstream license. Those
are not part of the `rocm-tests` source tree either. If submodule source, build
outputs, containers, or cached artifacts are redistributed, the corresponding
upstream license files and notices must be preserved.

Upstream license:

https://github.com/open-mpi/ompi/blob/main/LICENSE

## RCCL

UCC is configured with `--with-rccl` against the RCCL supplied by the ROCm
installation on the target host, and the resulting collectives link
`librccl.so` at runtime. RCCL is not cloned, built, or distributed by these
tests; it is used as provided by that ROCm install.

RCCL is a fork of NVIDIA's NCCL and is licensed under the BSD 3-Clause license,
with NVIDIA, AMD and Microsoft copyright holders. The full license text, the
DOE funding acknowledgement, and the Microsoft modification notice are
reproduced in `tests/e2e/rccl/NOTICES.md` in this repository.

Upstream repository:

https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl

Upstream license file:

https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl/LICENSE.txt

## OS build prerequisites

The three autotools builds performed here depend on host toolchain and
development packages (autoconf, automake, libtool, a C/C++ compiler, and the
usual build utilities). These OS packages are third-party software with their own
upstream licenses and are not part of the `rocm-tests` source tree.

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The UCC, UCX and OpenMPI clones, their dependencies and submodules, the
generated install prefixes, and all other third-party build artifacts are
external runtime/build artifacts. Do not treat those artifacts as MIT-licensed
`rocm-tests` source.

If any downstream packaging flow, CI cache, container image, release artifact, or
test-result bundle redistributes such material, it must retain the applicable
upstream license files, copyright notices, disclaimers, and any third-party
notices included by those projects.
