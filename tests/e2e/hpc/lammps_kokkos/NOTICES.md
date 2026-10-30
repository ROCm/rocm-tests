# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, and execute the external LAMMPS project at
runtime. LAMMPS and its build-time/runtime dependencies are third-party software
and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## LAMMPS

This test suite may clone LAMMPS from:

https://github.com/lammps/lammps

LAMMPS is a classical molecular dynamics code developed at Sandia National
Laboratories. **LAMMPS is distributed under the GNU General Public License,
version 2 (GPL-2.0).** The `rocm-tests` repository does not vendor or
redistribute LAMMPS source code, LAMMPS build outputs, the `lmp` binary, or any
LAMMPS input decks or potential files as part of its source tree. The cloned
LAMMPS checkout retains its upstream `LICENSE` file.

Because LAMMPS is GPL-2.0, any downstream packaging flow, CI cache, container
image, release artifact, or test-result bundle that redistributes the cloned
LAMMPS source tree, its build directory, or the resulting `lmp` binary is
distributing GPL-2.0 material and must comply with the GPL-2.0 terms, including
the corresponding-source obligations. Keep the LAMMPS checkout and build outputs
confined to the test workspace; do not fold them into `rocm-tests` release
artifacts.

Upstream license:

https://github.com/lammps/lammps/blob/develop/LICENSE

## Kokkos

The configuration built here enables the LAMMPS KOKKOS package, which builds the
Kokkos performance-portability library bundled in the LAMMPS checkout under
`lib/kokkos`. Kokkos is a third-party dependency licensed under Apache-2.0 with
the LLVM exception and is not part of the `rocm-tests` source tree.

Kokkos license information:

https://github.com/kokkos/kokkos/blob/master/LICENSE

## hipFFT and rocThrust

The KOKKOS package is configured with `FFT_KOKKOS=HIPFFT` and links against
hipFFT and rocThrust. These are provided by the ROCm installation supplied via
`--rock-dir` / `ROCK_DIR`; they are third-party dependencies and not part of the
`rocm-tests` source tree.

- https://github.com/ROCm/hipFFT/blob/develop/LICENSE.md
- https://github.com/ROCm/rocThrust/blob/develop/LICENSE

## MPI Runtime

The LAMMPS-Kokkos test builds its own MPI stack into the test build workspace
rather than using an MPI installation already present on the host, so that the
transport in use does not vary between machines and shift the reported scores.
Both components are pinned:

- OpenMPI, downloaded as an upstream release tarball (BSD-3-Clause)
- UCX, cloned from upstream and built with the ROCm transport (BSD-3-Clause)

OpenMPI additionally bundles PMIx and PRRTE, which carry their own upstream
license terms.

These are third-party dependencies and are not part of the `rocm-tests` source
tree. If their source, binaries, build outputs, containers, or cached artifacts
are redistributed, their upstream license files and notices must be preserved.

Upstream licenses:

- https://github.com/open-mpi/ompi/blob/main/LICENSE
- https://github.com/openucx/ucx/blob/master/LICENSE

## hwloc

When available, Kokkos is built with hwloc support for CPU thread and rank
pinning. hwloc is a third-party dependency installed from the host's OS package
repository (BSD-3-Clause) and is not part of the `rocm-tests` source tree.

https://github.com/open-mpi/hwloc/blob/master/COPYING

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The LAMMPS checkout, Kokkos, hipFFT, rocThrust, OpenMPI, UCX, hwloc, and all
generated third-party build artifacts are external runtime/build artifacts. Do
not treat those artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material. Note in particular the GPL-2.0 obligations that
attach to any redistribution of LAMMPS itself.
