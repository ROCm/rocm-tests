# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests may clone, build, and execute external projects at runtime.
Those projects and their build-time dependencies are third-party software
and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## GROMACS

This test suite may clone GROMACS from:

https://gitlab.com/gromacs/gromacs

Used by: `tests/e2e/hpc/gromacs/` (functional HIP MPI ctest).

The `rocm-tests` repository does not vendor or redistribute GROMACS source
code or GROMACS build outputs as part of its source tree. The cloned
checkout retains its upstream license file.

GROMACS is licensed under the GNU Lesser General Public License, version 2.1.

Upstream license:

https://gitlab.com/gromacs/gromacs/-/blob/main/COPYING.LIB

## OpenUCX

The functional build compiles OpenUCX so GROMACS can link an MPI that uses it.

https://github.com/openucx/ucx

OpenUCX is licensed under the BSD 3-Clause License.

## Open MPI

The functional build compiles Open MPI 4.1.4 and launches ctest with it.

https://www.open-mpi.org/

Open MPI is licensed under the 3-clause BSD license.

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The GROMACS, OpenUCX, and Open MPI checkouts, and all generated third-party
build artifacts, are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include the
applicable upstream license files, copyright notices, disclaimers, and notices
with the redistributed material.
