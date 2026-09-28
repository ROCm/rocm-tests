# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

These tests clone, build, and execute external projects at runtime, and drive
binaries shipped by the ROCm installation under test. Those projects are
third-party software and retain their own upstream license terms.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## cuda_memtest

The `cudamemtest` workload clones the cuda_memtest source at a configurable ref
from:

https://github.com/ComputationalRadiationPhysics/cuda_memtest

This is a HIP-enabled fork of the CUDA GPU memtest developed by the Innovative
Systems Lab at the National Center for Supercomputing Applications. The
`rocm-tests` repository does not vendor or redistribute cuda_memtest source code
or build outputs as part of its source tree.

cuda_memtest is licensed under the University of Illinois/NCSA Open Source
License, Copyright 2009 University of Illinois.

Note that the upstream repository carries no top-level `LICENSE` or `COPYING`
file; the license text and copyright notice appear in the header of each source
file (for example `cuda_memtest.h`). Any redistribution must therefore preserve
those per-file headers, since there is no separate license file to ship
alongside them.

If any downstream packaging flow, CI cache, container image, release artifact,
or test-result bundle redistributes the cloned cuda_memtest source tree, its
build directory, or its binaries, it must retain the above copyright notice,
the NCSA license conditions, and the disclaimers as they appear in the source
headers.

## TransferBench

The `transferbench` workload drives the `TransferBench` binary that ships with
ROCm Validation Suite. It does not clone, build, or install TransferBench from
source.

TransferBench upstream:

https://github.com/ROCm/TransferBench

TransferBench is licensed under the MIT license, Copyright (c) 2019-2025
Advanced Micro Devices, Inc.

## ROCm components driven as preinstalled binaries

The remaining workloads execute binaries provided by the ROCm installation
under test rather than building them here:

- `hipblaslt_bench` runs `hipblaslt-bench` from `{rock_dir}/bin`.
- `rvs_tst` and `rvs_iet_stress` run the `rvs` binary from ROCm Validation
  Suite, against configuration files from its installed `conf` tree.

These are ROCm components with their own upstream licenses and are not part of
the `rocm-tests` source tree. Their configuration files are read from the
installed package and are neither vendored nor redistributed here.

## Device configuration mapping

`tests/common/rvs_config_mapping.csv` records, per PCI device and revision,
which ROCm Validation Suite modules apply to that GPU and which configuration
directory each one reads. Every directory it names is one that the ROCm
Validation Suite package already ships. The file is shared with the RVS
module-test suite under `tests/e2e/system_tools/rvs/`.

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

The cuda_memtest clone, its build outputs, and every ROCm or third-party binary
these tests execute are external runtime/build artifacts. Do not treat those
artifacts as MIT-licensed `rocm-tests` source.

If any downstream workflow publishes or redistributes such artifacts, include
the applicable upstream license files, copyright notices, disclaimers, and
notices with the redistributed material.
