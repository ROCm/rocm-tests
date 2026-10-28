# Third-Party Notices

This project is licensed under the MIT License (see `LICENSE`). It also relies on
third-party software that is **downloaded at test runtime** and is **not** vendored
into this repository. Each such dependency remains under its own license; the
notices below are provided for attribution.

## CoralGemm

- **Repository:** https://github.com/AMD-HPC/CoralGemm
- **License:** MIT
- **Used by:** `tests/e2e/system_tools/amd_smi/test_ubb_power.py`
  (`test_ubb_power_workload` — cloned at test runtime as a GPU workload generator)

The MIT License text is reproduced at the repository link above.
No CoralGemm source files are committed to this repository.

## HeCBench

- Used by: `tests/e2e/compiler/hecbench/` (cloned at runtime; see the
  `hecbench_repo` fixture in `tests/e2e/compiler/hecbench/conftest.py`).
- Upstream: https://github.com/zjin-lcf/HeCBench
- License: BSD 3-Clause (see the `LICENSE` file in the upstream repository).
- Copyright (c) 2020, Argonne National Laboratory (Zheming Jin);
  Copyright (c) 2020-, Oak Ridge National Laboratory (Zheming Jin).

The HeCBench benchmark sources are fetched during test execution and compiled/run
on AMD GPU hardware. No HeCBench source is redistributed as part of this
repository; only the benchmark selection list (`subset.json`, MIT, see its
`.license` sidecar) and the pytest harness are checked in here.

## mixbench

- Used by: `tests/e2e/hip_runtime/test_hip_mixbench.py` (cloned at runtime; see the
  `_mixbench_repo` / `mixbench_hip_binary` fixtures in
  `tests/e2e/hip_runtime/conftest.py`).
- Upstream: https://github.com/ekondis/mixbench
- License: GNU General Public License v2 (see the `LICENSE` file in the upstream
  repository).
- Copyright (c) Elias Konstantinidis (in collaboration with Yiannis Cotronis).

The mixbench-hip microbenchmark sources are fetched during test execution and
compiled/run on AMD GPU hardware. No mixbench source is redistributed as part of
this repository; only the pytest harness is checked in here. mixbench is licensed
under GPL-2.0 license; it is neither vendored nor linked into this project's own
sources — it is built and executed as a standalone external binary at test time.

## Kokkos

- Used by: `tests/e2e/hpc/kokkos/test_gpu_rdc_kokkos.py` (cloned at runtime; see the
  `kokkos_rdc_build` fixture in `tests/e2e/hpc/kokkos/conftest.py`).
- Upstream: https://github.com/kokkos/kokkos
- License: Apache License v2.0 with LLVM Exceptions (see the `LICENSE` file in the upstream repository).
- Copyright (c) 2022 National Technology & Engineering Solutions of Sandia, LLC (NTESS).

The Kokkos performance-portability library sources are fetched during test execution,
built with HIP RDC enabled, and run on AMD GPU hardware. No Kokkos source is
redistributed as part of this repository; only the pytest harness is checked in here.

## RAJA

- Used by: `tests/e2e/hpc/raja/test_raja_functional.py` (cloned at runtime; see the
  `raja_functional_build` fixture in `tests/e2e/hpc/raja/conftest.py`).
- Upstream: https://github.com/LLNL/RAJA
- License: BSD 3-Clause (see the `LICENSE` file in the upstream repository).
- Copyright (c) 2016-26, Lawrence Livermore National Security, LLC.

The RAJA performance-portability library sources are fetched during test execution,
built with HIP, and the unit ctest suite is run on AMD GPU hardware. No RAJA source
is redistributed as part of this repository; only the pytest harness is checked in here.
