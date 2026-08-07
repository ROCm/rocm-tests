# Third-Party Notices — CRIU checkpoint/restore tests

The tests under `tests/e2e/recovery/criu/` **fetch, build, and run** third-party
open-source projects **at test time**. None of these projects' source or binaries are vendored,
committed, or redistributed as part of the rocm-tests repository — clones land in the
gitignored `output/` build directory (Kokkos) or on the test node itself (CRIU).

This file documents each component, exactly how it is used, and the resulting license
obligations. It is an engineering-compliance summary, not legal advice; final sign-off for
any product distribution should come from AMD OSS/legal review.

---

## Components and how they are used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| Kokkos | https://github.com/kokkos/kokkos | tag `4.2.01` | Apache-2.0 WITH LLVM-exception (permissive) |
| CRIU | https://github.com/checkpoint-restore/criu | tag `v4.1` | GPL-2.0-only (LGPL-2.1 for `lib/`) |

### Kokkos
1. `git clone` tag `4.2.01` into `output/external/recovery/kokkos` (pin with
   `ROCM_TEST_KOKKOS_REF`; override the URL with `ROCM_TEST_KOKKOS_URL`).
2. Build only the performance-benchmark target with CMake (`hipcc`, HIP backend, benchmarks
   enabled) into a `build/` tree inside the checkout. The Kokkos sources are **not modified**.
3. Run the built `Kokkos_PerformanceTest_Benchmark` binary as a subprocess and
   checkpoint/restore it with CRIU.

### CRIU
1. `git clone` tag `v4.1` onto the **test node** (`~/criu_src`, outside the repo) — see
   `tests/common/criu/installer.py`.
2. Build (`make`) and **install system-wide** (`sudo make install` → `/usr/local/sbin`);
   build the `amdgpu_plugin.so` and copy it to `/usr/lib/criu`.
3. Invoke the `criu` command-line tool (`criu dump` / `criu restore` / `criu check`) as a
   **separate process**. CRIU is **not modified** and **not linked** into rocm-tests code.

---

## Obligations assessment

- **No redistribution.** rocm-tests does not ship either project's code or binaries; both are
  obtained at runtime from their upstream repositories. GPL-2.0 and Apache-2.0 obligations attach
  to *distribution*, which does not occur here.
- **CRIU (GPL-2.0)** is used only via arm's-length CLI invocation. Per GPL-2.0 §0, *"The act of
  running the Program is not restricted."* Running a separate `criu` process is aggregation, not
  a derivative work, so GPL copyleft does not extend to rocm-tests. rocm-tests source remains
  under its own license (`SPDX-License-Identifier: MIT`).
- **Kokkos (Apache-2.0 WITH LLVM-exception)** permits use, modification, and redistribution with
  attribution. Kokkos is built **unmodified** and its binary is not redistributed, so the only
  attribution/notice obligations would attach on *distribution*, which does not occur here.
- **If distribution is ever added** (e.g. bundling the sources or built binaries): retain the
  Apache-2.0 license, `NOTICE` file, and attribution for Kokkos, and comply with GPL-2.0
  source-offer requirements for CRIU. This is out of scope for the current runtime-fetch model.

---

## Attribution

### Kokkos — Apache License 2.0 WITH LLVM-exception
```
Copyright the Kokkos authors / National Technology & Engineering Solutions of Sandia, LLC (NTESS).
Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights.
Licensed under the Apache License, Version 2.0 with LLVM exceptions.
```
Full license text: `LICENSE` in the Kokkos repository.

### CRIU — GNU General Public License, version 2 (LGPL-2.1 for `lib/`)
```
Copyright the CRIU project contributors (checkpoint-restore/criu).
Licensed under GPL-2.0-only; software under lib/ is licensed under LGPL-2.1.
Only version 2 of the GPL applies unless explicitly stated otherwise.
```
Full license text: `COPYING` in the CRIU repository.
