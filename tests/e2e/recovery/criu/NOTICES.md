# Third-Party Notices — CRIU checkpoint/restore tests

The tests under `tests/e2e/recovery/criu/` **fetch, build, and run** third-party
open-source projects **at test time**. None of these projects' source or binaries are vendored,
committed, or redistributed as part of the rocm-tests repository — clones land in the
gitignored `output/` build directory (hip-tests) or on the test node itself (CRIU).

This file documents each component, exactly how it is used, and the resulting license
obligations. It is an engineering-compliance summary, not legal advice; final sign-off for
any product distribution should come from AMD OSS/legal review.

---

## Components and how they are used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| hip-tests | https://github.com/ROCm/hip-tests | commit `3543bc3b9140e0a506ed3dec643b4def672bd171` | MIT |
| CRIU | https://github.com/checkpoint-restore/criu | tag `v4.1` | GPL-2.0-only (LGPL-2.1 for `lib/`) |

### hip-tests
1. `git clone` at the pinned commit into `output/test-binaries/recovery/hip_tests`.
2. **Modify** the `samples/2_Cookbook/0_MatrixTranspose` sample in place via the single
   AMD-authored `tests/common/criu/patch_matrix_transpose.py`: `MatrixTranspose.cpp` gets `<thread>`/`<chrono>`
   includes and a 100-iteration kernel-relaunch loop, and `CMakeLists.txt` gets the ROCm device-lib
   flag. No upstream file is replaced or redistributed.
3. Build a standalone binary with CMake's HIP language mode.
4. Run the built `MatrixTranspose` binary as a subprocess, checkpoint/restore it with CRIU.

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
  obtained at runtime from their upstream repositories. GPL-2.0 and MIT obligations attach to
  *distribution*, which does not occur here.
- **CRIU (GPL-2.0)** is used only via arm's-length CLI invocation. Per GPL-2.0 §0, *"The act of
  running the Program is not restricted."* Running a separate `criu` process is aggregation, not
  a derivative work, so GPL copyleft does not extend to rocm-tests. rocm-tests source remains
  under its own license (`SPDX-License-Identifier: MIT`).
- **hip-tests (MIT)** permits use, modification, and redistribution provided the copyright and
  permission notice are retained on *redistribution*. The in-place loop patch is allowed; the
  source and built binary are not redistributed (obtained at runtime), so no obligation attaches
  under the current runtime-fetch model.
- **If distribution is ever added** (e.g. bundling the sources or built binaries): retain the MIT
  copyright and permission notice for hip-tests, and comply with GPL-2.0 source-offer requirements
  for CRIU. This is out of scope for the current runtime-fetch model.

---

## Attribution

### hip-tests — MIT License
```
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
```
Full license text: `LICENSE` in the hip-tests repository (https://github.com/ROCm/hip-tests).

### CRIU — GNU General Public License, version 2 (LGPL-2.1 for `lib/`)
```
Copyright the CRIU project contributors (checkpoint-restore/criu).
Licensed under GPL-2.0-only; software under lib/ is licensed under LGPL-2.1.
Only version 2 of the GPL applies unless explicitly stated otherwise.
```
Full license text: `COPYING` in the CRIU repository.
