# Third-Party Notices — CRIU

The shared CRIU machinery under `tests/common/criu/` **fetches, builds, and installs** the
third-party CRIU project **on the test node at test run time**. CRIU's source and binaries are not
vendored, committed, or redistributed as part of the rocm-tests repository — the checkout lands
in the gitignored `output/` build tree and the built binary/plugin are installed on the node
itself.

This file documents the component, exactly how it is used, and the resulting license obligations.
It is an engineering-compliance summary, not legal advice; final sign-off for any product
distribution should come from AMD OSS/legal review.

---

## Component and how it is used

| Component | Upstream | Pinned ref | License |
|---|---|---|---|
| CRIU | https://github.com/checkpoint-restore/criu | tag `v4.1` | GPL-2.0-only (LGPL-2.1 for `lib/`) |

1. `git clone` tag `v4.1` onto the **test node** (into the `output/` build tree) — see
   `installer.py`.
2. Build (`make`) and **install system-wide** (`sudo make install-criu` → `/usr/local/sbin`);
   build the `amdgpu_plugin.so` and copy it to `/usr/lib/criu`.
3. Invoke the `criu` command-line tool (`criu dump` / `criu restore` / `criu check`) as a
   **separate process**. CRIU is **not modified** and **not linked** into rocm-tests code.

---

## Obligations assessment

- **No redistribution.** rocm-tests does not ship CRIU's code or binaries; it is obtained at
  runtime from upstream. GPL-2.0 obligations attach to *distribution*, which does not occur here.
- **Arm's-length CLI use.** CRIU is used only via CLI invocation. Per GPL-2.0 §0, *"The act of
  running the Program is not restricted."* Running a separate `criu` process is aggregation, not a
  derivative work, so GPL copyleft does not extend to rocm-tests. rocm-tests source remains under
  its own license (`SPDX-License-Identifier: MIT`).
- **Not modified.** CRIU is built unmodified, so GPL-2.0 §2(a) (mark changed files) never applies.
- **If distribution is ever added** (bundling CRIU source or binaries): comply with GPL-2.0
  source-offer requirements. This is out of scope for the current runtime-fetch model.

---

## Attribution

### CRIU — GNU General Public License, version 2 (LGPL-2.1 for `lib/`)
```
Copyright the CRIU project contributors (checkpoint-restore/criu).
Licensed under GPL-2.0-only; software under lib/ is licensed under LGPL-2.1.
Only version 2 of the GPL applies unless explicitly stated otherwise.
```
Full license text: `COPYING` in the CRIU repository.
