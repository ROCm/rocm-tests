# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, these scripts clone, build, and run the
external [HeCBench](https://github.com/zjin-lcf/HeCBench) benchmark suite, which
carries its own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does **not** vendor or redistribute the HeCBench source tree or any built
binaries; the `hecbench_repo` fixture clones it from upstream at runtime into a
gitignored build directory. The `subset.json` catalog in this directory contains
only benchmark names and stdout-parsing regexes authored by AMD — it is not
HeCBench source and does not incorporate any HeCBench code.

If a downstream packaging flow (a container image, CI cache layer, or test
artifact bundle) captures the cloned HeCBench source or built benchmark
binaries, that redistribution must retain the upstream copyright notices and
license terms, and must be evaluated against the copyleft obligations described
below.

---

## Upstream Top-Level License — BSD 3-Clause

HeCBench is distributed under a BSD 3-Clause license at its repository root:

```
Copyright (c) 2020,   Argonne National Laboratory  (Zheming Jin)
Copyright (c) 2020-,  Oak Ridge National Laboratory (Zheming Jin)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE.
```

This top-level license is what the `assert_license_present()` provenance guard in
`conftest.py` verifies exists in the checkout. Individual benchmark
subdirectories may carry their own, additional per-benchmark `LICENSE`/`COPYING`
files that take precedence for that benchmark; the notable cases are enumerated
below.

---

## Per-Benchmark License Obligations

The benchmarks exercised via `subset.json` span several licenses. Because
rocm-tests compiles and runs them on the test node but never redistributes the
source or binaries, no copyleft obligation triggers at test-execution time. The
following are documented so that any **caching or artifact-bundling flow** can be
evaluated for compliance before it distributes built material.

### GPLv3 — `flame`, `frna`, `prna`

These three benchmarks carry GPLv3 terms (`flame` via its `COPYING` file).
GPLv3 does not propagate by merely running the code, so in-process compile+run on
the test node is unaffected. However, if a distributed container image or
artifact bundle contains the compiled `flame`, `frna`, or `prna` binary, that
constitutes distribution under GPL and the containing layer must satisfy GPLv3
(offer of corresponding source, GPL-licensing of the combined work).

### LGPLv3 — 21 benchmarks

`ace`, `ans`, `bmf`, `car`, `ccs`, `che`, `cmembench`, `contract`, `face`,
`lebesgue`, `mcmd`, `miniFE`, `morphology`, `permutate`, `rushlarsen`, `segsort`,
`svd3x3`, `sw4ck`, `tensorT`, and `wlcpow` carry LGPLv3 terms. LGPL obligations
(Section 6 re-linking) trigger only on distribution of the linked binary. If
built LGPL benchmark binaries are cached in a distributed image or artifact,
that flow must provide the object files for re-linking or a shared-library
arrangement.

> Note: `flame` ships both a GPLv3 `COPYING` and LGPL-overlapping `LICENSE`
> text; treat it as the hard GPLv3 case above.

### BSD 4-Clause (advertising clause) — `adv`, `axhelm`, `hpl`, `sss`

These carry the original BSD 4-Clause license, whose advertising clause requires
that any advertising material mentioning features of the software display the
attribution notice. This applies only to marketing materials that name these
benchmarks by feature — unlikely in test runs, but noted for completeness.
(`axhelm` additionally carries DOE/Argonne notice clauses.)

### Third-party heritage notices

- **`wedford`** — its `LICENSE` is a pointer to
  [NVIDIA/apex](https://github.com/NVIDIA/apex/blob/master/LICENSE) (BSD 3-Clause);
  the benchmark derives from NVIDIA Apex mixed-precision utilities.
- **`mmcsf`** — Ohio State University Software Distribution License (permissive
  for research use; commercial distribution requires written permission — flag
  for legal review if used in commercial product validation).
- **`memtest`, `pns`** — University of Illinois/NCSA Open Source License.
- **`logprob`** — MIT, sourced from NVIDIA/vLLM.
- **`blockAccess`** — BSD 3-Clause, Copyright NVIDIA Corporation / Duane Merrill
  (CUB heritage).

### Permissive licenses (no additional obligation beyond attribution)

The remaining catalog benchmarks are under permissive terms — MIT, Apache 2.0
(`minibude` with LLVM Exception), BSD 2-Clause (`bm3d`, `sparkler`), BSD
3-Clause, Mozilla Public License 2.0 (`deredundancy`), Boost (`hexciton`), and
public domain / Unlicense (`chacha20`, `seam-carving`) — or fall back to the
top-level BSD 3-Clause when no per-benchmark license file is present. Retain the
respective copyright notices in any redistribution of source or binaries.

---

## Excluded Benchmarks (Non-Commercial Licenses)

Two upstream benchmarks carry **non-commercial-only** licenses that are
incompatible with commercial CI/testing use and cannot be redistributed or run
in a commercial validation pipeline. They have been **removed from
`subset.json`** so neither the nightly smoke nor the weekly full-suite test
collects, compiles, or runs them:

| Benchmark      | Upstream license                              | Reason for exclusion                 |
|----------------|-----------------------------------------------|--------------------------------------|
| `cm`           | Creative Commons BY-NC 3.0                     | Prohibits all commercial use         |
| `vanGenuchten` | University of Illinois Non-Commercial Use License | Prohibits all commercial use     |

Re-including either benchmark would require a separate license grant from the
respective copyright holder.

---

## Redistribution Guidance

The `rocm-tests` files in this directory (`test_hecbench.py`, `conftest.py`,
`subset.json`) are first-party, MIT-licensed AMD test code and contain no
HeCBench source. The HeCBench checkout and any built benchmark binaries are
runtime artifacts, not vendored source in this repository.

If a release, container image, cache, or test-artifact bundle includes the
cloned HeCBench source or built benchmark binaries, it must:

1. Retain the upstream top-level BSD 3-Clause notice and any per-benchmark
   `LICENSE`/`COPYING` files for the included benchmarks.
2. Satisfy GPLv3 obligations for any included `flame`, `frna`, or `prna` binary.
3. Satisfy LGPLv3 Section 6 re-linking obligations for any included LGPL
   benchmark binary.
4. Confirm the excluded non-commercial benchmarks (`cm`, `vanGenuchten`) are not
   present.

---

*This file is provided for compliance with the attribution and copyleft terms of
the external HeCBench suite exercised by this module. For questions about
licensing, consult the upstream repository linked above.*
