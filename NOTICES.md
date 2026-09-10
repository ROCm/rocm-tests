# Third-Party Notices

This project is licensed under the MIT License (see `LICENSE`). It also relies on
third-party software that is **downloaded at test runtime** and is **not** vendored
into this repository. Each such dependency remains under its own license; the
notices below are provided for attribution.

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
under GPL v2 (copyleft); it is neither vendored nor linked into this project's own
sources — it is built and executed as a standalone external binary at test time.
