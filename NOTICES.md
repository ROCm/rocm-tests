# Third-Party Notices

This project is licensed under the MIT License (see `LICENSE`). It also relies on
third-party software that is **downloaded at test runtime** and is **not** vendored
into this repository. Each such dependency remains under its own license; the
notices below are provided for attribution.

## BabelStream

- Used by: `tests/e2e/hip_runtime/test_babelstream.py` (cloned at runtime; see the
  `babelstream_repo` and `babelstream_binary` fixtures in `tests/e2e/hip_runtime/conftest.py`).
- Upstream: https://github.com/UoB-HPC/BabelStream
- License: Custom permissive (similar to BSD 3-Clause; see the `LICENSE` file in the
  upstream repository).
- Copyright (c) Tom Deakin and Simon McIntosh-Smith (University of Bristol);
  based on John D. McCalpin's original STREAM benchmark.

The BabelStream HIP memory bandwidth benchmark is fetched during test execution and
compiled/run on AMD GPU hardware via the HIP backend. No BabelStream source is
redistributed as part of this repository; only the pytest harness is checked in here.
BabelStream is licensed under a permissive license compatible with commercial and
academic use; results must comply with the BabelStream Run Rules when published.
