---
name: porter
description: Port tests from external sources (shell scripts, raw Python, legacy pytest, C++ gtest, other AMD frameworks) into the rocm-tests framework — maps foreign patterns to the correct executors, markers, and assertion style
user-invocable: true
---

# Agent: Legacy Porter

**Objective:** Port an external test — shell script, raw Python file, C++ gtest, non-compliant pytest, or another AMD framework's runner — into a fully framework-compliant `rocm-tests` pytest file pair (`conftest.py` + `test_*.py`).

---

## Section 1 — Framework Grounding

Read these files before starting:

1. `framework/markers/taxonomy.py` — `MARKER_SCHEMA`, `REQUIRED_DIMENSIONS`, `CATEGORY_PROFILES`
2. `framework/plugins/builder_plugin.py` — `compile_binary` signature, `ld_path`
3. `framework/plugins/remote_node_plugin.py` — `target_executor`
4. `framework/common/helpers.py` — `ExecutionResult` fields (`.ok`, `.exit_code`, `.stdout`, `.stderr`)
5. The **complete source file** to be ported
6. The **closest existing test area** in `tests/e2e/` — read both `conftest.py` and test file as reference

Optional:

7. `framework/plugins/artifacts_plugin.py` — `allure_reporter` (only if user requests Allure output)

---

## Section 2 — Source Type Identification

| Source type | What to expect | Generated files |
|---|---|---|
| **C++ gtest program** (`.cpp` with `EXPECT_*`/`ASSERT_*`) | Binary that self-validates; exits 0 on pass | `src/<name>.cpp` + `conftest.py` + `test_<name>.py` |
| **Shell script that compiles + runs a .cpp** | `hipcc` call + `./binary [args]` | `src/<source>.cpp` + `conftest.py` + `test_<name>.py` |
| **Shell script that only runs system binaries** | `rocm-smi`, `hipconfig`, `amd-smi` calls | `test_<name>.py` only |
| **Raw Python with subprocess** | `subprocess.run("rocm-smi ...")` or GPU Python API | `test_<name>.py` only (if system binary); else + `conftest.py` |
| **Non-compliant pytest** | Missing markers, `subprocess.run` in test body | Rewrite in place; retain `.cpp` if referenced |
| **AMD framework runner** | `rccl-tests` launcher, `rocBLAS-bench` scripts | `conftest.py` + `test_<name>.py` |
| **CI YAML step** | Inline bash steps in GitHub Actions | Extract each step → one test function each |

---

## Section 3 — Transformation Logic

### Step 1 — Extract Logic

Read the complete source. For each distinct GPU operation, record:

- **What it does** — the command or API being exercised
- **What it asserts** — expected output, return code, or computed value
- **What it guards** — optional dependencies, platform checks, minimum versions
- **What it sets up** — env vars, binary compilation, file creation, CLI arguments

Separate **setup** (pre-conditions) from **validation** (assertions). If source has multiple distinct GPU operations, create **one test function per operation** — never merge them into one giant test.

### Step 2 — Pattern Mapping

Apply this transformation table to every external pattern found:

| External pattern | rocm-tests equivalent | Notes |
|---|---|---|
| `subprocess.run(cmd, ...)` | `target_executor.run(cmd)` | Drop `check=True`; use `result.ok` |
| `subprocess.Popen(cmd, ...)` | `target_executor.run(cmd)` | Executor handles Popen internally |
| `os.environ["ROCR_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Injected automatically by `target_executor` |
| `os.environ["HIP_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Same — never set in test code |
| `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | ROCm uses `ROCR_`, never `CUDA_` |
| `hipcc source.cpp -o binary` | `compile_binary()` in `conftest.py` | Session-scoped, incremental, xdist-safe |
| `.hip` source file (not `.cpp`) | `cmake_build()` from `tests.common._cmake_build` via session fixture | hipcc cannot handle `.hip` extension; use CMake HIP language mode (see template 4d) |
| `cmake && make` / `cmake --build` in shell script | `cmake_build()` from `tests.common._cmake_build` in `conftest.py` | Preserve CMake build; wrap in session fixture; never define inline helper |
| Inline `_cmake_build()` in conftest | `from tests.common._cmake_build import cmake_build` | Use the shared helper to avoid duplication and ensure consistent clang++ discovery |
| `ROCM_PATH` exported in environment | Pass as `env ROCM_PATH={rock_dir}` in run cmd | Runtime path lookup for `.hsaco`; not a Python env var |
| `./binary [args]` | `target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary} [args]")` | `ld` from `ld_path["LD_LIBRARY_PATH"]` |
| `export VARIABLE=value` | Remove; pass as CLI arg or fixture config | Never set GPU env vars in test code |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic |
| `if not shutil.which("tool"): sys.exit(1)` | `pytest.skip("tool not available on this node")` | Graceful skip vs session abort |
| `try: import X \nexcept ImportError: sys.exit(1)` | `pytest.skip("X not installed")` inside test | Never `sys.exit` — use `pytest.skip` |
| `time.sleep(N)` | **Remove entirely** | Health checks handle GPU readiness |
| `assert proc.returncode == 0` | `assert result.ok` + sentinel/metric assertion | Add meaningful stdout check |
| `assert "ERROR" not in output` | `assert "ERROR" not in result.stdout` | Direct — stdout is a plain string |
| Hardcoded `/dev/renderD128` | Let executor handle — never hardcode device paths | |
| Hardcoded GPU index `device_id = 0` | Let executor manage allocation | |
| `ROCR_VISIBLE_DEVICES=0 ./binary` (pinned single index) | `@pytest.mark.gpu_indices([0])` + `target_executor.run(binary)` | Executor injects the device env automatically |
| Loop over explicit GPU indices (`for idx in [0, 2]: run_on(idx)`) | `manual_gpu_allocator.pin(gpu_index=idx)` context manager | Acquires/releases one index at a time within the test body |
| `logging.info("step X")` | Optional `allure_reporter.step("step X")` | Not required; add only if user requests Allure |
| Shell `${VAR:-default}` | `framework_config.section.field or "default"` | Only if the value is a framework config option |
| C++ `EXPECT_EQ(a, b)` | `assert a == b, f"Expected {b}, got {a}"` | In Python test post-processing |
| C++ `ASSERT_GT(val, thr)` | `assert val > thr, f"Got {val}, expected > {thr}"` | Direct translation |
| gtest binary stdout `[  PASSED  ]` | `assert "PASSED" in result.stdout` | gtest exits 0 and prints PASSED on success |

### Step 3 — C++ Source Handling (Primary Path)

**This is the most common porting case.** C++ sources — whether gtest programs or standalone HIP binaries — must be compiled via `conftest.py` + `compile_binary`. Never wrap them in `python3 -c`.

**Decision: which artifacts to generate?**

| Source type | src/ file | conftest.py | test_*.py |
|---|---|---|---|
| `.cpp` gtest or standalone binary | Copy/adapt into `tests/e2e/<domain>/src/` | YES — compile_binary fixture with correct flags | YES — runs binary, asserts exit + stdout |
| `.hip` source (CMake HIP mode) | Copy into `tests/e2e/<domain>/src/` | YES — `cmake_build()` from `tests.common._cmake_build` + binary fixture | YES |
| Shell script calling `hipcc source.cpp` | Extract `.cpp` into `tests/e2e/<domain>/src/` | YES — compile_binary replicates the hipcc call | YES |
| Shell script running system binaries only | Not needed | Not needed | YES — target_executor.run("rocm-smi ...") |
| Raw Python using subprocess on system binary | Not needed | Not needed | YES |

**hipcc flags for gtest binaries:**

```python
extra_flags=[
    f"-I{rock_dir}/include",
    f"-L{rock_dir}/lib",
    "-lgtest",
    "-lgtest_main",
    "-lpthread",
    "-lamdhip64",
]
```

**gtest assertion translation table:**

| C++ gtest | Python post-processing of binary stdout |
|---|---|
| Any `EXPECT_*/ASSERT_*` | gtest exits 0 on all pass; assert `result.ok` + `"PASSED" in result.stdout` |
| `EXPECT_EQ(a, b)` | Binary prints `a == b` or fails; Python asserts `result.ok` |
| `EXPECT_GT(val, threshold)` | Binary validates internally; Python asserts `result.ok` |
| Binary emits numeric result | `parse_metric(result.stdout, "KEY")` + `assert value > threshold` |

**For gtest sources: keep gtest assertions in the C++ binary.** Do NOT translate `EXPECT_*` into Python assertions. Let gtest self-validate and exit non-zero on failure. The Python test asserts `result.ok` and `"PASSED" in result.stdout`.

**`.hip` sources and CMake builds:** When the source uses `.hip` extensions or its `CMakeLists.txt` calls `enable_language(HIP)`, generate a `conftest.py` that imports `cmake_build` from `tests.common._cmake_build` (see creator.md template 6h) instead of using `compile_binary`. `cmake_build()` automatically handles `-DCMAKE_CXX_COMPILER`, `-DROCM_PATH`, and `-DCMAKE_PREFIX_PATH`. Use `pytest.skip()` when clang++ is absent.

### Step 4 — Resolve Markers (Profile-Aware)

**Look up `CATEGORY_PROFILES` for the target directory FIRST.** Only declare markers that are NOT already auto-injected.

| Target directory | Auto-injected | What to declare |
|---|---|---|
| `tests/e2e/compiler/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hwq_heuristic/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hip_runtime/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hipblaslt/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocprim/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocm_libs/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/dry_run/` or new directory | none | All 4 required dimensions + `runtime.*` |

> **Authoritative source:** Always read `framework/markers/taxonomy.py → CATEGORY_PROFILES` before placing a ported test. The table above mirrors current entries — verify before use.

**`runtime.*` is NEVER auto-injected. Always declare it explicitly.**

Estimate wall time from the source to assign `runtime.*`:

| Source wall time | Marker |
|---|---|
| < 5 min | `runtime.fast` |
| 5–30 min | `runtime.medium` |
| 30 min – 2 hr | `runtime.longevity` |
| Hours | `runtime.soak` + `ci.weekly` |

### Step 5 — Re-structure Output

Rewrite into the standard `rocm-tests` pattern. Every ported file must have:

- Copyright header + SPDX identifier
- Module docstring with `Ported from:` and `Validates:` sections
- `scope="session"` on every `compile_binary` fixture in `conftest.py`
- `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` as the run command for compiled binaries
- `assert result.ok` with a diagnostic message (exit code + truncated stdout + stderr)
- At least one stdout assertion beyond `result.ok` when the binary emits a detectable output

### Step 6 — Validate

After generating files, present:

```bash
# 1. Collection test (no GPU needed)
pytest tests/e2e/<domain>/test_<name>.py --collect-only -q --no-gpu

# 2. DryRun (fixture wiring check)
pytest tests/e2e/<domain>/test_<name>.py --no-gpu -v

# 3. GPU run (with ROCm install)
pytest tests/e2e/<domain>/test_<name>.py -v --rock-dir=/path/to/rocm
```

---

## Section 4 — Code Templates

### 4a. C++ Gtest Binary Porting (3 files)

**Source:** `external/tests/test_hip_feature.cpp` using gtest  
**Generated:** `tests/e2e/<domain>/src/test_hip_feature.cpp` + `conftest.py` + `test_hip_feature.py`

**`conftest.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Build fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import pytest

_SUBDIR = "<domain>"
_SRC = "tests/e2e/<domain>/src/test_hip_feature.cpp"
_NAME = "test_hip_feature"


@pytest.fixture(scope="session")
def test_hip_feature_binary(compile_binary, rock_dir: str) -> str:
    """Compile test_hip_feature.cpp via hipcc with gtest; return binary path."""
    return compile_binary(
        src=_SRC,
        output_name=_NAME,
        std="c++17",
        opt="-O2",
        include_dirs=["tests/common/include"],
        extra_flags=[
            f"-I{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-lgtest",
            "-lgtest_main",
            "-lpthread",
            "-lamdhip64",
        ],
        subdir=_SUBDIR,
    )
```

**`test_hip_feature.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hip_feature.py — HIP feature validation (ported from gtest).

Ported from: <source file or external framework>

The gtest binary self-validates: exits 0 when all test cases pass.
gtest prints "[  PASSED  ] N tests." to stdout on success.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <list profile markers>

Explicit markers:
    runtime.<budget>
"""

import pytest


@pytest.mark.runtime.<budget>
def test_hip_feature(
    target_executor,
    ld_path: dict,
    test_hip_feature_binary: str,
):
    """Run the HIP feature gtest binary on an AMD GPU.

    Exits 0 when all gtest cases pass; non-zero on any failure.
    """
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {test_hip_feature_binary}"
    )
    assert result.ok, (
        f"test_hip_feature failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # gtest prints "[  PASSED  ] N tests." on success
    assert "PASSED" in result.stdout, (
        f"gtest did not report PASSED:\n{result.stdout[:1000]}"
    )
```

---

### 4b. Shell Script Porting (3 files)

**Source:**
```bash
#!/bin/bash
export ROCR_VISIBLE_DEVICES=0
hipcc tests/hip_kernel.cpp -o /tmp/hip_test -O2 -std=c++17
/tmp/hip_test --iterations=100
if [ $? -ne 0 ]; then exit 1; fi
echo "TEST_PASSED"
```

**`conftest.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Build fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import pytest

_SUBDIR = "<domain>"
_SRC = "tests/e2e/<domain>/src/hip_kernel.cpp"
_NAME = "hip_test"


@pytest.fixture(scope="session")
def hip_test_binary(compile_binary) -> str:
    """Compile hip_kernel.cpp via hipcc; return absolute binary path."""
    return compile_binary(
        src=_SRC,
        output_name=_NAME,
        std="c++17",
        opt="-O2",
        subdir=_SUBDIR,
    )
```

**`test_hip_kernel.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_hip_kernel.py — HIP kernel validation.

Ported from: scripts/run_hip_test.sh

Validates:
    1. HIP kernel executes successfully for 100 iterations.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.fast
"""

import pytest


@pytest.mark.runtime.fast
def test_hip_kernel(
    target_executor,
    ld_path: dict,
    hip_test_binary: str,
):
    """Run the HIP kernel binary for 100 iterations on an AMD GPU."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {hip_test_binary} --iterations=100"
    )
    assert result.ok, (
        f"hip_kernel failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "TEST_PASSED" in result.stdout, (
        f"hip_kernel did not print TEST_PASSED:\n{result.stdout[:1000]}"
    )
```

---

### 4c. System Binary / Raw Python Porting (1 file only)

When the source only invokes system binaries (no C++ compilation needed):

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what the ported script validated>.

Ported from: <source file path>

Validates:
    1. <assertion extracted from source>

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.fast
"""

import pytest


@pytest.mark.runtime.fast
def test_<name>(target_executor):
    """<What this test verifies>."""
    # Optional guard if source had a dependency check:
    # check = target_executor.run("which rocm-smi")
    # if not check.ok:
    #     pytest.skip("rocm-smi not available on this node")

    result = target_executor.run("rocm-smi --showid")
    assert result.ok, (
        f"rocm-smi failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "GPU" in result.stdout, (
        f"rocm-smi output missing GPU info:\n{result.stdout[:1000]}"
    )
```

---

### 4d. CMake-Based Build Porting (`.hip` or multi-target CMakeLists.txt)

**Source:** A shell script or CI step that calls `cmake && cmake --build && ./binary`, or a `.hip` source that requires HIP language mode.

**Generated:** `tests/e2e/<domain>/src/<source>.hip` + `conftest.py` (using `cmake_build` from shared helper) + `test_<name>.py`

**`conftest.py`:** Import the shared helper — never define an inline `_cmake_build()`:
```python
from tests.common._cmake_build import cmake_build, find_rocm_clangpp  # noqa: F401
```
Use the CMake fixture pattern from creator.md section 6h. Key rules:
- `cmake_build()` automatically locates clang++ via `find_rocm_clangpp()` and passes `-DROCM_PATH`, `-DCMAKE_PREFIX_PATH`, `-DCMAKE_CXX_COMPILER`
- Accept `gpu_arch: str | None` in the build fixture and pass it as `cmake_build(gpu_arch=gpu_arch)`
- Use `pytest.skip()` when `clang++` is absent (optional path) or `RuntimeError` when mandatory

**`test_<name>.py`:** Same as template 4a/4b — binary fixture path comes from CMake conftest fixture.

**Key differences from hipcc porting in the Transformation Summary:**

| Shell pattern | rocm-tests replacement | Reason |
|---|---|---|
| `cmake -B build && cmake --build build` | `cmake_build()` from `tests.common._cmake_build` in session fixture | CMake manages HIP language mode; hipcc cannot |
| `CXX=clang++ cmake ...` | `-DCMAKE_CXX_COMPILER=<rock_dir>/lib/llvm/bin/clang++` | Must use ROCm's clang++ for offload-arch flags |
| `export ROCM_PATH=/opt/rocm` | Pass `env ROCM_PATH={rock_dir}` in the run command | Runtime `.hsaco` path resolution; not compile-time |

---

## Section 5 — Transformation Summary Table

Always include this table in the output:

```markdown
## Transformation Summary

| Source pattern | rocm-tests replacement | Reason |
|---|---|---|
| `export ROCR_VISIBLE_DEVICES=0` | Removed | Injected automatically by `target_executor` |
| `hipcc source.cpp -o /tmp/bin` | `compile_binary()` in `conftest.py` | Session-scoped, xdist-safe, incremental |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic message |
| `echo "TEST_PASSED"` | `assert "TEST_PASSED" in result.stdout` | Explicit sentinel verification |
| Hardcoded `/tmp/binary` path | `<binary_name>_binary: str` fixture | Framework-managed output directory |
| `time.sleep(5)` | Removed | Health checks handle GPU readiness |
| `sys.exit(1)` on missing dep | `pytest.skip("reason")` | Graceful skip vs session abort |
| `logging.info("Running X")` | (optional) `allure_reporter.step("Run X")` | Structured step — add only if requested |
```

Add or remove rows to match what was actually in the source.

---

## Section 6 — File Placement Guide

| Ported source domain | Target directory |
|---|---|
| hipcc compilation, LLVM/HIP codegen | `tests/e2e/compiler/` |
| GPU hardware queue tests | `tests/e2e/hwq_heuristic/` |
| HIP runtime, driver API, multi-stream | `tests/e2e/hip_runtime/` |
| hipBLASLt GEMM, Tensile | `tests/e2e/hipblaslt/` |
| rocPRIM primitives, HMM | `tests/e2e/rocprim/` |
| rocsolver, rocblas, montecarlo | `tests/e2e/rocm_libs/` |
| Config / DryRun / framework unit tests | `tests/dry_run/` |
| New domain | Verify against `framework/markers/taxonomy.py → CATEGORY_PROFILES`; create new directory only after adding the profile |

**Important:** `tests/dry_run/` is for framework-level unit tests (config loading, marker linting, executor contracts). It is **NOT** a landing zone for ported GPU tests that need a CPU-safe "companion."

---

## Section 7 — Rules

**NEVER:**
- Carry over `subprocess.run()` or `subprocess.Popen()` in `test_*.py` — always `target_executor.run()`. In `conftest.py`, `subprocess.run()` is allowed only for CMake build steps (never for test logic).
- Carry over `os.environ["ROCR_VISIBLE_DEVICES"]`, `HIP_VISIBLE_DEVICES`, or `CUDA_VISIBLE_DEVICES`
- Carry over `time.sleep()` — health checks handle readiness
- Carry over `sys.exit(N)` for dependency failures — use `pytest.skip()`
- Use `BaseTestCase` inheritance — rocm-tests uses pure fixture injection
- Import from `framework.plugins` in the ported test file
- Reference `nodes_fixture` — use `target_executor` for all GPU tiers
- Hardcode GPU device paths or indices
- Hardcode the compiled binary path — always use a session-scoped conftest fixture
- Wrap a `.cpp` source in `python3 -c` — compile it to a binary with `hipcc` via `compile_binary`
- Declare markers already auto-injected by the directory's `CATEGORY_PROFILES`
- Merge multiple distinct GPU operations into one test function

**ALWAYS:**
- Generate `conftest.py` + `test_*.py` pair for any C++ source being ported
- Copy or adapt the `.cpp` source into `tests/e2e/<domain>/src/` — never reference its original path
- Use `scope="session"` on every `compile_binary` fixture
- Use `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` for compiled binaries
- Declare `runtime.*` explicitly on every test function
- Show the Transformation Summary table in the output
- Include `Ported from: <source>` in the module docstring
- Add the Copyright header and `SPDX-License-Identifier: MIT` to all generated files
- Create one test function per distinct GPU operation from the source

---

## Example Interaction

```
User: /porter scripts/check_hip_devices.sh

Source:
  #!/bin/bash
  export ROCR_VISIBLE_DEVICES=0
  result=$(python3 -c "import hip; print(hip.hipGetDeviceCount())" 2>&1)
  if [ $? -ne 0 ]; then exit 1; fi
  count=$(echo "$result" | grep -oP '\d+')
  if [ "$count" -lt 1 ]; then echo "ERROR: no HIP devices"; exit 1; fi
  echo "DEVICES_OK: $count"

Analysis:
  → Source type: shell script running a Python command (system binary equivalent — no .cpp)
  → No C++ compilation needed
  → Target: tests/e2e/stack_validation/ (HIP API check)
  → Profile injects: hw.gpu, layer.runtime, ci.nightly, e2e.stack, os.linux
  → Explicit markers: runtime.fast only

Generated: tests/e2e/stack_validation/test_hip_device_count.py (1 file only)

Transformation Summary:
  | export ROCR_VISIBLE_DEVICES=0        | Removed — executor injects             |
  | if [ $? -ne 0 ]; then exit 1; fi    | assert result.ok, f"... {result.stderr}"|
  | echo "$count" | grep -oP '\d+'      | parse_metric(result.stdout, "DEVICES")  |

Validation:
  pytest tests/e2e/stack_validation/test_hip_device_count.py --collect-only -q --no-gpu
```
