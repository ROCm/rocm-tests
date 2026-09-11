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
2. `framework/plugins/builder_plugin.py` — `compile_binary` signature, `cmake_build_dir`, `external_build`, `ld_path`
3. `framework/plugins/remote_node_plugin.py` — `target_executor`, `start_background()` API
4. `framework/common/helpers.py` — `ExecutionResult` fields (`.ok`, `.exit_code`, `.stdout`, `.stderr`, `.duration`)
5. The **complete source file** to be ported
6. The **closest existing test area** in `tests/e2e/` — read both `conftest.py` and test file as reference

Optional:

7. `framework/plugins/artifacts_plugin.py` — `allure_reporter` (only if user requests Allure output)

---

## Section 2 — OSS Compliance Gate

Run this gate before transforming any code. It is mandatory regardless of how the source is provided.

| Check | Action |
|---|---|
| Source contains a recognized OSS license header (MIT, Apache-2.0, BSD-2/3, ISC) | Preserve the original license header in the ported C++ source; note license in the Python module docstring |
| Source references an external project | Identify the upstream repo and its license; call `external_build.assert_license_present()` in conftest if the repo is cloned at test time |
| Source has copyleft license (GPL, LGPL, AGPL) | Flag to user before proceeding; linking/bundling may restrict distribution of the test binary |
| License file absent | Add `pytest.skip("license not verified for <upstream>")` guard; alert user |
| Source contains references to proprietary, NDA-restricted, or internal-only identifiers | STOP — do not port; inform the user of the specific identifiers found |
| Source contains hardcoded credentials, API tokens, or webhook URLs | Strip entirely; document that `ROCM_TEST_*` env vars must be used instead |

**Prohibited identifier patterns** (stop porting if found, alert user):

```
NDA | NPI | internal only | confidential | do not distribute | proprietary
```

---

## Section 3 — Source Type Identification

| Source type | What to expect | Generated files |
|---|---|---|
| **C++ gtest program** (`.cpp` with `EXPECT_*`/`ASSERT_*`) | Binary that self-validates; exits 0 on pass | `src/<name>.cpp` + `conftest.py` + `test_<name>.py` |
| **Shell script that compiles + runs a .cpp** | `hipcc` call + `./binary [args]` | `src/<source>.cpp` + `conftest.py` + `test_<name>.py` |
| **Shell script that only runs system binaries** | `rocm-smi`, `hipconfig`, `amd-smi` calls | `test_<name>.py` only |
| **Shell script with background process + event monitor** | Long-running daemon + trigger command + log parsing | `conftest.py` (frozen dataclass env) + `test_<name>.py` |
| **Raw Python with subprocess** | `subprocess.run("rocm-smi ...")` or GPU Python API | `test_<name>.py` only (if system binary); else + `conftest.py` |
| **Non-compliant pytest** | Missing markers, `subprocess.run` in test body | Rewrite in place; retain `.cpp` if referenced |
| **AMD framework runner** | `rccl-tests` launcher, `rocBLAS-bench` scripts | `conftest.py` + `test_<name>.py` |
| **CI YAML step** | Inline bash steps in GitHub Actions | Extract each step → one test function each |
| **External repo + CMake** | `git clone` + `cmake --build` + `./binary` | `conftest.py` (external_build.clone_repo) + `test_<name>.py` |
| **CTest suite** (`CMakeLists.txt` with `add_test()` entries) | `ctest --test-dir <build>` runs 1–N subtests; ctest exits non-zero if any fail | `conftest.py` (`cmake_build_dir`) + single `test_<name>.py` function |

---

## Section 4 — Transformation Logic

### Step 1 — Extract Logic

Read the complete source. For each distinct GPU operation, record:

- **What it does** — the command or API being exercised
- **What it asserts** — expected output, return code, or computed value
- **What it guards** — optional dependencies, platform checks, minimum versions
- **What it sets up** — env vars, binary compilation, file creation, CLI arguments
- **Port timestamp** — record `ported_on: <ISO-8601 date>` in the module docstring
- **Upstream ref** — record `upstream_ref: <URL or local path> @ <commit/tag if known>`

Separate **setup** from **validation**. If the source has multiple distinct GPU operations, create **one test function per operation** — never merge them.

### Step 2 — Pattern Mapping

Apply this transformation table to every external pattern found:

| External pattern | rocm-tests equivalent | Notes |
|---|---|---|
| `subprocess.run(cmd, ...)` | `target_executor.run(cmd)` | Drop `check=True`; use `result.ok` |
| `subprocess.Popen(cmd, ...)` | `target_executor.run(cmd)` | Executor handles Popen internally |
| Long-running daemon + later log parse | `target_executor.start_background(cmd, log_path=..., console_label=...)` + `monitor.stop(timeout=N)` | `stop()` returns `ExecutionResult` |
| `os.environ["ROCR_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Injected automatically by `target_executor` |
| `os.environ["HIP_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Same |
| `hipcc source.cpp -o binary` | `compile_binary()` in `conftest.py` | Session-scoped, incremental, xdist-safe |
| `.hip` source file (not `.cpp`) | `cmake_build_dir()` from builder_plugin via session fixture | hipcc cannot handle `.hip` extension |
| `cmake && make` / `cmake --build` in shell | `cmake_build_dir()` in `conftest.py` | Wrap in session fixture |
| `git clone <url> && cmake && make` | `external_build.clone_repo(url, dest, ref=<pin>)` + build | Pin ref to tag/SHA; assert license |
| `ctest --test-dir <build> [args]` | `target_executor.run(f"ctest --test-dir {build_dir} --output-on-failure -j4", timeout=N)` | ctest exits non-zero if any subtest fails; assert `result.ok` + `"0 tests failed" in result.stdout` |
| `set_tests_properties(... TIMEOUT N)` | `timeout=N` on `target_executor.run()` + matching `runtime.*` marker | Derive `runtime.fast` (<300 s), `runtime.medium` (<1800 s), `runtime.soak` (longer) |
| `PASS_REGULAR_EXPRESSION "<pat>"` | `assert "<pat>" in result.stdout` | Only add if pattern is not already covered by `"0 tests failed"` sentinel |
| `FAIL_REGULAR_EXPRESSION "<pat>"` | `assert "<pat>" not in result.stdout` | Add after `assert result.ok` |
| `ROCM_PATH` exported in environment | Pass as `env ROCM_PATH={rock_dir}` in run command | Runtime path lookup |
| `./binary [args]` | `target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary} [args]")` | `ld` from `ld_path["LD_LIBRARY_PATH"]` |
| `export VARIABLE=value` | Remove; pass as CLI arg or `env VAR=val` in run cmd | Never set GPU env vars in test code |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic |
| `if not shutil.which("tool"): sys.exit(1)` | `pytest.skip("tool not available on this node")` | Graceful skip |
| `try: import X \nexcept ImportError: sys.exit(1)` | `pytest.skip("X not installed")` inside test | Never `sys.exit` |
| `time.sleep(N)` | **Remove entirely** | Health checks handle GPU readiness |
| `assert proc.returncode == 0` | `assert result.ok` + sentinel/metric assertion | Add meaningful stdout check |
| `assert "ERROR" not in output` | `assert "ERROR" not in result.stdout` | Direct string check |
| Hardcoded `/dev/renderD128` | Let executor handle — never hardcode device paths | |
| Hardcoded GPU index `device_id = 0` | Let executor manage allocation | |
| `ROCR_VISIBLE_DEVICES=0 ./binary` (pinned single index) | `@pytest.mark.gpu_indices([0])` + `target_executor.run(binary)` | Executor injects device env |
| Loop over explicit GPU indices (`for idx in [0, 2]`) | `manual_gpu_allocator.pin(gpu_index=idx)` context manager | |
| Complex setup state (binary path, subcommand, scratch dir) | Frozen `@dataclass` in `conftest.py` | Single typed env object per test |
| `logging.info("step X")` | Optional `allure_reporter.step("step X")` | Add only if user requests Allure |
| Shell `${VAR:-default}` | `framework_config.section.field or "default"` | Only for framework config options |
| C++ `EXPECT_EQ(a, b)` | Binary self-validates; Python asserts `result.ok` + `"PASSED" in result.stdout` | Keep gtest assertions in C++ |
| C++ `ASSERT_GT(val, thr)` | Binary self-validates; Python asserts `result.ok` | Same — let gtest exit non-zero |
| gtest binary stdout `[  PASSED  ]` | `assert "PASSED" in result.stdout` | gtest exits 0 and prints PASSED |

### Step 3 — C++ Source Handling (Primary Path)

**Decision: which artifacts to generate?**

| Source type | src/ file | conftest.py | test_*.py |
|---|---|---|---|
| `.cpp` gtest or standalone binary | Copy/adapt into `tests/e2e/<domain>/src/` | YES — `compile_binary` fixture with correct flags | YES |
| `.hip` source (CMake HIP mode) | Copy into `tests/e2e/<domain>/src/` | YES — `cmake_build_dir()` + binary fixture | YES |
| Shell script calling `hipcc source.cpp` | Extract `.cpp` into `tests/e2e/<domain>/src/` | YES — `compile_binary` replicates the hipcc call | YES |
| Shell script running system binaries only | Not needed | Not needed | YES |
| Raw Python using subprocess on system binary | Not needed | Not needed | YES |
| External repo + cmake build | External repo stays external | YES — `external_build.clone_repo()` + license check | YES |

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

**For gtest sources: keep gtest assertions in the C++ binary.** Do NOT translate `EXPECT_*` into Python assertions. Let gtest self-validate and exit non-zero on failure. The Python test asserts `result.ok` and `"PASSED" in result.stdout`.

### Step 4 — Resolve Markers (Profile-Aware)

**Look up `CATEGORY_PROFILES` for the target directory FIRST.** Only declare markers that are NOT already auto-injected. Always read `framework/markers/taxonomy.py → CATEGORY_PROFILES` directly — the taxonomy file is the source of truth.

| Target directory | Auto-injected | What to declare |
|---|---|---|
| `tests/e2e/compiler/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hwq_heuristic/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hip_runtime/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hip_directed/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hipblaslt/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocprim/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocm_libs/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocm_examples/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/kfd/` | `hw.gpu`, `layer.driver`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rccl/` | `hw.multi_gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` + `gpu_count(N)` |
| `tests/e2e/hpc/quda/` | `hw.multi_gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` + `gpu_count(N)` |
| `tests/e2e/recovery/criu/` | `hw.gpu`, `layer.runtime`, `ci.weekly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/system_tools/amd_smi/events/` | `hw.gpu`, `layer.runtime`, `ci.weekly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/dry_run/` or new directory | none | All 4 required dimensions + `runtime.*` |

Estimate wall time from the source to assign `runtime.*`:

| Source wall time | Marker |
|---|---|
| < 5 min | `runtime.fast` |
| 5–30 min | `runtime.medium` |
| 30 min – 2 hr | `runtime.soak` |
| Hours | `runtime.soak` + `ci.weekly` |

### Step 5 — Re-structure Output

Every ported file must have:

- Copyright header + SPDX identifier
- Module docstring with:
  - `Ported from:` — original source path or URL
  - `Upstream ref:` — commit SHA, tag, or "unknown" with date
  - `Ported on:` — ISO-8601 date when the port was created
  - `Validates:` — numbered list of independently testable assertions
  - `OSS license:` — license of any third-party code included or cloned
- `scope="session"` on every `compile_binary` fixture in `conftest.py`
- `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` as the run command for compiled binaries
- `assert result.ok` with a diagnostic message (exit code + truncated stdout + stderr)
- At least one stdout assertion beyond `result.ok` when the binary emits detectable output

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

## Section 5 — Code Templates

### 5a. C++ Gtest Binary Porting (3 files)

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

Ported from:   <source file or external framework>
Upstream ref:  <URL or path> @ <commit/tag or "unknown">
Ported on:     <ISO-8601 date>
OSS license:   <SPDX-License-Identifier from upstream>

Validates:
    1. HIP feature self-validates via gtest binary (exits 0 on all cases pass).

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
    """Run the HIP feature gtest binary on an AMD GPU."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {test_hip_feature_binary}"
    )
    assert result.ok, (
        f"test_hip_feature failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "PASSED" in result.stdout, (
        f"gtest did not report PASSED:\n{result.stdout[:1000]}"
    )
```

### 5b. Shell Script Porting (3 files)

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

Ported from:   scripts/run_hip_test.sh
Upstream ref:  local @ unknown
Ported on:     <ISO-8601 date>

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

### 5c. System Binary / Raw Python Porting (1 file only)

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what the ported script validated>.

Ported from:   <source file path>
Upstream ref:  <URL or path> @ <commit/tag or "unknown">
Ported on:     <ISO-8601 date>

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
    result = target_executor.run("rocm-smi --showid")
    assert result.ok, (
        f"rocm-smi failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "GPU" in result.stdout, (
        f"rocm-smi output missing GPU info:\n{result.stdout[:1000]}"
    )
```

### 5d. CMake-Based Build Porting (`.hip` or multi-target CMakeLists.txt)

**`conftest.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- CMake build fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def _domain_cmake_build_dir(cmake_build_dir, rock_dir: str, gpu_arch: str | None) -> str:
    """Build all <domain> binaries via CMake; return build directory path."""
    return cmake_build_dir(
        src="tests/e2e/<domain>/src",
        subdir="<domain>",
        rocm_path=rock_dir,
        gpu_arch=gpu_arch,
        gpu_arch_var="GPU_ARCH",
        label="<domain>",
    )


@pytest.fixture(scope="session")
def <binary_name>_binary(_domain_cmake_build_dir: str) -> str:
    """Return path to <binary_name> built by CMake."""
    path = os.path.join(_domain_cmake_build_dir, "<binary_name>")
    assert os.path.isfile(path), f"Binary not built: {path}"
    return path
```

### 5e. Background Process / Event Monitor Porting

Use when the source runs a long-lived daemon and then triggers an event to observe.

**`conftest.py`** — frozen dataclass env pattern:

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Environment fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest

logger = logging.getLogger("rocm.test")


@dataclass(frozen=True)
class <Domain>Env:
    tool: str
    subcommand: str
    scratch_dir: str


@pytest.fixture
def <domain>_env(target_executor, tmp_path) -> <Domain>Env:
    """Resolve tool path and subcommand; fail fast if unavailable."""
    tool = "<tool-binary>"
    check = target_executor.run(f"which {tool}")
    if not check.ok:
        pytest.skip(f"{tool} not found on this node")

    sub_check = target_executor.run(f"{tool} <subcommand> --help")
    if not sub_check.ok:
        pytest.fail(f"{tool} <subcommand> unavailable:\n{sub_check.stderr[:500]}")

    scratch = str(tmp_path / "<domain>_scratch")
    target_executor.run(f"mkdir -p {scratch}")
    return <Domain>Env(tool=tool, subcommand="<subcommand>", scratch_dir=scratch)
```

**`test_<name>.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <event monitoring scenario>.

Ported from:   <source file path>
Upstream ref:  <URL or path> @ <commit/tag or "unknown">
Ported on:     <ISO-8601 date>

Validates:
    1. <tool> detects <event> emitted by <trigger>.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.fast
"""

import logging

import pytest

logger = logging.getLogger("rocm.test")


@pytest.mark.runtime.fast
def test_<name>(target_executor, <domain>_env):
    """Verify <tool> captures <event> triggered by <trigger>."""
    env = <domain>_env

    monitor = target_executor.start_background(
        f"{env.tool} {env.subcommand} --monitor-flags",
        log_path=f"{env.scratch_dir}/monitor.log",
        console_label="<event-monitor>",
    )
    assert monitor.is_alive, "Monitor failed to start"

    trigger = target_executor.run(f"{env.tool} <trigger-subcommand>")
    assert trigger.ok, (
        f"<trigger> failed (exit={trigger.exit_code}):\n"
        f"stdout: {trigger.stdout[:2000]}\nstderr: {trigger.stderr[:500]}"
    )

    stop_result = monitor.stop(timeout=15.0)
    assert "<EXPECTED_EVENT>" in stop_result.stdout, (
        f"Monitor did not capture <EXPECTED_EVENT>:\n{stop_result.stdout[:2000]}"
    )
```

### 5f. External Repo Build Porting

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- External build fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def <external>_binary(external_build, rock_dir: str, framework_config) -> str:
    """Clone, license-check, and build <external>; return binary path."""
    build_timeout = float(framework_config.therock.build_timeout_secs)
    clone_path = external_build.clone_repo(
        "<upstream_url>",
        "<domain>/<repo_name>",
        ref="<tag_or_sha>",
        timeout=build_timeout,
    )
    external_build.assert_license_present(clone_path)

    result = external_build.run(
        f"make -C {clone_path} MPI_ENABLED=0 ROCM_PATH={rock_dir}"
    )
    assert result.ok, f"Build of <external> failed:\n{result.stderr}"

    binary = os.path.join(str(clone_path), "build", "<binary_name>")
    assert os.path.isfile(binary), f"Expected binary not produced: {binary}"
    return binary
```

### 5g. CTest Suite Porting (1 or N subtests → 1 pytest function)

Do NOT create one pytest function per `add_test()` entry. Run the whole suite as a unit and let CTest own subtest reporting. Only split if the user explicitly needs per-subtest CI markers or different `runtime.*` budgets.

**`conftest.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Build fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import pytest

_SUBDIR = "<domain>"


@pytest.fixture(scope="session")
def <domain>_build_dir(cmake_build_dir, rock_dir: str, gpu_arch: str | None) -> str:
    """Build <domain> via CMake; return build directory path for ctest."""
    return cmake_build_dir(
        src="tests/e2e/<domain>/src",
        subdir=_SUBDIR,
        rocm_path=rock_dir,
        gpu_arch=gpu_arch,
        gpu_arch_var="GPU_ARCH",
    )
```

**`test_<name>.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — Runs the full <domain> CTest suite.

Ported from:   <source CMakeLists.txt or script path>
Upstream ref:  <URL or path> @ <commit/tag or "unknown">
Ported on:     <ISO-8601 date>

Validates:
    1. All registered CTest subtests pass (ctest exits 0, "0 tests failed").

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.<budget>
"""

import pytest


@pytest.mark.runtime.<budget>
def test_<name>_ctest_suite(<domain>_build_dir: str, target_executor):
    """Run all CTest subtests; fail if any subtest fails."""
    result = target_executor.run(
        f"ctest --test-dir {<domain>_build_dir} --output-on-failure -j4",
        timeout=<N>.0,
    )
    assert result.ok, (
        f"CTest suite failed (exit={result.exit_code}):\n"
        f"{result.stdout[-3000:]}\nstderr: {result.stderr[:500]}"
    )
    assert "0 tests failed" in result.stdout, (
        f"CTest reported test failures:\n{result.stdout[-3000:]}"
    )
```

**`runtime.*` and `timeout=` guidance:**

| Max `set_tests_properties(... TIMEOUT N)` across all subtests | `runtime.*` marker | `timeout=` |
|---|---|---|
| < 300 s | `runtime.fast` | `300.0` |
| 300 – 1800 s | `runtime.medium` | `1800.0` |
| > 1800 s | `runtime.soak` + `ci.weekly` | actual max + 20% buffer |

---

## Section 6 — Transformation Summary Table

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

## Section 7 — File Placement Guide

| Ported source domain | Target directory |
|---|---|
| hipcc compilation, LLVM/HIP codegen | `tests/e2e/compiler/` |
| GPU hardware queue tests | `tests/e2e/hwq_heuristic/` |
| HIP runtime, driver API, multi-stream | `tests/e2e/hip_runtime/` |
| Catch2-based HIP directed tests | `tests/e2e/hip_directed/` |
| hipBLASLt GEMM, Tensile | `tests/e2e/hipblaslt/` |
| rocPRIM primitives, HMM | `tests/e2e/rocprim/` |
| rocsolver, rocblas, montecarlo | `tests/e2e/rocm_libs/` |
| ROCm official examples | `tests/e2e/rocm_examples/` |
| kernel driver / KFD, amdgpu module | `tests/e2e/kfd/` |
| RCCL collective communication | `tests/e2e/rccl/` |
| amd-smi event monitoring | `tests/e2e/system_tools/amd_smi/events/` |
| GPU process checkpoint-restore | `tests/e2e/recovery/criu/` |
| Config / DryRun / framework unit tests | `tests/dry_run/` |
| New domain | Verify against `framework/markers/taxonomy.py → CATEGORY_PROFILES`; create new directory only after adding the profile |

**`tests/dry_run/` is NOT a landing zone for ported GPU tests.**

---

## Section 8 — Rules

**NEVER:**
- Carry over `subprocess.run()` or `subprocess.Popen()` in `test_*.py` — always `target_executor.run()`
- Carry over `os.environ["ROCR_VISIBLE_DEVICES"]`, `HIP_VISIBLE_DEVICES`
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
- Create one pytest function per `add_test()` entry in a CTest suite — run the suite as a unit with `ctest --test-dir {build_dir} --output-on-failure`; only split if the user explicitly requires per-subtest CI markers or different `runtime.*` budgets
- Reference external project code without calling `external_build.assert_license_present()`
- Include proprietary or NDA-restricted identifiers — stop the port and alert the user

**ALWAYS:**
- Run the OSS Compliance Gate (Section 2) before transforming any code
- Generate `conftest.py` + `test_*.py` pair for any C++ source being ported
- Copy or adapt the `.cpp` source into `tests/e2e/<domain>/src/` — never reference its original path
- Use `scope="session"` on every `compile_binary` fixture
- Use `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` for compiled binaries
- Declare `runtime.*` explicitly on every test function
- Show the Transformation Summary table in the output
- Include `Ported from:`, `Upstream ref:`, and `Ported on:` in the module docstring
- Add the Copyright header and `SPDX-License-Identifier: MIT` to all generated files
- Create one test function per distinct GPU operation from the source
- Pin external repo `ref` to a tag or commit SHA for reproducibility
