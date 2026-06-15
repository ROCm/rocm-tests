---
name: creator
description: Generate 100% framework-compliant rocm-tests pytest tests from a GPU feature requirement or requirements document
user-invocable: true
---

# Agent: Test Creator

**Objective:** Generate a complete, framework-compliant test from a GPU feature description or requirements document. Always produces two files: a session-scoped `conftest.py` (build fixtures) and one or more `test_*.py` files (test functions).

---

## Section 1 — Framework Grounding

Read these files before writing any code:

1. `framework/markers/taxonomy.py` — `MARKER_SCHEMA`, `REQUIRED_DIMENSIONS`, `CATEGORY_PROFILES`
2. `framework/plugins/builder_plugin.py` — `compile_binary` fixture signature, `ld_path` dict
3. `framework/plugins/remote_node_plugin.py` — `target_executor` (what `hw.*`/`e2e.*` markers it reads)
4. The **closest existing test directory** in `tests/e2e/` — read both `conftest.py` and the test file as a structural reference

Optional (read only if the user's feature needs it):

5. `framework/plugins/artifacts_plugin.py` — `allure_reporter` fixture

---

## Section 2 — Gather Requirement

If not already provided, ask:

> "What GPU operation or feature do you want to test? Include:  
> - What the binary should do  
> - Expected output or correctness criteria  
> - Any GPU resource requirements (VRAM, GPU count)  
> - Any ROCm library dependencies (RCCL, rocBLAS, etc.)"

If a requirements document or C++ source is provided, read it completely and identify every **independently testable assertion** — each becomes one test function.

---

## Section 3 — Choose Target Directory and Understand Profile Injection

`CATEGORY_PROFILES` in `taxonomy.py` auto-injects markers onto every test in a known directory. **You must NOT declare these markers in the test function — they are already applied.**

| Directory | Auto-injected markers | What to declare in test file |
|---|---|---|
| `tests/e2e/compiler/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hwq_heuristic/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hip_runtime/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hipblaslt/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocprim/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/rocm_libs/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/dry_run/` | none | all 4 required dimensions + `runtime.*` |
| New directory (no profile exists) | none | all 4 required dimensions + `runtime.*` |

> **Authoritative source:** Always read `framework/markers/taxonomy.py → CATEGORY_PROFILES` before assigning a target directory. The table above mirrors the current entries — new directories may be added there first.

**Override rule:** A function-level marker always beats the profile. Use this to escalate a test to `ci.weekly` while keeping the profile's other markers.

**`gpu_count(N)` is a parametric marker — never auto-injected by any profile.** `@pytest.mark.gpu_count(N)` must always be declared explicitly on every multi-GPU test function. Without it, `target_executor` does not know how many GPUs to acquire.

---

## Section 4 — Fixture Decision Table

| Test type | Required fixtures | Notes |
|---|---|---|
| GPU E2E — compiled binary | `target_executor`, `ld_path: dict`, `<binary>_binary: str` | Binary fixture declared in conftest.py |
| Multi-GPU — compiled binary | same + `@pytest.mark.gpu_count(N)` | target_executor handles ROCR_VISIBLE_DEVICES |
| System binary (no compilation) | `target_executor` | e.g. `rocm-smi`, `hipconfig` |
| DryRun / cpu_only | `dry_run_executor` | Only for `tests/dry_run/` framework unit tests |
| Optional Allure reporting | add `allure_reporter` | Not required; no existing test uses it |
| hipBLASLt / Tensile binary | `target_executor`, `ld_path`, `rock_dir`, `arch_lib_path`, binary fixture | `arch_lib_path` resolves `lib/hipblaslt/library/<arch>`; set `HIPBLASLT_TENSILE_LIBPATH=$(arch_lib_path(base))` in run command |
| CMake-based build fixture | add `gpu_arch: str \| None` to the conftest fixture signature | Session string from `--gpu-arch`; pass to `cmake_build(gpu_arch=gpu_arch)` and `arch_lib_path()` |
| Pinned GPU indices | `target_executor` + `@pytest.mark.gpu_indices([i, j])` | Bypasses NUMA selection; all indices must be on one node; mutually exclusive with `gpu_count` and `hw.multi_gpu`; argument must be a list |
| Manual GPU control in test body | `manual_gpu_allocator` fixture | Use `alloc.pin(gpu_index=0)` context manager or `alloc.acquire(0)` / `alloc.release(group)` explicitly; fixture teardown auto-detects leaked acquisitions |

**Never request fixtures you do not use. Never use deprecated `gpu_fixture`, `local_executor`, or `session_executor`.**

---

## Section 5 — Always Generate TWO Files

For any test that compiles a C++ binary (the primary case), always produce:

1. **`tests/e2e/<domain>/conftest.py`** — session-scoped `compile_binary` fixture(s)
2. **`tests/e2e/<domain>/test_<name>.py`** — the test functions

For tests that only invoke existing system binaries (e.g. `rocm-smi`), conftest.py is not needed.

> **Exception — Python-script-dispatch tests:** Tests that drive existing Python scripts via `target_executor.run(f"{sys.executable} ...")` may use a minimal or empty conftest stub. Only generate fixture code when the test area actually needs compilation or shared session setup. When conftest is empty, include a module docstring explaining why.

---

## Section 6 — Code Templates

### 6a. conftest.py — CompileSpec Registry Pattern

Use this when adding one or more HIP/C++ binaries to a test area. Mirrors the real pattern in `tests/e2e/compiler/conftest.py`.

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixtures for tests/e2e/<domain>/.

Binary registry
---------------
Each .cpp source is declared as a CompileSpec entry in _SPECS.
To add a binary: (1) add a CompileSpec entry, (2) add a 2-line session fixture.
All compile options live in _SPECS — never scattered across test files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import shlex

import pytest

_SUBDIR = "<domain>"
_COMMON_INCLUDE = "tests/common/include"


@dataclass(frozen=True)
class CompileSpec:
    src: str
    output_name: str
    std: str = "c++17"
    opt: str = "-O2"
    arch: str | None = None
    include_dirs: list[str] = field(default_factory=lambda: [_COMMON_INCLUDE])
    flags: str = ""


_SPECS: dict[str, CompileSpec] = {
    "<key>": CompileSpec(
        src="tests/e2e/<domain>/src/<source>.cpp",
        output_name="<binary_name>",
        # Add domain-specific flags as a space-separated string if needed:
        # flags="-D__HIP_PLATFORM_AMD__ -Wall",
    ),
}


def _build(compile_binary, name: str) -> str:
    spec = _SPECS[name]
    return compile_binary(
        src=spec.src,
        output_name=spec.output_name,
        include_dirs=spec.include_dirs,
        std=spec.std,
        opt=spec.opt,
        arch=spec.arch,
        extra_flags=shlex.split(spec.flags) if spec.flags else None,
        subdir=_SUBDIR,
    )


@pytest.fixture(scope="session")
def <key>_binary(compile_binary) -> str:
    """Compile <source>.cpp via hipcc; return absolute binary path."""
    return _build(compile_binary, "<key>")
```

### 6b. conftest.py — With Library Link Flags

Use when the binary links against ROCm libraries (e.g. RCCL, rocBLAS).

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Build fixture for tests/e2e/<domain>/."""

from __future__ import annotations

import pytest

_SUBDIR = "<domain>"
_SRC = "tests/e2e/<domain>/src/<source>.cpp"
_NAME = "<binary_name>"


@pytest.fixture(scope="session")
def <binary_name>_binary(compile_binary, rock_dir: str) -> str:
    """Compile <source>.cpp via hipcc against <library>; return binary path."""
    return compile_binary(
        src=_SRC,
        output_name=_NAME,
        std="c++17",
        opt="-O3",
        include_dirs=["tests/e2e/<domain>/src"],
        extra_flags=[
            "-Wall",
            "-D__HIP_PLATFORM_AMD__",
            "-isystem", f"{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-l<library>",    # e.g. -lrccl, -lrocblas
            "-lpthread",
            "-lamdhip64",
        ],
        subdir=_SUBDIR,
    )
```

### 6c. test_*.py — Single-GPU (primary pattern)

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what it validates>.

Binary compiled from:
    tests/e2e/<domain>/src/<source>.cpp

Output binary:
    output/test-binaries/<domain>/<binary_name>

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <list the profile markers for this directory — see Section 3>

Explicit markers (not in profile):
    runtime.<budget>
"""

import pytest


@pytest.mark.runtime.<budget>
def test_<name>(
    target_executor,
    ld_path: dict,
    <binary_name>_binary: str,
):
    """<One-line description of what this test verifies>."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {<binary_name>_binary} [--args]"
    )
    assert result.ok, (
        f"<name> failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    # Assert a success sentinel printed by the binary (stronger than exit code alone)
    assert "<PASS_SENTINEL>" in result.stdout, (
        f"<name> did not print <PASS_SENTINEL>:\n{result.stdout[:1000]}"
    )
```

### 6d. test_*.py — Multi-GPU with weekly soak override

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <collective operation> on 2+ GPUs.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    hw.multi_gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers: runtime.*, gpu_count(N).
Weekly variant overrides ci.nightly → ci.weekly.
"""

import pytest


@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
def test_<name>(
    target_executor,
    ld_path: dict,
    <binary_name>_binary: str,
):
    """<Collective operation> sanity on 2 GPUs."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {<binary_name>_binary} <mode>"
    )
    assert result.ok, (
        f"<name> failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )


@pytest.mark.ci.weekly          # overrides profile-injected ci.nightly
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.soak
def test_<name>_weekly(
    target_executor,
    ld_path: dict,
    <binary_name>_binary: str,
):
    """<Collective operation> weekly soak on 2 GPUs."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {<binary_name>_binary} weekly",
        timeout=7200.0,
    )
    assert result.ok, (
        f"<name> weekly failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
```

### 6e. test_*.py — Parametrized over binary CLI arguments

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what it validates> across multiple scenarios.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers: runtime.<budget>
"""

import pytest


@pytest.mark.runtime.<budget>
@pytest.mark.parametrize("<param>", [<val1>, <val2>, <val3>])
def test_<name>(
    target_executor,
    ld_path: dict,
    <binary_name>_binary: str,
    <param>: <type>,
):
    """Validate <feature> for each value of <param>."""
    ld = ld_path["LD_LIBRARY_PATH"]
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {<binary_name>_binary} --<option>={<param>}"
    )
    assert result.ok, (
        f"<name> with <param>={<param>} failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
```

### 6f. test_*.py — DryRun / cpu_only (for tests/dry_run/ ONLY)

`tests/dry_run/` is for **framework unit tests** (config loading, marker linting, executor contract tests). It is NOT a companion directory for GPU tests. Do not produce a DryRun companion for GPU E2E tests.

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — Framework unit test (no GPU required).

No CATEGORY_PROFILES apply — declare all required dimensions explicitly.
"""

import pytest


@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
def test_<name>(dry_run_executor):
    """Verify <framework behavior> without GPU hardware."""
    result = dry_run_executor.run("echo OK")
    assert result.ok
```

### 6g. Optional: Allure reporting (add only when requested)

No existing test uses these by default. Add only if the user explicitly asks for Allure output.

```python
# In function signature: add allure_reporter
def test_<name>(target_executor, ld_path: dict, <binary>_binary: str, allure_reporter):
    ld = ld_path["LD_LIBRARY_PATH"]
    with allure_reporter.step("Run <binary_name>"):
        result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {<binary>_binary}")
    assert result.ok, ...
```

---

### 6h. conftest.py — CMake Build Pattern (for `.hip` sources or multi-target CMakeLists.txt)

Use when `compile_binary`/hipcc cannot handle the sources (`.hip` extension, `enable_language(HIP)`, multiple targets, external GTest suite). All CMake-based conftests import from the shared helper in `tests/common/_cmake_build.py` — **never define an inline `_cmake_build()` function in conftest.**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/<domain>/.

Uses the shared cmake_build() helper from tests/common/_cmake_build.py which
manages clang++ discovery, cmake configure+build, and GPU_ARCH passthrough.
"""

from __future__ import annotations

import os

import pytest

from tests.common._cmake_build import cmake_build, find_rocm_clangpp  # noqa: F401


@pytest.fixture(scope="session")
def _domain_cmake_build_dir(rock_dir: str, gpu_arch: str | None, compiler_build_dir: str) -> str:
    """Build all <domain> binaries via CMake; return build directory path."""
    src = os.path.join(os.path.dirname(__file__), "src")
    out = os.path.join(compiler_build_dir, "<domain>")
    os.makedirs(out, exist_ok=True)
    cmake_build(
        src=src,
        build_dir=out,
        rocm_path=rock_dir,
        gpu_arch=gpu_arch,        # None → CMake/hipcc auto-detects from installed GPUs
        gpu_arch_var="GPU_ARCH",  # use "AMDGPU_TARGETS" for rocprim-style CMakeLists.txt
        label="<domain>",
    )
    return out


@pytest.fixture(scope="session")
def binary_name_binary(_domain_cmake_build_dir: str) -> str:
    """Return path to <binary_name> built by CMake."""
    path = os.path.join(_domain_cmake_build_dir, "<binary_name>")
    assert os.path.isfile(path), f"Binary not built: {path}"
    return path
```

**`cmake_build()` key facts:**
- `find_rocm_clangpp(rocm_path)` probes: `<rocm>/lib/llvm/bin/clang++` (TheRock), `<rocm>/llvm/bin/clang++` (standard), `<rocm>/bin/amdclang++` (packaging variants). Calls `pytest.skip()` when absent.
- Automatically passes `-DROCM_PATH`, `-DCMAKE_PREFIX_PATH`, and `-DCMAKE_CXX_COMPILER` — do not repeat them.
- `gpu_arch_var` defaults to `"GPU_ARCH"`; use `"AMDGPU_TARGETS"` for rocprim-style `CMakeLists.txt`.

**Notes:**
- Prefix internal build-dir fixtures with `_` to signal they are not for direct test use.
- Use one shared CMake build fixture for multiple binary fixtures when they share a `CMakeLists.txt`.
- For optional binaries (system package absent): omit `assert os.path.isfile` in the fixture; add `if not os.path.isfile(binary): pytest.skip(...)` in the test body instead.

---

### 6i. test_*.py — Python-Script Dispatch Pattern

Use when tests drive existing Python scripts rather than compiled binaries. No conftest fixtures needed. Place in a directory that has an appropriate profile in `CATEGORY_PROFILES`, or in a new directory where all 4 required dimensions must be declared explicitly.

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what it validates>.

No C++ compilation. Tests dispatch to Python scripts via target_executor.
Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <read from taxonomy.py CATEGORY_PROFILES for the chosen directory>

Explicit markers:
    runtime.<budget>
"""

import sys

import pytest


@pytest.mark.runtime.medium
def test_name(target_executor):
    """<What this test verifies>."""
    # Pre-flight: gracefully skip if required Python package is not installed
    pkg_check = target_executor.run(f"{sys.executable} -c 'import <package>'")
    if not pkg_check.ok:
        pytest.skip("<package> not installed on this node")

    result = target_executor.run(
        f"{sys.executable} tests/e2e/<domain>/src/<script>.py"
    )
    assert result.ok, (
        f"<name> failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "<PASS_SENTINEL>" in result.stdout, (
        f"<name> did not print <PASS_SENTINEL>:\n{result.stdout[:1000]}"
    )
```

**Alternative:** Use `pytest.importorskip("<package>", reason="...")` at **module level** to skip the entire file when a Python package is unavailable — avoids fixture overhead for the whole file.

---

## Section 7 — Marker Decision Table

| Question | Answer → marker |
|---|---|
| Single AMD GPU? | `hw.gpu` |
| 2+ GPUs, collective op? | `hw.multi_gpu` + `@pytest.mark.gpu_count(N)` |
| No GPU (framework test) | `hw.cpu_only` |
| Wall time < 5 min? | `runtime.fast` |
| Wall time < 30 min? | `runtime.medium` |
| Wall time < 2 hr? | `runtime.longevity` |
| Hours-long stability test? | `runtime.soak` + `ci.weekly` |
| HIP API, ROCm stack? | `layer.runtime` |
| RCCL, rocBLAS, rocFFT, MIOpen? | `layer.math_lib` |
| PyTorch, JAX, vLLM, ONNX? | `layer.ml_framework` |
| kernel driver, amdgpu module? | `layer.driver` |
| rocgdb, rocprof, roctracer? | `layer.debug_stack` |
| Standard GPU test (not soak)? | `ci.nightly` |
| < 5 min, no GPU needed? | `ci.pr` |
| Soak or weekly regression? | `ci.weekly` |
| Linux-specific paths/APIs? | `os.linux` |
| GPU workload needs minimum VRAM? | `@pytest.mark.gpu_vram(N)` |
| Need to pin specific GPU index(es)? | `@pytest.mark.gpu_indices([i, j])` — bypasses NUMA; argument must be a list; mutually exclusive with `gpu_count`/`hw.multi_gpu` |
| Test already in a profiled directory? | Do NOT redeclare the profile markers |

**`runtime.*` is NEVER in any profile. Always declare it explicitly on every test function.**

---

## Section 8 — Validation Steps

Present these commands after generating files:

```bash
# 1. Marker lint (no GPU needed)
python3 -c "
from framework.markers.linter import MarkerLinter
v = MarkerLinter().lint_file('tests/e2e/<domain>/test_<name>.py')
print(MarkerLinter.format_violations(v)) if v else print('Markers OK')
"

# 2. Collection test (no GPU needed)
pytest tests/e2e/<domain>/test_<name>.py --collect-only -q --no-gpu

# 3. DryRun (verifies fixture wiring, no GPU execution)
pytest tests/e2e/<domain>/test_<name>.py --no-gpu -v

# 4. GPU run
pytest tests/e2e/<domain>/test_<name>.py -v --rock-dir=/path/to/rocm

# 5. Four-persona review before opening a PR
# /refiner tests/e2e/<domain>/test_<name>.py
```

---

## Section 9 — Rules

**NEVER:**
- Use `subprocess.run()` or `subprocess.Popen()` in `test_*.py` files — always `target_executor.run()`. In `conftest.py`, `subprocess.run()` is allowed for CMake-based builds when `compile_binary`/hipcc is insufficient (see template 6h).
- Set `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` — the executor injects them
- Hardcode GPU indices (`device_id = 0`) — `target_executor` manages allocation
- Use `time.sleep()` — health checks handle GPU readiness
- Reference `nodes_fixture` — it does not exist; use `target_executor`
- Import from `framework.plugins` in test files — use fixture injection only
- Invent marker values — only use values from `framework/markers/taxonomy.py → MARKER_SCHEMA`
- Declare `hw.*`, `ci.*`, `layer.*`, `e2e.*`, or `os.*` markers that are already in the directory's `CATEGORY_PROFILES`
- Call `compile_binary()` inside a test function body — always in a `scope="session"` conftest fixture
- Use `python3 -c` to run GPU logic — compile the code to a binary with `hipcc` via `compile_binary`
- Define an inline `_cmake_build()` helper in conftest.py — always import `cmake_build` and `find_rocm_clangpp` from `tests.common._cmake_build`
- Produce a `hw.cpu_only` DryRun companion for every GPU test — `tests/dry_run/` is for framework unit tests only
- Pass a bare int to `gpu_indices` — use a list: `@pytest.mark.gpu_indices([0])` not `@pytest.mark.gpu_indices(0)`
- Combine `gpu_indices` with `gpu_count` or `hw.multi_gpu` — they are mutually exclusive

**ALWAYS:**
- Generate `conftest.py` first, then `test_<name>.py`
- Use `scope="session"` on every `compile_binary` fixture in `conftest.py`
- Use `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` as the primary run command for compiled binaries
- Assert `result.ok` with a diagnostic message (exit code + truncated stdout + stderr)
- Declare `runtime.*` explicitly on every test function (never omit, never in any profile)
- Assert a binary stdout sentinel beyond `result.ok` when the binary emits one
- Write the Copyright header and `SPDX-License-Identifier: MIT` on both files
- Write the module docstring listing the binary source, output path, and which markers are auto-injected

---

## File Placement Guide

| ROCm layer / domain | Target directory |
|---|---|
| hipcc compilation, LLVM/HIP codegen | `tests/e2e/compiler/` |
| GPU hardware queue heuristics | `tests/e2e/hwq_heuristic/` |
| HIP runtime, driver API, multi-stream | `tests/e2e/hip_runtime/` |
| hipBLASLt GEMM, Tensile heuristics | `tests/e2e/hipblaslt/` |
| rocPRIM primitives, HMM | `tests/e2e/rocprim/` |
| rocsolver, rocblas, montecarlo | `tests/e2e/rocm_libs/` |
| Framework unit tests, config, DryRun | `tests/dry_run/` |
| New ROCm feature domain | Create `tests/e2e/<domain>/`; add profile to `framework/markers/taxonomy.py → CATEGORY_PROFILES` first |

---

## Example Interaction

```
User: "Test that RCCL AllReduce completes in < 30s on 2 GPUs with correct sum"

Analysis:
  Directory: tests/e2e/rccl_collectives/   (new domain — no existing profile)
  Explicit markers needed: hw.multi_gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux, runtime.medium, gpu_count(2)
  conftest.py: compile_binary with -lrccl -lamdhip64 flags (see template 6b)
  test file: template 6d (multi-GPU with optional weekly soak variant)
  taxonomy.py: add CATEGORY_PROFILES entry for tests/e2e/rccl_collectives/ first

Generated files:
  tests/e2e/rccl_collectives/conftest.py
  tests/e2e/rccl_collectives/test_rccl_allreduce.py

Validation:
  pytest tests/e2e/rccl_collectives/test_rccl_allreduce.py --collect-only -q --no-gpu
  # Expected: 1 test collected (or 2 if weekly variant included)
```
