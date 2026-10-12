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
2. `framework/plugins/builder_plugin.py` — `compile_binary` fixture signature, `cmake_build_dir` factory, `ld_path` dict, `external_build` fixture
3. `framework/plugins/remote_node_plugin.py` — `target_executor` (what `hw.*`/`e2e.*` markers it reads), `start_background()` API
4. `framework/common/helpers.py` — `ExecutionResult` fields (`.ok`, `.exit_code`, `.stdout`, `.stderr`, `.duration`)
5. The **closest existing test directory** in `tests/e2e/` — read both `conftest.py` and the test file as a structural reference

Optional (read only if the user's feature needs it):

6. `framework/plugins/artifacts_plugin.py` — `allure_reporter` fixture
7. `tests/common/criu/fixtures.py` — when checkpoint-restore is involved
8. `tests/common/ml_provisioning/fixtures.py` — when PyTorch provisioning is needed

---

## Section 2 — Gather Requirement

If not already provided, ask:

> "What GPU operation or feature do you want to test? Include:
> - What the binary should do
> - Expected output or correctness criteria
> - Any GPU resource requirements (VRAM, GPU count)
> - Any ROCm library dependencies (RCCL, rocBLAS, etc.)
> - Any third-party dependencies (external repos, system packages)"

If a requirements document or C++ source is provided, read it completely and identify every **independently testable assertion** — each becomes one test function.

**OSS Compliance Gate — run before generating any code:**

Before writing tests that reference external projects or third-party code, classify the dependency:

| Dependency type | Action required |
|---|---|
| First-party ROCm (ROCm/*, AMD repos) | None — proceed |
| OSS with permissive license (MIT, Apache-2.0, BSD-2/3) | Call `external_build.assert_license_present(src_dir)` in conftest; note the license in the module docstring |
| OSS with copyleft license (GPL, LGPL, AGPL) | Flag to user: linking/bundling may restrict distribution; confirm before proceeding |
| License absent or unknown | `pytest.skip("license not verified")` guard; alert user |
| Proprietary / internal-only / NDA-restricted | STOP — do not generate a test; inform the user |

---

## Section 3 — Choose Target Directory and Understand Profile Injection

`CATEGORY_PROFILES` in `taxonomy.py` auto-injects markers onto every test in a known directory. **You must NOT declare these markers in the test function — they are already applied.**

> **Authoritative source:** Always read `framework/markers/taxonomy.py → CATEGORY_PROFILES` before assigning a target directory. The table below mirrors current entries — new directories may be added there first. When in doubt, read the file.

| Directory | Auto-injected markers | What to declare in test file |
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
| `tests/e2e/ml_frameworks/torchvision/` | `hw.multi_gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` + `gpu_count(N)` |
| `tests/e2e/ml_frameworks/apex/` | `hw.multi_gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` + `gpu_count(N)` |
| `tests/e2e/recovery/criu/` | `hw.gpu`, `layer.runtime`, `ci.weekly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/system_tools/amd_smi/events/` | `hw.gpu`, `layer.runtime`, `ci.weekly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/e2e/hpc/ucx/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` | `runtime.*` only |
| `tests/dry_run/` | none | all 4 required dimensions + `runtime.*` |
| New directory (no profile exists) | none | all 4 required dimensions + `runtime.*` |

**Override rule:** A function-level marker always beats the profile. Use this to escalate a test to `ci.weekly` while keeping the profile's other markers.

**`gpu_count(N)` is a parametric marker — never auto-injected by any profile.** Always declare explicitly on every multi-GPU test function.

---

## Section 4 — Fixture Decision Table

| Test type | Required fixtures | Notes |
|---|---|---|
| GPU E2E — compiled binary | `target_executor`, `ld_path: dict`, `<binary>_binary: str` | Binary fixture declared in conftest.py |
| Multi-GPU — compiled binary | same + `@pytest.mark.gpu_count(N)` | target_executor handles ROCR_VISIBLE_DEVICES |
| System binary (no compilation) | `target_executor` | e.g. `rocm-smi`, `hipconfig`, `amd-smi` |
| External repo build | `external_build`, `target_executor`, `rock_dir` | Clone via `external_build.clone_repo()`; assert license |
| Background process + monitor | `target_executor` + `cpu_executor` | `start_background()` API; structured env state in frozen dataclass |
| DryRun / cpu_only | `dry_run_executor` | Only for `tests/dry_run/` framework unit tests |
| Optional Allure reporting | add `allure_reporter` | Not required in most tests; wrap `target_executor.run()` calls |
| hipBLASLt / Tensile binary | `target_executor`, `ld_path`, `rock_dir`, `arch_lib_path`, binary fixture | `arch_lib_path` resolves `lib/hipblaslt/library/<arch>` |
| CMake-based build | add `gpu_arch: str \| None` to conftest fixture | Use `cmake_build_dir()` factory from `builder_plugin` |
| Pinned GPU indices | `target_executor` + `@pytest.mark.gpu_indices([i, j])` | Bypasses NUMA; list argument required; mutually exclusive with `gpu_count`/`hw.multi_gpu` |
| Manual GPU control in test body | `manual_gpu_allocator` fixture | `alloc.pin(gpu_index=0)` context manager |
| PyTorch workload | `require_torch`, `torch_python`, `target_executor`, `ld_path` | Never import torch on coordinator |
| CRIU checkpoint-restore | `criu` (session fixture from `ensure_criu_runtime`) | See §8 of skills.md |
| Container-mode test | `@pytest.mark.container(extra_run_flags=...)` + `target_executor` | Container options declared as marker, not in conftest |

**Never request fixtures you do not use. Never use deprecated `gpu_fixture`, `local_executor`, or `session_executor`.**

---

## Section 5 — Always Generate TWO Files

For any test that compiles a C++ binary (the primary case), always produce:

1. **`tests/e2e/<domain>/conftest.py`** — session-scoped `compile_binary` fixture(s)
2. **`tests/e2e/<domain>/test_<name>.py`** — the test functions

For tests that only invoke existing system binaries (e.g. `rocm-smi`), conftest.py is not needed.

> **Exception — Python-script-dispatch tests:** Tests that drive existing Python scripts via `target_executor.run(f"{sys.executable} ...")` may use a minimal or empty conftest stub. Only generate fixture code when the test area actually needs compilation or shared session setup. When conftest is empty, include a module docstring explaining why.

> **Exception — background-process tests:** Use a frozen `@dataclass` (not a conftest fixture) to bundle resolved environment state when the test setup is complex. See template 6j.

---

## Section 6 — Code Templates

### 6a. conftest.py — CompileSpec Registry Pattern

Use when adding one or more HIP/C++ binaries to a test area.

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
    <list the profile markers for this directory — see taxonomy.py>

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

`tests/dry_run/` is for **framework unit tests** (config loading, marker linting, executor contract tests). It is NOT a companion directory for GPU tests.

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

```python
# In function signature: add allure_reporter
def test_<name>(target_executor, ld_path: dict, <binary>_binary: str, allure_reporter):
    ld = ld_path["LD_LIBRARY_PATH"]
    with allure_reporter.step("Run <binary_name>"):
        result = target_executor.run(f"env LD_LIBRARY_PATH={ld} {<binary>_binary}")
    assert result.ok, ...
```

### 6h. conftest.py — CMake Build Pattern (for `.hip` sources or multi-target CMakeLists.txt)

Use when `compile_binary`/hipcc cannot handle the sources.

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- CMake build fixtures for tests/e2e/<domain>/.

Uses cmake_build_dir() from builder_plugin which manages clang++ discovery,
cmake configure+build, incremental caching, and GPU_ARCH passthrough.
"""

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
        gpu_arch=gpu_arch,        # None → CMake/hipcc auto-detects
        gpu_arch_var="GPU_ARCH",  # use "AMDGPU_TARGETS" for rocprim-style CMakeLists.txt
        label="<domain>",
    )


@pytest.fixture(scope="session")
def <binary_name>_binary(_domain_cmake_build_dir: str) -> str:
    """Return path to <binary_name> built by CMake."""
    path = os.path.join(_domain_cmake_build_dir, "<binary_name>")
    assert os.path.isfile(path), f"Binary not built: {path}"
    return path
```

**Notes:**
- `cmake_build_dir()` automatically handles `-DROCM_PATH`, `-DCMAKE_PREFIX_PATH`, and the ROCm clang++ compiler.
- Prefix internal build-dir fixtures with `_` to signal they are not for direct test use.
- For optional binaries: omit `assert os.path.isfile` in the fixture; put `if not os.path.isfile(binary): pytest.skip(...)` in the test body instead.

### 6i. test_*.py — Python-Script Dispatch Pattern

Use when tests drive existing Python scripts rather than compiled binaries.

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

**Alternative:** Use `pytest.importorskip("<package>", reason="...")` at **module level** to skip the entire file when a package is unavailable.

### 6j. conftest.py + test_*.py — Background Process with Frozen Dataclass Env

Use when the test setup is complex (resolved binary paths, checked subcommands, scratch directories). Bundle state in a frozen dataclass rather than multiple fixtures.

```python
# conftest.py
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Environment setup for tests/e2e/<domain>/.

Uses a frozen dataclass to bundle all resolved state so test functions
receive a single typed object rather than many positional fixtures.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pytest

logger = logging.getLogger("rocm.test")


@dataclass(frozen=True)
class <Domain>Env:
    binary: str
    subcommand: str
    scratch_dir: str


@pytest.fixture
def <domain>_env(target_executor, tmp_path) -> <Domain>Env:
    """Resolve binary path and subcommand; fail fast if unavailable."""
    binary = "<tool>"
    check = target_executor.run(f"which {binary}")
    if not check.ok:
        pytest.skip(f"{binary} not found on this node")

    sub_check = target_executor.run(f"{binary} <subcommand> --help")
    if not sub_check.ok:
        pytest.fail(f"{binary} <subcommand> subcommand unavailable:\n{sub_check.stderr}")

    scratch = str(tmp_path / "<domain>_scratch")
    target_executor.run(f"mkdir -p {scratch}")
    return <Domain>Env(binary=binary, subcommand="<subcommand>", scratch_dir=scratch)
```

```python
# test_<name>.py
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what it validates>, including background event monitoring.

Uses start_background() for long-running monitor processes.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.<budget>
"""

import logging

import pytest

logger = logging.getLogger("rocm.test")


@pytest.mark.runtime.fast
def test_<name>_with_monitor(target_executor, <domain>_env):
    """<What this test verifies> with a background monitor."""
    env = <domain>_env

    monitor = target_executor.start_background(
        f"{env.binary} <monitor-cmd>",
        log_path=f"{env.scratch_dir}/monitor.log",
        console_label="<monitor>",
    )
    assert monitor.is_alive, "Monitor failed to start"

    trigger = target_executor.run(f"{env.binary} <trigger-cmd>")
    assert trigger.ok, (
        f"<trigger> failed (exit={trigger.exit_code}):\n"
        f"stdout: {trigger.stdout[:2000]}\nstderr: {trigger.stderr[:500]}"
    )

    stop_result = monitor.stop(timeout=10.0)
    assert "<EXPECTED_EVENT>" in stop_result.stdout, (
        f"Monitor did not capture <EXPECTED_EVENT>:\n{stop_result.stdout[:2000]}"
    )
```

### 6k. conftest.py — External Repo Build (Pattern C)

Use when the test needs to build a third-party project from source.

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- External build fixtures for tests/e2e/<domain>/.

Clones <upstream_repo> and builds the required binary.
License: <SPDX-License-Identifier from upstream> — verified by assert_license_present().
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def <external>_binary(external_build, rock_dir: str, framework_config) -> str:
    """Clone and build <external> binary; return absolute path."""
    build_timeout = float(framework_config.therock.build_timeout_secs)
    clone_path = external_build.clone_repo(
        "<upstream_url>",
        "<domain>/<repo_name>",
        ref="<tag_or_sha>",       # pin to tag/SHA for reproducibility
        timeout=build_timeout,
    )
    # Verify license before using any code from the cloned repo
    external_build.assert_license_present(clone_path)

    result = external_build.run(
        f"make -C {clone_path} MPI_ENABLED=0 ROCM_PATH={rock_dir}"
    )
    assert result.ok, f"Build failed:\n{result.stderr}"

    binary = os.path.join(str(clone_path), "build", "<binary_name>")
    assert os.path.isfile(binary), f"Expected binary not produced: {binary}"
    return binary
```

---

## Section 7 — Marker Decision Table

| Question | Answer → marker |
|---|---|
| Single AMD GPU? | `hw.gpu` |
| 2+ GPUs, collective op? | `hw.multi_gpu` + `@pytest.mark.gpu_count(N)` |
| No GPU (framework test) | `hw.cpu_only` |
| Wall time < 5 min? | `runtime.fast` |
| Wall time < 30 min? | `runtime.medium` |
| Hours-long stability test? | `runtime.soak` + `ci.weekly` |
| HIP API, ROCm stack, amd-smi? | `layer.runtime` |
| RCCL, rocBLAS, rocFFT, MIOpen, rocPRIM? | `layer.math_lib` |
| PyTorch, JAX, vLLM, ONNX? | `layer.ml_framework` |
| kernel driver, KFD, amdgpu module? | `layer.driver` |
| Standard GPU test (not soak)? | `ci.nightly` |
| < 5 min, no GPU, no network needed? | `ci.pr` |
| Soak or weekly regression? | `ci.weekly` |
| Linux-specific paths/APIs? | `os.linux` |
| GPU workload needs minimum VRAM? | `@pytest.mark.gpu_vram(N)` |
| Need to pin specific GPU index(es)? | `@pytest.mark.gpu_indices([i, j])` — list required; mutually exclusive with `gpu_count`/`hw.multi_gpu` |
| Test already in a profiled directory? | Do NOT redeclare the profile markers |
| Container runtime options needed? | `@pytest.mark.container(ipc="host", extra_run_flags="...")` |
| Per-test container image override? | `@pytest.mark.container_image("rocm/pytorch:6.3")` |

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
- Use `subprocess.run()` or `subprocess.Popen()` in `test_*.py` files — always `target_executor.run()`. In `conftest.py`, `subprocess.run()` is allowed for CMake-based builds only.
- Set `ROCR_VISIBLE_DEVICES` or `HIP_VISIBLE_DEVICES` — the executor injects them
- Hardcode GPU indices (`device_id = 0`) — `target_executor` manages allocation
- Use `time.sleep()` — health checks handle GPU readiness
- Reference `nodes_fixture` — it does not exist; use `target_executor`
- Import from `framework.plugins` in test files — use fixture injection only
- Invent marker values — only use values from `framework/markers/taxonomy.py → MARKER_SCHEMA`
- Declare `hw.*`, `ci.*`, `layer.*`, `e2e.*`, or `os.*` markers that are already in the directory's `CATEGORY_PROFILES`
- Call `compile_binary()` inside a test function body — always in a `scope="session"` conftest fixture
- Use `python3 -c` to run GPU logic — compile to a binary with `hipcc` via `compile_binary`
- Produce a `hw.cpu_only` DryRun companion for every GPU test — `tests/dry_run/` is for framework unit tests only
- Pass a bare int to `gpu_indices` — use a list: `@pytest.mark.gpu_indices([0])` not `@pytest.mark.gpu_indices(0)`
- Combine `gpu_indices` with `gpu_count` or `hw.multi_gpu` — they are mutually exclusive
- Import `torch` on the coordinator process — run all PyTorch code via `target_executor.run(f"{torch_python} ...")`
- Reference external project code without first calling `external_build.assert_license_present()`
- Include hardcoded credentials, webhook URLs, or API tokens — use `ROCM_TEST_*` environment variables

**ALWAYS:**
- Generate `conftest.py` first, then `test_<name>.py`
- Use `scope="session"` on every `compile_binary` fixture in `conftest.py`
- Use `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` as the primary run command for compiled binaries
- Assert `result.ok` with a diagnostic message (exit code + truncated stdout + stderr)
- Declare `runtime.*` explicitly on every test function (never omit, never in any profile)
- Assert a binary stdout sentinel beyond `result.ok` when the binary emits one
- Write the Copyright header and `SPDX-License-Identifier: MIT` on both files
- Write the module docstring listing the binary source, output path, and which markers are auto-injected
- Call `external_build.assert_license_present(clone_path)` before using any cloned external code
- Add `logger = logging.getLogger("rocm.test")` when the test needs structured logging (background-process tests)

---

## File Placement Guide

| ROCm layer / domain | Target directory |
|---|---|
| hipcc compilation, LLVM/HIP codegen | `tests/e2e/compiler/` |
| GPU hardware queue heuristics | `tests/e2e/hwq_heuristic/` |
| HIP runtime, driver API, multi-stream, IPC | `tests/e2e/hip_runtime/` |
| Catch2-based HIP directed tests | `tests/e2e/hip_directed/` |
| hipBLASLt GEMM, Tensile heuristics | `tests/e2e/hipblaslt/` |
| rocPRIM primitives, HMM | `tests/e2e/rocprim/` |
| rocsolver, rocblas, montecarlo | `tests/e2e/rocm_libs/` |
| ROCm official examples | `tests/e2e/rocm_examples/` |
| kernel driver / KFD, amdgpu module | `tests/e2e/kfd/` |
| RCCL collective communication | `tests/e2e/rccl/` |
| QUDA lattice QCD | `tests/e2e/hpc/quda/` |
| GPU process checkpoint-restore | `tests/e2e/recovery/criu/` |
| PyTorch / torchvision | `tests/e2e/ml_frameworks/<framework>/` |
| amd-smi event monitoring | `tests/e2e/system_tools/amd_smi/events/` |
| Framework unit tests, config, DryRun | `tests/dry_run/` |
| New ROCm feature domain | Create `tests/e2e/<domain>/`; add profile to `framework/markers/taxonomy.py → CATEGORY_PROFILES` first |
