# Framework Skills & Architecture Manual

A comprehensive reference for Claude Code operating inside `rocm-tests`. Covers framework architecture, all agentic skills and their internal processes, compiler and provisioning patterns, shared test utilities, scheduling, and contribution standards. Read this before writing, reviewing, or porting any test code.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Workflow & Lifecycle](#2-workflow--lifecycle)
3. [Agent Skills](#3-agent-skills)
4. [Dynamic Scheduling](#4-dynamic-scheduling)
5. [Compiler Patterns — Building Test Binaries](#5-compiler-patterns--building-test-binaries)
6. [Shared Test Utilities (`tests/common/`)](#6-shared-test-utilities-testscommon)
7. [PyTorch Provisioning](#7-pytorch-provisioning)
8. [CRIU Checkpoint-Restore Support](#8-criu-checkpoint-restore-support)
9. [Debugging Standards](#9-debugging-standards)
10. [Contribution Quality Gates](#10-contribution-quality-gates)

---

## 1. System Overview

`rocm-tests` is a **pytest-based system end-to-end test framework**. It validates the full ROCm software stack — kernel driver → HIP runtime → compute libraries → ML frameworks — on real AMD GPU hardware (nightly/weekly CI). Tests can also run in DryRun mode (no GPU required) for PR validation.

### Architecture Layers

```
┌──────────────────────────────────────────────────────────────────────────┐
│  pytest invocation (CLI / CI / Claude Code skills)                       │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  conftest.py — Plugin Registration                               │    │
│  │   MarkDecorator.__getattr__ patch (enables @pytest.mark.ci.pr)   │    │
│  │                                                                  │    │
│  │  Plugin Stack — load order is contractual:                       │    │
│  │    [1] markers_plugin   → CATEGORY_PROFILES injection            │    │
│  │    [2] session_plugin   → framework_config, run_ctx, log attach  │    │
│  │    [3] gpu_plugin       → --no-gpu / --gpu-arch / --mock-gpu     │    │
│  │    [4] remote_node_plugin → NodePool fleet, target_executor      │    │
│  │    [5] scheduling_plugin → DynamicScheduler, xdist ordering      │    │
│  │    [6] executor_plugin  → container_executor, cpu_executor       │    │
│  │    [7] os_plugin        → os_adapter, platform_name, skip hook   │    │
│  │    [8] health_plugin    → pre/post GPU health gates              │    │
│  │    [9] artifacts_plugin → allure_reporter, GPU dump on failure   │    │
│  │   [10] prereqs_plugin   → session-level ROCm version checks      │    │
│  │   [11] retry_plugin     → --retry-count, per-test retry          │    │
│  │   [12] reports_plugin   → Allure labels, terminal summary        │    │
│  │   [13] builder_plugin   → compile_binary, cmake_build_dir, ld_path│   │
│  │   [14] install_plugin   → --pre-install rocm/pkg/pytorch         │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  framework/ — Core Engine                                        │    │
│  │    config/      ← rocm-test.toml → env vars → CLI flags cascade  │    │
│  │    common/      ← ExecutionResult, Outcome, parse_metric(),      │    │
│  │                   executor_log_path(), gpu_monitor_log_path()    │    │
│  │    executors/   ← AbstractExecutor + 6 concrete backends         │    │
│  │    nodes/       ← NodePool fleet: NodeSlot, GpuFileLock          │    │
│  │    scheduling/  ← DynamicScheduler, SchedulePolicy               │    │
│  │    builder/     ← BinaryBuilder (hipcc, xdist-safe)              │    │
│  │    gpu/         ← GpuDetector, GpuAllocator, BackgroundMonitor   │    │
│  │    markers/     ← MARKER_SCHEMA taxonomy + MarkerLinter          │    │
│  │    reporting/   ← AllureReporter, step(), report_metric()        │    │
│  │    os_adapter/  ← Linux + Windows GPU enumeration                │    │
│  │    rocm/libs/   ← hip.py, rccl.py, amd_smi.py, stack.py         │    │
│  └──────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  tests/ — Test Suite                                             │    │
│  │    common/                ← shared utilities (NOT test files)    │    │
│  │      factories.py         ← fake_gpu_info(), fake_execution_result() │ │
│  │      spirv.py             ← assert_spirv_offload_bundle()        │    │
│  │      ml_provisioning/     ← PyTorch provisioning engine          │    │
│  │      criu/                ← CRIU checkpoint-restore helpers      │    │
│  │    dry_run/               ← ci.pr, hw.cpu_only tests (no GPU)    │    │
│  │    e2e/                                                          │    │
│  │      compiler/            ← hipcc, SPIR-V compilation            │    │
│  │      hip_runtime/         ← HIP driver API, stream, IPC          │    │
│  │      hip_directed/        ← Catch2-based HIP directed tests      │    │
│  │      hipblaslt/           ← GEMM heuristics, shape-boundary      │    │
│  │      hwq_heuristic/       ← GPU hardware queue tests             │    │
│  │      rccl/                ← RCCL collective communication        │    │
│  │      rocm_examples/       ← ROCm official examples suite         │    │
│  │      rocm_libs/           ← rocsolver, rocblas, montecarlo       │    │
│  │      rocprim/             ← rocPRIM primitives, multi-GPU HMM    │    │
│  │      hpc/quda/            ← QUDA lattice QCD tests               │    │
│  │      recovery/criu/       ← GPU process checkpoint-restore       │    │
│  │      ml_frameworks/       ← PyTorch on ROCm                      │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### Plugin Load Order — Two Hard Constraints

1. **`markers_plugin` must be first.** It injects `hw.*`, `ci.*`, and `layer.*` markers from `CATEGORY_PROFILES` during `pytest_collection_modifyitems`. `scheduling_plugin` and `gpu_plugin` both read those markers and must see them fully applied.

2. **`session_plugin` must be second.** It provides `framework_config` and `run_ctx` — foundational session fixtures consumed by `remote_node_plugin`, `executor_plugin`, `health_plugin`, `builder_plugin`, and `install_plugin`. Plugin fixtures are globally visible regardless of directory layout; conftest-defined fixtures are scoped only to tests below that conftest's directory.

### Executor Hierarchy

All GPU tests receive a `NodeExecutorGroup` from `target_executor`. Test code never branches on executor type.

| Executor | Role | When active |
|---|---|---|
| `DryRunExecutor` | Synthetic stub; always `exit_code=0` | `--no-gpu` / `hw.cpu_only` |
| `CpuExecutor` | Real subprocess, no GPU env | `hw.cpu_only` needing real commands |
| `LocalExecutor` | Subprocess + `ROCR_VISIBLE_DEVICES` | Local `hw.gpu` / `hw.multi_gpu` |
| `ContainerExecutor` | Docker/Podman + AMD device passthrough | `--container-mode` |
| `SshExecutor` | SSH + `ROCR_VISIBLE_DEVICES` injection | Remote `hw.gpu` / `hw.multi_gpu` |
| `NodeExecutorGroup` | Uniform container wrapping 1 or N executors | Always — the type `target_executor` yields |
| `BackgroundProcess` | Thread-safe daemon; `.is_alive`, `.stop()` → `ExecutionResult` | `executor.start_background(cmd, log_path=...)` |
| `NoOpBackgroundProcess` | Stub; same API, never alive | `DryRunExecutor.start_background()` |

**Fixture decision guide:**

| Test markers | `target_executor` yields | Test code pattern |
|---|---|---|
| `hw.gpu` | `NodeExecutorGroup(1 exec)` | `target_executor.run(cmd)` |
| `hw.multi_gpu` + `gpu_count(N)` | `NodeExecutorGroup(1 exec, ROCR=0,1,...)` | `target_executor.run(cmd)` |
| `e2e.multinode` + `gpu_count(N)` | `NodeExecutorGroup(N execs, 1 per node)` | `for e in target_executor: e.run(cmd)` |
| `--no-gpu` (any) | `NodeExecutorGroup(DryRunExecutor)` | `target_executor.run(cmd)` |

**Never set `ROCR_VISIBLE_DEVICES` in test code** — executor injects it automatically.

### Config Cascade

Priority order (lowest → highest):

```
Code defaults → rocm-test.toml → ROCM_TEST_* env vars → pytest CLI flags
```

Section-to-dataclass: `[framework]` → `FrameworkSection`, `[gpu]` → `GpuSection`, `[therock]` → `TheRockSection`, `[reporting]` → `ReportingSection`.

---

## 2. Workflow & Lifecycle

### Bootstrap

```bash
git clone https://github.com/ROCm/rocm-tests.git
cd rocm-tests
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # or: uv pip install -r requirements-dev.txt

# Verify wiring before writing code
pytest tests/ --collect-only -q --no-gpu   # must collect without errors
pytest tests/ -m "ci.pr" --no-gpu -v       # must pass (PR gate)

# Start Claude Code — all skills auto-load from .claude/agents/
claude
```

No `/init` required. `CLAUDE.md`, `.claude/agents/*.md`, and `.claude/settings.json` load automatically.

### Execution Pipeline (per test)

```
Test Selected
    │
    ▼
Config Load                   (once per session — framework_config fixture)
    │
    ▼
Prereq Check                  (ROCm version, driver, GPU count)
    │── FAIL ───────────────→  SESSION ABORT
    │
    ▼
GPU Acquire                   (GpuAllocator semaphore pool; blocks until slot free)
    │
    ▼
GPU Health Pre-check          (temp, ECC errors, VRAM free, clock state)
    │── HEALTH_FAIL ─────────→  Outcome = HEALTH_FAIL (hardware, not test logic)
    │
    ▼
Execute Command               (target_executor.run(cmd) / start_background())
    │── Timeout ─────────────→  Outcome = TIMEOUT
    │
    ▼
Artifact Capture              (on failure: GPU dump, lsmod, stdout/stderr)
    │
    ▼
GPU Health Post-check
    │── HEALTH_FAIL ─────────→  Outcome = HEALTH_FAIL
    │
    ▼
Outcome Classification        (PASS / FAIL / TIMEOUT / KILLED / ERROR /
                               HEALTH_FAIL / PERF_DROP / REGRESSION)
    │
    ├── PASS + perf test ────→  Baseline Compare → PERF_DROP if out-of-band
    ├── FAIL + retries left ─→  Re-run with artifact capture; tag FLAKY if later pass
    │
    ▼
GPU Release → Allure JSON written → Session log updated
```

### Output Locations

| Artifact | Location |
|---|---|
| Session log | `output/artifacts/session.log` |
| Allure results | `output/artifacts/allure-results/` |
| Executor logs | `output/artifacts/executor-logs/` |
| Compiled binaries | `output/test-binaries/<subdir>/` |
| GPU info | `output/artifacts/gpu-info-<node>.log` |
| Runtime data | Path passed to `--collect-runtimes` |

---

## 3. Agent Skills

Three built-in skills are accessible via slash commands. Each is backed by a sub-agent definition in `.claude/agents/`. Agents read the live framework source before producing output — they never invent marker values or fixture names.

### Automated Marker Lint (PostToolUse Hook)

`.claude/settings.json` runs `MarkerLinter` automatically every time Claude writes or edits a file under `tests/`. You will see `Marker lint: OK` or a violation list with the exact function and missing dimension before proceeding.

---

### `/creator` — Generate a complete test file

**What it does:** Produces a full, marker-compliant pytest test file from a natural-language GPU feature description or requirements document. Each independently testable assertion becomes its own test function.

**Internal process (6 steps):**

1. **Gather requirement** — reads the description; prompts if not provided.
2. **Resolve markers** — applies the decision table below to every dimension.
3. **Declare resources** — adds `@pytest.mark.gpu_vram(N)` / `@pytest.mark.gpu_count(N)` / `@pytest.mark.container_image(...)` where needed.
4. **Select fixtures** — `target_executor` for GPU tests; `dry_run_executor` for `hw.cpu_only`.
5. **Write file** — copyright + module docstring (`Validates:` list) + module-level script constants + `allure_reporter.step()`-wrapped executor calls + `@pytest.mark.parametrize` where multi-value.
6. **Validate + next-steps checklist** — collect-only → DryRun → GPU run.

**Marker decision table:**

| Dimension | Decision Rule |
|---|---|
| `layer.*` | `runtime`: HIP API / driver; `math_lib`: rocBLAS/RCCL/rocFFT; `ml_framework`: PyTorch on ROCm |
| `ci.*` | `pr`: fast + DryRun-safe (< 5 min, no GPU download); `nightly`: typical E2E; `weekly`: soak |
| `hw.*` | `gpu`: one GPU; `multi_gpu`: two or more; `cpu_only`: DryRun / framework tests |
| `runtime.*` | `fast` (<5 min); `medium` (<30 min); `soak` (hours) |
| `os.*` | `linux` for all current E2E tests |
| `e2e.*` | `stack`: full-stack validation; `multinode`: multi-node collectives |

**Rules enforced by the creator agent:**
- Never `subprocess.run()` or `subprocess.Popen()` — always `target_executor.run(cmd)`
- Never set `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` — executor injects automatically
- Never `time.sleep()` — health checks handle GPU readiness
- Never use `nodes_fixture` — use `target_executor` for all GPU tiers
- Never `from framework.plugins import ...` — use fixture injection only
- Never import `torch` on the coordinator process — run all PyTorch code via `target_executor.run(f"{torch_python} ...")`
- Always: module docstring with numbered `Validates:` list
- Always: `allure_reporter.step()` wrapping every `target_executor.run()` call
- Always: strong assertion (`parse_metric()` + threshold) — `exit_code == 0` alone is weak

**Example:**
```
/creator
> Validate that rocBLAS SGEMM completes without error on gfx1100 with a 4096×4096 matrix

→ layer: math_lib, ci: nightly, hw: gpu, runtime: medium, os: linux
→ Creates: tests/e2e/rocm_libs/test_rocblas_sgemm.py
```

---

### `/refiner [review-as <persona>] <file>` — Review and extend an existing test

**What it does:** Operates in two modes:

- **Review** (default): Applies the 4-persona checklist, runs marker lint, and reports top-3 improvements with before/after code.
- **Extend** (when user says "add", "extend", or describes a new variant): Adds test functions or parametrize — never removes or renames existing functions.

**Internal process — Review:**

1. Reads the target file, `framework/markers/taxonomy.py`, `framework/markers/linter.py`, and `framework/plugins/artifacts_plugin.py`.
2. Runs marker lint — surfaces violations per dimension per function.
3. Applies all four persona checklists (or a specific persona if requested).
4. Ranks top-3 improvements with concrete before/after code.

**Anti-patterns flagged:**

| Category | Pattern |
|---|---|
| ERROR | `time.sleep(N)`, `os.environ["ROCR_VISIBLE_DEVICES"]`, `subprocess.run()`, `import torch` at module level, `from framework.plugins import` |
| ERROR | `nodes_fixture`, hardcoded `/dev/renderD128`, `sys.exit()` in test body |
| WARNING | Assertion only on `result.exit_code`; no stdout threshold check |
| WARNING | ML test with no NaN/Inf guard; no `pytest.skip` for optional prereq |
| WARNING | `ci.pr` + `runtime.medium` conflict; test downloading models marked `ci.pr` |

**Four review personas:**

#### `developer`
GPU API correctness, assertion strength, HIP invocation patterns.
- `target_executor.run(cmd)` must be wrapped in `allure_reporter.step()` for Allure traceability
- Assertion quality: `exit_code == 0` alone is WEAK — `parse_metric()` + threshold is STRONG
- Wrong precision (`f32_r` vs `f64_r`; `torch.float32` vs `torch.float64`)
- Edge cases: VRAM near limit, multi-GPU rank interactions, thermal throttle behavior

#### `tester`
Coverage uniqueness, missing failure modes, parametrize opportunities.
- What if the required library is not installed? → `pytest.skip`, not crash
- What if VRAM is insufficient? → clear error message, not hang
- Assertion quality scale: `exit_code==0` (WEAK) → sentinel string (MEDIUM) → `parse_metric()` + threshold (STRONG) → NaN/Inf guard (STRONGEST)
- Parametrize over: GPU arch, input sizes, data types (f16/f32/f64/bf16), batch sizes

#### `automation`
Marker accuracy, runtime weight vs actual wall time, CI gate placement.
- `ci.pr` + `runtime.medium` = CONFLICT — medium tests must be `ci.nightly` or higher
- `hw.multi_gpu` without `e2e.multinode` → missing Allure grouping for collective tests
- Wrong `runtime.*` weight misleads `DynamicScheduler` → longer nightly wall time
- Tests downloading models or requiring network access must NOT be `ci.pr`

#### `devops`
VRAM requirements, prerequisite declarations, health gate impact, artifact volume.
- gfx1100 (RX 7900 XTX): 24 GB VRAM; gfx942 (MI300X): 192 GB VRAM — safe on MI300X may OOM on gfx1100
- Missing `@pytest.mark.gpu_vram(N)` when workload needs a minimum VRAM threshold
- Soak tests logging per-second stdout can generate GB of artifacts — use `ci.weekly` to gate them out of nightly

**Extension types — Extend mode:**

| User request | Pattern applied |
|---|---|
| "multi-GPU variant" | New function: `hw.multi_gpu` + `e2e.multinode`; still `target_executor` |
| "test more sizes" / "parametrize" | `@pytest.mark.parametrize(...)` on new function |
| "negative test" / "what if it fails" | New function: `hw.cpu_only` + `dry_run_executor`; assert non-zero exit |
| "soak variant" / "run longer" | New function: `ci.weekly` + `runtime.soak` + explicit `timeout` arg |

**Usage:**
```bash
/refiner tests/e2e/rocm_libs/test_rocblas_sgemm.py        # full 4-persona review
/refiner review-as developer tests/e2e/compiler/test_hipcc.py
/refiner tests/e2e/hip_runtime/test_multi_stream.py add a soak variant
```

**Output format (Review mode):**
```markdown
## Refine: tests/e2e/<area>/test_<name>.py

### Marker Lint
✅ All required dimensions present
OR
❌ VIOLATION: test_foo(): Missing required marker dimension: ci

### Developer  [finding or ✓]
### Tester     [finding or ✓]
### Automation [finding or ✓]
### DevOps     [finding or ✓]

## Top 3 Improvements
### 1. [Title] — Why: ... / Before (line N) + After code
```

---

### `/porter <source-file>` — Port an external test into rocm-tests

**What it does:** Takes an external test — shell script, raw Python, non-compliant pytest, C++ gtest — and rewrites it as a fully framework-compliant rocm-tests pytest file.

**Internal process (5 steps):**

1. **Identify Logic** — reads source; records each operation, assertion, and guard.
2. **Map Capabilities** — applies the transformation table to every external pattern.
3. **Resolve Markers** — determines `hw/ci/layer/runtime/os` for each extracted test case.
4. **Re-structure** — writes copyright + module docstring + module-level scripts + `allure_reporter.step()` + `parse_metric()`. One test function per independently testable assertion.
5. **Validate** — `--collect-only` to confirm pytest discovers the ported test; shows transformation summary table.

**Transformation table:**

| External Pattern | rocm-tests Replacement | Reason |
|---|---|---|
| `subprocess.run(cmd)` | `target_executor.run(cmd)` | Executor handles env, logging, timeout |
| `os.environ["ROCR_VISIBLE_DEVICES"] = "0"` | Removed | Injected automatically by executor |
| `if not shutil.which("tool"): sys.exit(1)` | `pytest.skip("tool not available")` | Graceful skip vs session abort |
| `try: import torch \nexcept ImportError: sys.exit(1)` | `pytest.skip("PyTorch not installed")` | Never sys.exit; use pytest.skip |
| `time.sleep(N)` | Removed | Health checks handle GPU readiness |
| `assert proc.returncode == 0` | `assert result.ok` + `parse_metric()` | Stronger assertion; metric in Allure |
| Hardcoded `/dev/renderD128` | `os_adapter.list_gpu_device_paths()[0]` | Never hardcode device paths |
| `logging.info("step X")` | `allure_reporter.step("step X")` | Structured observability in Allure |
| C++ `EXPECT_EQ(a, b)` | `assert a == b, f"Expected {b}, got {a}"` | Direct translation |
| Shell `${VAR:-default}` | `framework_config.section.field or "default"` | Config cascade replaces shell defaults |

---

## 4. Dynamic Scheduling

`DynamicScheduler` distributes GPU tests efficiently across an xdist worker pool. It is a **no-op when `--no-gpu` is active** (`config._node_pool is None`).

### xdist_group Assignment

| Test type | Detection | Assigned `xdist_group` |
|---|---|---|
| Multinode | `@pytest.mark.e2e.multinode` | `"multinode_0"`, `"multinode_1"`, … (unique per test) |
| Multi-GPU | `@pytest.mark.hw.multi_gpu` or `gpu_count(N>1)` | `"multi_gpu_{count}_{idx}"` (unique per test) |
| Single-GPU | `@pytest.mark.hw.gpu` only | None — worksteal across free workers |

Unique groups allow separate xdist workers to run different multi-GPU tests in parallel — each worker holds its own GPU file locks simultaneously.

### Sort Policies

**`resource-most`** (default): multinode first → multi-GPU by count DESC → single-GPU. Heavy multi-GPU tests start while single-GPU tests fill idle slots via worksteal.

**`resource-least`**: single-GPU first → multi-GPU by count ASC → multinode. Use for fast-feedback smoke runs.

### VRAM Headroom

`@pytest.mark.gpu_vram(N)` declares minimum VRAM in GB. The allocator enforces:

```
assignable = (total_vram_gb - headroom_gb) >= test_gpu_vram_requirement
```

Architecture VRAM reference: gfx1100 (RX 7900 XTX) = 24 GB; gfx942 (MI300X) = 192 GB.

### CLI Reference

| Scenario | Command pattern |
|---|---|
| Local DryRun | `pytest tests/ --no-gpu` |
| Local single GPU, nightly | `pytest tests/e2e/ -m "hw.gpu and ci.nightly"` |
| Remote fleet (default policy) | `pytest tests/e2e/ --remote-node host.yaml -n 8` |
| Remote fleet, fast-feedback | `pytest tests/e2e/ --remote-node host.yaml -n 8 --schedule-policy resource-least` |
| VRAM guard + fleet | `pytest tests/e2e/ --remote-node host.yaml -n 8 --vram-headroom-gb 4.0` |
| Audit runtimes | `pytest tests/e2e/ --remote-node host.yaml -n 8 --collect-runtimes output/runtimes.json` |

### Debugging Scheduling

Enable `log_level = "debug"` in `rocm-test.toml` or set `ROCM_TEST_FRAMEWORK_LOG_LEVEL=debug` to see per-item `xdist_group` assignment logs. Use `--collect-only -q --no-gpu` to preview the sorted test order without running.

---

## 5. Compiler Patterns — Building Test Binaries

All GPU tests that need compiled binaries use one of three patterns. Choose based on source type and build complexity. Never call `hipcc` directly from test code.

### Pattern A: Single C++ source via `compile_binary`

Use for a single `.cpp` file compiled with `hipcc`. The result is an xdist-safe incremental build (file-lock protected).

```python
# tests/e2e/my_area/conftest.py
import pytest

@pytest.fixture(scope="session")
def my_kernel(compile_binary):
    return compile_binary(
        src="tests/e2e/my_area/src/kernel.cpp",
        output_name="kernel",
        subdir="my_area",      # → output/test-binaries/my_area/kernel
        arch="gfx942",         # optional; None = hipcc auto-detects
    )
```

```python
# tests/e2e/my_area/test_kernel.py
@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.fast
def test_kernel_launches(my_kernel, target_executor, allure_reporter):
    with allure_reporter.step("Run compiled kernel"):
        result = target_executor.run(f"{my_kernel} --mode verify")
    assert result.ok, result.stderr
```

### Pattern B: CMake build via `cmake_build_dir`

Use when `.hip` file extensions require `enable_language(HIP)`, or multiple binaries share one `CMakeLists.txt`, or GTest targets are needed.

```python
# tests/e2e/my_area/conftest.py
import os, pytest

@pytest.fixture(scope="session")
def my_binary(cmake_build_dir, rock_dir, gpu_arch):
    build = cmake_build_dir(
        src="tests/e2e/my_area/src",
        subdir="my_area",
        rocm_path=rock_dir,
        gpu_arch=gpu_arch,
        gpu_arch_var="AMDGPU_TARGETS",   # or "GPU_ARCH" per CMakeLists convention
    )
    return os.path.join(build, "my_binary")
```

The `cmake_build_dir` factory handles:
- `find_rocm_clangpp()` — locates the ROCm-bundled `clang++`
- Incremental build caching via fingerprint on source tree + arch
- xdist safety (same file-lock mechanism as `BinaryBuilder`)

### Pattern C: Third-party repo via `external_build.clone_repo`

Use when the test needs to build a third-party project from source (e.g., CRIU, rccl-tests, a custom benchmark). `clone_repo` places the checkout in the managed workspace and is xdist-safe.

```python
# tests/e2e/my_area/conftest.py
import pytest

@pytest.fixture(scope="session")
def rccl_tests_binary(external_build, target_executor, rock_dir):
    clone_path = external_build.clone_repo(
        "https://github.com/ROCm/rccl-tests.git",
        "rccl-tests/main",         # subdir under managed workspace
        ref="main",
    )
    result = target_executor.run(f"make -C {clone_path} MPI_ENABLED=0")
    assert result.ok, result.stderr
    return str(clone_path / "build" / "all_reduce_perf")
```

**Rules for Pattern C:**
- The clone runs on the host/SSH coordinator — it does not reach inside containers. For container targets, transfer the built binary or use the CRIU container pattern (see §8).
- Always validate the clone with `assert result.ok` before using the path.
- Pin the `ref` to a tag or commit SHA for reproducible CI builds.

### `CompileSpec` Registry Pattern

When an area has many compiled binaries, register them all in a single `_SPECS` dict and generate fixtures programmatically. This avoids duplicated `compile_binary` call sites:

```python
# tests/e2e/my_area/conftest.py
from dataclasses import dataclass
from typing import Optional
import pytest

@dataclass
class CompileSpec:
    src: str
    output_name: str
    arch: Optional[str] = None

_SPECS = {
    "kernel_a": CompileSpec(src="tests/e2e/my_area/src/a.cpp", output_name="kernel_a"),
    "kernel_b": CompileSpec(src="tests/e2e/my_area/src/b.cpp", output_name="kernel_b", arch="gfx942"),
}

def _build(name, compile_binary):
    spec = _SPECS[name]
    return compile_binary(src=spec.src, output_name=spec.output_name,
                          subdir="my_area", arch=spec.arch)

@pytest.fixture(scope="session")
def kernel_a(compile_binary): return _build("kernel_a", compile_binary)

@pytest.fixture(scope="session")
def kernel_b(compile_binary): return _build("kernel_b", compile_binary)
```

---

## 6. Shared Test Utilities (`tests/common/`)

`tests/common/` is import-only — `norecursedirs` prevents pytest from collecting it as tests. Import from here; never copy logic into test files.

### `tests/common/factories.py` — Synthetic test objects

Use for unit tests that need GPU or executor objects without real hardware:

```python
from tests.common.factories import fake_gpu_info, fake_execution_result

def test_classifier_on_fail():
    result = fake_execution_result(exit_code=1, stderr="hipErrorOutOfMemory")
    outcome = classify(result)
    assert outcome == Outcome.FAIL
```

### `tests/common/spirv.py` — SPIR-V validation

```python
from tests.common.spirv import assert_spirv_offload_bundle

def test_spirv_offload(my_spirv_binary, target_executor):
    result = target_executor.run(f"llvm-spirv --dump {my_spirv_binary}")
    assert_spirv_offload_bundle(result.stdout, expected_arch="gfx942")
```

### `tests/common/ml_provisioning/` — PyTorch provisioning engine

Exports four fixtures via `tests/conftest.py`: `pytorch_env`, `torch_python`, `require_torch`, `require_torch_tunableop`. See **§7 PyTorch Provisioning** for the full workflow.

Key files:
- `fixtures.py` — three-phase fixture logic; sanity cache; pre-install result promotion
- `spec.py` — CLI/config spec parsing; mode constants; `gfx_family_for_arch()` mapping
- `provisioner.py` — venv creation, channel resolution, uv/pip install, sanity validation
- `providers.py` — package names; node-side sanity snippet
- `workload.py` — `workload_failure_detail()` for `hipErrorInvalidImage` diagnostics

### `tests/common/criu/` — CRIU checkpoint-restore

Provides `ensure_criu_runtime` (host/SSH path) and `ensure_criu_runtime_target` (container path) plus the `CRIU` command prefix constant. See **§8 CRIU Checkpoint-Restore Support** for the full workflow.

Key files:
- `fixtures.py` — session-scoped `criu_runtime` fixtures consuming `ensure_criu_runtime` / `ensure_criu_runtime_target`
- `installer.py` — standalone installer; can be run manually on a node before tests
- `steps.py` — helper functions for checkpoint/restore/migrate sequences

---

## 7. PyTorch Provisioning

The provisioning system installs ROCm-enabled `torch`, `torchvision`, and `torchaudio` wheels into a managed virtual environment **on the execution node**. The coordinator process never imports PyTorch.

For the complete CLI option reference, installation flow, and rocm-test.toml configuration, see [`docs/pytorch-provisioning.md`](../docs/pytorch-provisioning.md).

### Two Entry Paths

| Path | Trigger | Scope |
|---|---|---|
| **Path A — lazy / per-test** | Test requests `pytorch_env` fixture (no `--pre-install`) | Function-scoped; retried on transient failure |
| **Path B — explicit / session** | `--pre-install pytorch=...` CLI flag | `pytest_sessionstart`; runs once per node before collection |

Both write to a shared sanity cache. After the first successful provision, all subsequent fixture calls return from cache with no subprocess.

### Installation Modes

| Mode | Wheel channel | Fallback |
|---|---|---|
| `auto` | `multiarch → family` (in order) | Yes — intermediate failures are warnings |
| `multiarch` | `multiarch_index` (single) | Single candidate; no fallback |
| `family` | `family_index_base/<gfx_family>/` | Single candidate; no fallback |
| `staging` | `staging_index` (single) | Single candidate; never auto-selected |

`mode=auto` is the right default for nightly CI. Use `mode=multiarch` or `mode=family` for deterministic single-candidate installs.

### GFX Family Mapping

| GPU | Architecture | Family index suffix |
|---|---|---|
| MI300X / MI325X | gfx942 | `gfx94X-dcgpu` |
| MI350X | gfx950 | `gfx950-dcgpu` |
| MI250X / MI210 | gfx90a | `gfx90a-dcgpu` |
| MI100 | gfx908 | `gfx908-dcgpu` |
| RDNA3 (RX 7900 series) | gfx1100–gfx1103 | `gfx110X-all` |
| RDNA4 (RX 9070 series) | gfx1200–gfx1201 | `gfx120X-all` |

### Three-Phase Fixture Resolution

Every `pytorch_env` call walks three phases in order:

```
Phase 1 — sanity cache hit (O(1), no subprocess)
    config._framework_sanity_ok["pytorch:{node_label}"] exists?
    └── yes → return cached result (all tests after first success land here)

Phase 2 — pre-install result (Path B)
    config._framework_provision_results[node_label] exists?
    └── result.ok  → promote to sanity cache → return
    └── result.fail → pytest.fail (pre-install is definitive; never retried)

Phase 3 — lazy install (Path A; retried per test until first success)
    → provision_framework(runner=executor, spec=spec, ...)
    → result.ok  → write sanity cache → return
    → result.fail → pytest.fail (do NOT write cache; next test retries Phase 3)
```

### Writing Tests that Use PyTorch

```python
import pytest

@pytest.mark.ci.nightly
@pytest.mark.hw.gpu
@pytest.mark.layer.ml_framework
@pytest.mark.runtime.medium
def test_pytorch_tensor_on_gpu(require_torch, torch_python, target_executor, ld_path):
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld_path['LD_LIBRARY_PATH']} "
        f"{torch_python} -c \""
        "import torch; t = torch.tensor([1.0]).cuda(); "
        "assert t.item() == 1.0; print('OK')\""
    )
    assert result.ok, result.stderr
    assert "OK" in result.stdout
```

**Golden rules for PyTorch tests:**
- **Never import `torch` on the coordinator.** Run all PyTorch code via `target_executor.run(f"{torch_python} ...")`.
- **Use `require_torch`, not `pytorch_env` directly**, unless you need the provision result object.
- **`require_torch` calls `pytest.fail`, not `pytest.skip`.** An unusable environment is a hard failure — provisioning was configured explicitly.
- **`require_torch_tunableop` calls `pytest.skip`.** TunableOp absence is a build variant, not an error.
- **Pass `ld_path` whenever running against a TheRock build** — it provides `LD_LIBRARY_PATH=<rock_dir>/lib:...`.
- **Never hardcode wheel index URLs** in test code — use `rocm-test.toml [frameworks]` or `--pre-install`.

### Quick CLI Examples

```bash
# Auto mode (default for nightly CI)
pytest tests/e2e/ --pre-install "pytorch=mode=auto"

# Explicit multiarch for gfx942
pytest tests/e2e/ --pre-install "pytorch=mode=multiarch,device=gfx942"

# Pinned torch version — no fallback
pytest tests/e2e/ --pre-install \
  "pytorch=mode=multiarch,device=gfx942,torch=2.14.0a0+rocm7.12.0a20260716"

# Family v2 explicit
pytest tests/e2e/ --pre-install "pytorch=mode=family,gfx_family=gfx94X-dcgpu"

# Staging (pre-release validation only)
pytest tests/e2e/ --pre-install "pytorch=mode=staging,device=gfx942"
```

---

## 8. CRIU Checkpoint-Restore Support

`tests/common/criu/` provides shared infrastructure for GPU process checkpoint-restore tests. Tests using it live under `tests/e2e/recovery/criu/`.

### Two Entry Points

**`ensure_criu_runtime(external_build, cmake_executor, framework_config) → str`**

For bare-metal and `--remote-node` suites where the workload runs on the host or SSH node. Uses `external_build.clone_repo` to place the CRIU source on the host, then transfers the installer and builds remotely.

**`ensure_criu_runtime_target(target_executor, framework_config) → str`**

For container tests (`@pytest.mark.container`). `external_build.clone_repo` cannot reach inside a container filesystem, so the installer is transferred into the target via base64, then self-clones and builds inside the container.

Both return the `CRIU` command prefix string:

```
sudo -n env "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" criu
```

### How They Work

1. **Passwordless sudo check** — `sudo -n true` must succeed, or `pytest.skip` is called with a descriptive message.
2. **Ready check** — verifies the amdgpu plugin exists at `/usr/lib/criu/amdgpu_plugin.so` and `criu check` reports "Looks good". Returns immediately if already ready.
3. **Auto-install** — disabled via `ROCM_TEST_CRIU_AUTO_INSTALL=0`; version pinned via `ROCM_TEST_CRIU_VERSION` (default `v4.1`).
4. **Re-verify** — checks ready state again after install; `pytest.fail` on persistent failure.

### Environment Variables

| Variable | Default | Effect |
|---|---|---|
| `ROCM_TEST_CRIU_AUTO_INSTALL` | `"1"` | Set to `"0"` to skip auto-install and skip the test instead |
| `ROCM_TEST_CRIU_VERSION` | `"v4.1"` | Git tag/branch of CRIU to build |
| `CRIU_REPO` | `"https://github.com/checkpoint-restore/criu.git"` | Override the CRIU repository URL |

### Usage Pattern

```python
# tests/e2e/recovery/criu/conftest.py
import pytest
from tests.common.criu.fixtures import ensure_criu_runtime

@pytest.fixture(scope="session")
def criu(external_build, cmake_executor, framework_config):
    return ensure_criu_runtime(external_build, cmake_executor, framework_config)
```

```python
# tests/e2e/recovery/criu/test_gpu_checkpoint.py
import pytest

@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
def test_checkpoint_restore(criu, target_executor, allure_reporter):
    with allure_reporter.step("Launch GPU workload"):
        bg = target_executor.start_background("./my_gpu_workload", log_path="output/...")
    with allure_reporter.step("Checkpoint"):
        result = target_executor.run(f"{criu} dump -D /tmp/ckpt -t {bg.pid}")
        assert result.ok, result.stderr
    with allure_reporter.step("Restore"):
        result = target_executor.run(f"{criu} restore -D /tmp/ckpt")
        assert result.ok, result.stderr
```

### Manual Pre-Install (CI nodes without network access)

```bash
# On the test node before the suite runs:
python3 tests/common/criu/installer.py v4.1

# Disable auto-install in pytest (node already has CRIU):
export ROCM_TEST_CRIU_AUTO_INSTALL=0
pytest tests/e2e/recovery/criu/ -m "hw.gpu and ci.nightly" ...
```

---

## 9. Debugging Standards

### Log Levels

Set `log_level` in `rocm-test.toml` or `ROCM_TEST_FRAMEWORK_LOG_LEVEL`:

| Level | Effect |
|---|---|
| `normal` (default) | Framework info, test RUNNING/PASS/FAIL banners, plugin summaries |
| `debug` | `stream_stderr` on executors — live subprocess stderr to console |
| `verbose` | Both `stream_stdout` and `stream_stderr` — full live subprocess output |

### Per-Test Executor Logs

```python
result = target_executor.run(
    "./my_kernel --benchmark",
    log_path="output/artifacts/executor-logs/test_foo__kernel.log",
)
```

For background daemons:
```python
with cpu_executor.start_background(
    "rocm-smi --showmetrics --interval=2",
    log_path="output/artifacts/executor-logs/test_foo__monitor.log",
) as monitor:
    result = target_executor.run("./my_kernel")
    assert monitor.is_alive
# monitor.stop_result → ExecutionResult with full daemon output
```

### Artifacts on Failure

`artifacts_plugin` auto-attaches to Allure on every failed test attempt:
- GPU state dump (`amd-smi metric` output)
- Full stdout + stderr of the failed command

Check the Allure step for the failed attempt first — the artifact is attached there.

### DryRun Mode for Fixture Debugging

```bash
pytest tests/e2e/stack_validation/test_hip_runtime.py --no-gpu -v --tb=long
```

`DryRunExecutor` always returns `exit_code=0, stdout="DRY_RUN=1\nRESULT_OK"`. Use this to debug fixture wiring, marker resolution, and plugin interactions before touching real hardware.

### Health Check Failures

`HEALTH_FAIL` is distinct from `FAIL`. It indicates a hardware-side issue. Check:
1. `output/artifacts/gpu-info-<node>.log` — GPU state snapshot at session start
2. The Allure step for the health check — includes `amd-smi` JSON snapshot
3. `[gpu]` thresholds in `rocm-test.toml` — may need tuning for high-load environments

---

## 10. Contribution Quality Gates

All gates are enforced by the PostToolUse hook and CI pipeline. A PR will not be merged if any gate fails.

### Required Marker Compliance

Every `test_*` function must carry at least one marker from each required dimension:
- `hw.*` — `gpu` / `multi_gpu` / `cpu_only`
- `ci.*` — `pr` / `nightly` / `weekly`
- `layer.*` — `runtime` / `math_lib` / `ml_framework`

`runtime.*` is not linter-enforced but must always be declared — omitting it disables smart-sharding runtime weights and misleads `DynamicScheduler`.

**Authoritative source:** `framework/markers/taxonomy.py → MARKER_SCHEMA`. Add new values there first; never add them only in test files.

### Assertion Quality

GPU tests must assert beyond `exit_code == 0`. Use `parse_metric()` from `framework.common.helpers` to extract a numeric value and assert a threshold. `hw.cpu_only` DryRun tests are exempt.

### Structural Conventions

- Copyright + SPDX header: `# Copyright Advanced Micro Devices, Inc. / # SPDX-License-Identifier: MIT`
- Module docstring with numbered `Validates:` list — required on every test file
- Inline scripts as triple-quoted module-level constants (`_SCRIPT_NAME = '''...'''`), not inside functions
- `allure_reporter.step(...)` wrapping every `target_executor.run()` call
- Google-style docstrings on all public functions and modules in `framework/`
- Type hints on all function signatures in `framework/` (enforced by `mypy`)

### CI Gate Compatibility

- `ci.pr` + `runtime.medium` is a **hard conflict** — medium tests must be `ci.nightly` or higher
- `ci.pr` tests that download models or require network access are **rejected**
- Soak tests must be `ci.weekly`

### Forbidden Patterns

```
subprocess.run() / subprocess.Popen()      →  target_executor.run(cmd)
time.sleep(N)                              →  remove; health checks handle readiness
HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES →  injected by executor; never set in tests
from framework.plugins / import framework.plugins  →  use fixture injection only
nodes_fixture                              →  does not exist; use target_executor
import torch  (at coordinator module level) →  run via target_executor + torch_python
sys.exit(N) in test body                   →  pytest.skip / pytest.fail
hardcoded /dev/renderD128                  →  os_adapter.list_gpu_device_paths()[0]
```

### Adding a New Plugin

1. Create `framework/plugins/my_plugin.py` following the existing pattern.
2. Add to `conftest.py → pytest_plugins` (remember: `markers_plugin` must be first; `session_plugin` must be second).
3. Document the plugin in `CLAUDE.md → Architecture → Layer Stack`.
4. Add a `hw.cpu_only` / `ci.pr` DryRun test under `tests/dry_run/` to validate it in CI without GPU hardware.

### Adding a New Marker Value

1. Add to `framework/markers/taxonomy.py → MARKER_SCHEMA`.
2. `MarkerLinter` will immediately accept it in test files.
3. Update `CATEGORY_PROFILES` in the same file if tests live under a new well-known directory.
