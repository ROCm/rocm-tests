# Framework Skills & Architecture Manual

A deep-dive reference for Claude Code operating inside `rocm-tests`. This document covers the framework architecture layers, every agentic skill and its internal process, the automated quality gate pipeline, debugging workflows, and contribution standards. Refer to this file before writing tests, agents, or framework code.

---

## 1. System Overview

`rocm-tests` is a **pytest-based system e2e test framework**. It validates the full ROCm software stack — kernel driver → HIP runtime → compute libraries → ML frameworks — on real GPU hardware (nightly/weekly). New tests added shall auto subscribe to pre-defined test markers to run in CI (nightly/weekly)

### Architecture Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│  pytest invocation (CLI / CI / Claude Code skills)                  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  conftest.py — Plugin Registration (markers_plugin FIRST)   │   │
│  │                                                             │   │
│  │  Plugin Stack (ordered — load sequence matters):            │   │
│  │    markers_plugin   → category-profile marker injection     │   │
│  │    gpu_plugin       → --no-gpu / --gpu-arch / --mock-gpu    │   │
│  │    remote_node_plugin → NodePool fleet, target_executor     │   │
│  │    scheduling_plugin → DynamicScheduler, xdist ordering     │   │
│  │    executor_plugin  → container_executor, cpu_executor      │   │
│  │    os_plugin        → os_adapter, platform_name, skip hook  │   │
│  │    health_plugin    → pre/post GPU health gates             │   │
│  │    artifacts_plugin → Allure attachment on failure          │   │
│  │    prereqs_plugin   → session-level ROCm version checks     │   │
│  │    retry_plugin     → --retry-count, retry_fixture          │   │
│  │    reports_plugin   → Allure labels, terminal summary       │   │
│  │    builder_plugin   → compile_binary, ld_path fixtures      │   │
│  │    install_plugin   → --pre-install rocm/pkg on fleet nodes │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  framework/ — Core Engine                                   │   │
│  │    config/      ← rocm-test.toml → env → CLI cascade       │   │
│  │    common/      ← ExecutionResult, Outcome, parse_metric(),  │   │
│  │                   executor_log_path(), gpu_monitor_log_path() │  │
│  │    executors/   ← AbstractExecutor + 8 concrete backends    │   │
│  │    nodes/       ← NodePool fleet: NodeSlot, GpuFileLock     │   │
│  │    scheduling/  ← DynamicScheduler, SchedulePolicy          │   │
│  │    builder/     ← BinaryBuilder (hipcc, xdist-safe)         │   │
│  │    gpu/         ← GpuDetector, GpuAllocator, BackgroundMon  │   │
│  │    markers/     ← MARKER_SCHEMA taxonomy + MarkerLinter     │   │
│  │    reporting/   ← AllureReporter, step(), report_metric()   │   │
│  │    os_adapter/  ← Linux + Windows GPU enumeration           │   │
│  │    rocm/libs/   ← hip.py, rccl.py, amd_smi.py, stack.py    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  tests/ — Test Suite                                        │   │
│  │    dry_run/               ← ci.pr, hw.cpu_only, no GPU      │   │
│  │    e2e/compiler/          ← layer.runtime, hipcc            │   │
│  │    e2e/hwq_heuristic/     ← layer.runtime, HW queue tests   │   │
│  │    e2e/hip_runtime/       ← layer.runtime, HIP driver API   │   │
│  │    e2e/hipblaslt/         ← layer.math_lib, GEMM heuristics │   │
│  │    e2e/rocm_libs/         ← layer.math_lib, rocsolver/blas  │   │
│  │    e2e/rocprim/           ← layer.math_lib, rocPRIM + HMM   │   │
│  │    common/                ← shared helpers (NOT tests)      │   │
│  │      _cmake_build.py      ← cmake_build() + find_rocm_clangpp() │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Executor Hierarchy

All GPU tests receive a `NodeExecutorGroup` from `target_executor`. Test code never branches on executor type.

| Executor | Role | When |
|---|---|---|
| `DryRunExecutor` | Synthetic stub; always `exit_code=0` | `--no-gpu` / `hw.cpu_only` |
| `CpuExecutor` | Real subprocess, no GPU env | `hw.cpu_only` needing real commands |
| `LocalExecutor` | Subprocess + `ROCR_VISIBLE_DEVICES` | Local `hw.gpu` / `hw.multi_gpu` |
| `ContainerExecutor` | Docker/Podman + AMD device passthrough | `--container-mode` |
| `SshExecutor` | SSH + `ROCR_VISIBLE_DEVICES` injection; handles remote GPU tests directly | Remote `hw.gpu` / `hw.multi_gpu` |
| `NodeExecutorGroup` | Uniform container returned by all GPU fixtures | Always — wraps 1 or N executors |
| `BackgroundProcess` | Thread-safe daemon; context manager; `.is_alive`, `.stop()` → `ExecutionResult` | Returned by `executor.start_background(cmd, log_path=...)` |
| `NoOpBackgroundProcess` | Stub for `DryRunExecutor.start_background()`; same API, never alive | `--no-gpu` / `hw.cpu_only` background calls |

### Config Cascade

Priority order (lowest → highest):

```
Code defaults → rocm-test.toml → ROCM_TEST_* env vars → pytest CLI flags
```

Section-to-dataclass mapping: `[framework]` → `FrameworkSection`, `[gpu]` → `GpuSection`, `[therock]` → `TheRockSection`, `[reporting]` → `ReportingSection`.

---

## 2. Workflow & Lifecycle

### Bootstrap

```bash
# Clone and set up the dev environment
git clone <repo-url> && cd rocm-tests
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime + lint/type/docs/security tools

# Start Claude Code (all skills auto-load from .claude/agents/)
claude
```

No `/init` is required. `CLAUDE.md`, `.claude/agents/*.md`, and `.claude/settings.json` are loaded automatically by Claude Code at session start.

### Execution Pipeline (per-test)

Every test — regardless of CI tier or executor type — passes through this pipeline:

```
Test Selected
    │
    ▼
Config Load                   (once per session, not per test)
    │
    ▼
Prereq Check                  (ROCm version, driver loaded, GPU count)
    │── Required FAIL ──────→  SESSION ABORT (all tests skipped)
    │
    ▼
GPU Acquire                   (GpuAllocator: semaphore pool; blocks if all GPUs busy)
    │
    ▼
GPU Health Pre-check          (temp, ECC errors, VRAM free, clock state)
    │── HEALTH_FAIL ─────────→  Outcome = HEALTH_FAIL (hardware issue, not test logic failure)
    │
    ▼
Execute Command               (target_executor.run(cmd) or start_background())
    │── Timeout ─────────────→  Outcome = TIMEOUT
    │
    ▼
Artifact Capture              (on each failed attempt: GPU dump, lsmod, stdout/stderr)
    │
    ▼
GPU Health Post-check
    │── HEALTH_FAIL ─────────→  Outcome = HEALTH_FAIL
    │
    ▼
Outcome Classification        (PASS / FAIL / TIMEOUT / KILLED / ERROR / HEALTH_FAIL / PERF_DROP / REGRESSION)
    │
    ├── PASS + perf test ────→  Baseline Compare → PERF_DROP if out-of-band
    │
    ├── FAIL + retries left ─→  Re-run with artifact capture; tag FLAKY if later pass
    │
    ▼
GPU Release                   (returned to allocator pool)
    │
    ▼
Allure JSON written            (for every outcome — HEALTH_FAIL and REGRESSION appear in dashboard)
    │
    ▼
Session Log updated            (output/artifacts/session.log — append-safe across xdist workers)
```

### Observation

| Output | Location | Notes |
|---|---|---|
| Session log | `output/artifacts/session.log` | All test RUNNING/PASS/FAIL banners, xdist-safe append |
| Allure results | `output/artifacts/allure-results/` | Per-test JSON; generate HTML with `allure generate` |
| Allure HTML report | `build/allure-report/` | Multi-run dashboard via `--allure-db N` |
| Lightweight HTML | `--html=build/report.html --self-contained-html` | pytest-html; no Allure CLI needed (`pytest-html<4`) |
| Executor logs | `output/artifacts/executor-logs/` | Per-test log files from `start_background()` |
| Compiled binaries | `output/test-binaries/<subdir>/` | HIP/C++ kernels compiled by `BinaryBuilder` |
| GPU info | `output/artifacts/gpu-info-<node>.log` | `amd-smi list` diagnostic snapshot at session start |
| Runtime data | `--collect-runtimes PATH` | Per-test wall-clock duration + outcome as JSON |

**Live GPU telemetry** is captured via `GpuBackgroundMonitor` during test execution and attached to Allure steps as JSON. Sampling interval is controlled by `[gpu] monitor_interval_secs` in `rocm-test.toml`.

**Session-wide `[RUNNING]` banner format** (from `reports_plugin`):
```
[RUNNING gw0] tests/e2e/compiler/test_hip_compile.py::test_hipcc_basic  (14:32:01)
```

---

## 3. Agent Skills

Claude Code ships three built-in agentic skills accessible via slash commands. Each is backed by a sub-agent definition in `.claude/agents/`. Agents read the framework source before producing output — they never invent marker values or fixture names.

---

### `/creator` — Generate a complete test file

**What it does:** Produces a full, marker-compliant pytest test file from a natural-language GPU feature description or requirements document. Each independently testable assertion becomes its own test function.

**Internal process (6 steps):**

1. **Gather requirement** — reads the description; asks if not provided.
2. **Resolve markers** — applies the decision table below to every dimension.
3. **Declare resources** — adds `@pytest.mark.gpu_vram(N)` / `@pytest.mark.gpu_count(N)` / `@pytest.mark.container_image(...)` where needed.
4. **Select fixtures** — `target_executor` for GPU tests; `dry_run_executor` for `hw.cpu_only`.
5. **Write file** — copyright + module docstring (`Validates:` list) + module-level script constants + `allure_reporter.step()`-wrapped executor calls + `@pytest.mark.parametrize` where multi-value.
6. **Validate + next-steps checklist** — collect-only → DryRun → GPU run.

**Marker decision table:**

| Dimension | Decision Rule |
|---|---|
| `layer.*` | `runtime`: HIP API; `math_lib`: rocBLAS/RCCL/rocFFT |
| `ci.*` | `pr`: fast + DryRun-safe (< 5 min, no GPU download); `nightly`: typical E2E; `weekly`: soak |
| `hw.*` | `gpu`: one GPU; `multi_gpu`: two or more; `cpu_only`: DryRun / framework tests |
| `runtime.*` | `fast` <5 min; `medium` <30 min; `soak` hours |
| `os.*` | `linux` for all current E2E tests |
| `e2e.*` | `stack`: full-stack; `multinode`: multi-node collectives |

**Rules enforced:**
- Never `subprocess.run()` / `subprocess.Popen()` — always `target_executor.run(cmd)`
- Never set `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` — executor injects automatically
- Never `time.sleep()` — health checks handle GPU readiness
- Never `nodes_fixture` — use `target_executor` for all GPU tiers
- Never `from framework.plugins import ...` — use fixture injection only
- Always module docstring with numbered `Validates:` list
- Always `allure_reporter.step()` around every `target_executor.run()` call
- Strong assertion: numeric threshold checks on stdout content, not just `exit_code == 0`

**Example:**
```
/creator
> Validate that rocBLAS SGEMM completes without error on gfx1100

→ layer: math_lib, ci: nightly, hw: gpu, runtime: medium, os: linux
→ Creates: tests/e2e/rocm_libs/test_rocblas_sgemm.py
```

---

### `/refiner [review-as <persona>] <file>` — Review and extend an existing test

**What it does:** Operates in two modes detected from the user's phrasing:

- **Mode A — Review** (default when user says "review", "refine", "check"): Applies the 4-persona checklist, runs marker lint, detects flakiness, and reports top-3 improvements with before/after code.
- **Mode B — Extend** (when user says "add", "extend", or describes a new variant): Adds new test functions or parametrize — never removes or renames existing functions.

**Internal process — Review:**

1. Reads the target file, `framework/markers/taxonomy.py`, `framework/markers/linter.py`, `framework/plugins/executor_plugin.py`, and `framework/plugins/artifacts_plugin.py`.
2. Runs marker lint — surfaces violations per dimension per function.
3. Applies all four persona checklists (or a specific persona if requested).
4. Ranks top-3 improvements with concrete before/after code.

**Review Criteria:**

| Category | Key Anti-patterns Flagged |
|---|---|
| Efficiency | Function-scoped fixture that should be session-scoped; `allure_reporter` in DryRun test; duplicate marker decorators; `parse_metric` imported inside function body |
| Stability (ERROR) | `time.sleep(N)`, `os.environ["ROCR_VISIBLE_DEVICES"]`, `subprocess.run()`, `nodes_fixture`, `from framework.plugins import` |
| Stability (WARNING) | Hardcoded GPU index integer, assertion only on `result.exit_code` with no stdout check, ML test with no NaN/Inf guard |
| Coverage | No `pytest.skip` for optional prereq, no OOM/VRAM-exhaustion test, no invalid-input negative test, no `hw.cpu_only` DryRun companion for `ci.pr` coverage |

**Four personas:**

#### `developer`
GPU API correctness, assertion strength, HIP invocation patterns.
- Wrong precision (`f32_r` vs `f64_r`; `torch.float32` vs `torch.float64`)
- `target_executor.run(cmd)` must be wrapped in `allure_reporter.step()` for traceability
- Assertion quality: `exit_code == 0` alone is WEAK — `parse_metric()` + threshold is STRONG
- Edge cases: VRAM near limit, multi-GPU rank interactions, thermal throttle behavior

#### `tester`
Coverage uniqueness, missing failure modes, parametrize opportunities.
- What if the required library is not installed? → `pytest.skip`, not crash
- What if VRAM is insufficient? → clear error message, not hang
- Assertion quality scale: `exit_code==0` (WEAK) → sentinel string (MEDIUM) → `parse_metric()` + threshold (STRONG) → NaN/Inf guard (STRONGEST)
- Parametrize over: GPU arch, input sizes, data types (f16/f32/f64/bf16), batch sizes

#### `automation`
Marker accuracy, runtime weight vs wall time, CI gate placement.
- `ci.pr` + `runtime.medium` = CONFLICT — medium tests must be `ci.nightly` or higher
- `hw.multi_gpu` without `e2e.multinode` → missing Allure grouping for collective tests
- Wrong `runtime.*` weight misleads `DynamicScheduler` → longer nightly wall time
- Tests downloading models or requiring network access must NOT be `ci.pr`

#### `devops`
VRAM requirements, prerequisite declarations, health gate impact, artifact volume.
- gfx1100 (RX 7900 XTX): 24 GB VRAM; gfx942 (MI300X): 192 GB VRAM — safe on MI300X may OOM on gfx1100
- Missing `@pytest.mark.gpu_vram(N)` when workload needs a minimum VRAM threshold
- Missing `pytest.skip` for optional prereqs (PyTorch, specific ROCm version) = silent failure
- Artifact volume: soak tests logging per-second stdout can generate GB — use `runtime.soak` + `ci.weekly` markers to gate them out of nightly

**Internal process — Extend:**

1. Detects extension type from the description:

   | Request | Pattern Applied |
   |---|---|
   | "run on 2 GPUs" / "multi-GPU variant" | New function: `hw.multi_gpu` + `e2e.multinode`; still `target_executor` |
   | "test more sizes" / "parametrize" | `@pytest.mark.parametrize(...)` on new function |
   | "what if it fails" / "negative test" | New function: `hw.cpu_only` + `dry_run_executor`, assert non-zero exit |
   | "run longer" / "soak variant" | New function: `ci.weekly` + `runtime.soak` + explicit `timeout` arg |
   | "weekly regression" | New function: `ci.weekly` + `runtime.soak` |

2. Shows clear diff: added lines, unchanged functions explicitly noted.
3. Validates with `--collect-only` — all original + new functions must appear.

**Usage:**
```bash
/refiner tests/e2e/stack_validation/test_hip_runtime.py     # full 4-persona review
/refiner review-as developer tests/e2e/compiler/test_hip_compile.py  # single persona
/refiner tests/e2e/hip_runtime/test_multi_stream.py add an event-dependency variant
```

**Output format (Review mode):**
```markdown
## Refine: tests/e2e/<domain>/test_<name>.py

### Marker Lint
✅ All required dimensions present — hw.gpu, ci.nightly, layer.math_lib
OR
❌ VIOLATION: test_foo(): Missing required marker dimension: ci

### Developer   [finding or ✓]
### Tester      [finding or ✓]
### Automation  [finding or ✓]
### DevOps      [finding or ✓]

## Top 3 Improvements
### 1. [Title] — Why: ... / Before (line N) + After code
### 2. ...
### 3. ...
```

---

### `/porter <source-file>` — Port an external test into rocm-tests

**What it does:** Takes an external test — shell script, raw Python, non-compliant pytest, C++ gtest, or AMD framework test — and rewrites it as a fully framework-compliant rocm-tests pytest file.

**Supported source types:**

| Source Type | Common Patterns |
|---|---|
| Shell scripts (`.sh`) | `rocm-smi` / `hipcc` invocations; `exit 1` on failure; no assertions |
| Raw Python scripts | `subprocess.run()`, `os.environ["ROCR_VISIBLE_DEVICES"]`, `sys.exit()` for errors |
| Non-compliant pytest | Missing `hw.*`/`ci.*`/`layer.*` markers; `subprocess.run()` in test body |
| C++ gtest programs | `EXPECT_EQ`, `ASSERT_GT` — translated to Python assertions |
| Other AMD frameworks | `hip_test_base.py`, `rocBLAS-bench` runners, `rccl-tests` launchers |

**Internal process (5 steps):**

1. **Identify Logic** — reads source; records what each operation does, what it asserts, what it guards.
2. **Map Capabilities** — applies the transformation table to every external pattern.
3. **Resolve Markers** — determines `hw/ci/layer/runtime/os` for each extracted test case.
4. **Re-structure** — rewrites into rocm-tests pattern: copyright + module docstring + module-level scripts + `allure_reporter.step()` + `parse_metric()`. One test function per independently testable assertion.
5. **Validate** — `--collect-only` to confirm pytest discovers the ported test; shows transformation summary table.

**Transformation table (always shown in output):**

| External Pattern | rocm-tests Replacement | Reason |
|---|---|---|
| `subprocess.run(cmd)` | `target_executor.run(cmd)` | Executor handles env, logging, timeout |
| `subprocess.Popen(cmd)` | `target_executor.run(cmd)` | Same — executor handles Popen internally |
| `os.environ["ROCR_VISIBLE_DEVICES"] = "0"` | Removed | Injected automatically by executor |
| `os.environ["HIP_VISIBLE_DEVICES"] = "0"` | Removed | Same — never set device env in tests |
| `if not shutil.which("rocm-smi"): sys.exit(1)` | `pytest.skip("rocm-smi not available")` | Graceful skip vs session abort |
| `try: import torch \nexcept ImportError: sys.exit(1)` | `pytest.skip("PyTorch not installed")` | Never sys.exit — use pytest.skip |
| `time.sleep(N)` | Removed | Health checks handle GPU readiness |
| `assert proc.returncode == 0` | `assert result.ok` + `parse_metric()` | Stronger assertion; metric in Allure |
| Hardcoded `/dev/renderD128` | `os_adapter.list_gpu_device_paths()[0]` | Never hardcode device paths |
| `logging.info("step X")` | `allure_reporter.step("step X")` | Structured observability in Allure |
| C++ `EXPECT_EQ(a, b)` | `assert a == b, f"Expected {b}, got {a}"` | Direct translation |
| C++ `ASSERT_GT(value, threshold)` | `assert value > threshold, f"Got {value}, expected > {threshold}"` | Direct translation |
| Shell `${VAR:-default}` | `framework_config.section.field or "default"` | Config cascade replaces shell defaults |

**Example:**
```
/porter scripts/check_hip_devices.sh

→ Identify: checks HIP returns >= 1 device; asserts exit code 0 and numeric count
→ Map: ROCR_VISIBLE_DEVICES → Removed; bash count check → parse_metric("DEVICE_COUNT")
→ Markers: layer.runtime, ci.pr, hw.gpu, runtime.fast, os.linux
→ Creates: tests/e2e/stack_validation/test_hip_device_count.py

Transformation Summary:
| export ROCR_VISIBLE_DEVICES=0     | Removed — executor injects              |
| if [ $? -ne 0 ]; then exit 1; fi  | assert result.exit_code == 0, ...       |
| echo "$count" | grep -oP '\d+'    | parse_metric(result.stdout, "DEVICE_COUNT") |

Validation:
  pytest tests/e2e/stack_validation/test_hip_device_count.py --collect-only -q --no-gpu
```

---

## 4. Dynamic Scheduling 

Dynamic scheduling in test framework distributes GPU tests efficiently.

---

### 4.1 Mechanism Overview

| Mechanism | Module | Scope | Algorithm | Trigger |
|---|---|---|---|---|
| `DynamicScheduler` | `framework/scheduling/dynamic_scheduler.py` | Multi-node + single-node with xdist | Resource-demand sort + xdist_group assignment | `pytest_collection_modifyitems` |

---

### 4.2 DynamicScheduler — Full Workflow

Activated by `scheduling_plugin.pytest_collection_modifyitems`. Runs once per session after collection. **No-op when `--no-gpu` is active** (no `_node_pool` on config).

#### Step 1: xdist_group Assignment

Each test is classified and assigned a group name (or none) for xdist routing:

| Test type | Detection | Assigned `xdist_group` |
|---|---|---|
| Multinode | `@pytest.mark.e2e.multinode` | `"multinode_0"`, `"multinode_1"`, … (unique per test) |
| Multi-GPU | `@pytest.mark.hw.multi_gpu` or `@pytest.mark.gpu_count(N>1)` | `"multi_gpu_{count}_{idx}"` (unique per test) |
| Single-GPU | `@pytest.mark.hw.gpu` only | None — worksteal across free workers |

**Why unique groups matter:** Each multi-GPU and multinode test gets its own group name so separate xdist workers can run *different* multi-GPU tests in parallel — each worker holds its own GPU file locks simultaneously. Without unique groups, tests would serialize on a single worker.

`gpu_count` resolution: reads `@pytest.mark.gpu_count(N)` first; falls back to `2` when `hw.multi_gpu` is present without an explicit count.

#### Step 2: Resource-Demand Sort

Tests are sorted in-place using the active `--schedule-policy`:

**`resource-most`** (default) — Sort key (lower runs first):

```
Tier 0: e2e.multinode            → (0, 0)
Tier 1: multi_gpu by count DESC  → (1, -count)   # higher count = earlier
Tier 2: single_gpu               → (2, 0)
```

Multi-GPU workers block inside `target_executor` fixture waiting for GPU slot acquisition. While blocked, other workers continue stealing single-GPU items from the tail of the queue. Single-GPU tests fill available GPU slots *emergently* — the scheduler does not interleave; utilisation emerges from worksteal.

**`resource-least`** — Sort key (lower runs first):

```
Tier 0: single_gpu               → (0, 0)
Tier 1: multi_gpu by count ASC   → (1, count)    # lower count = earlier
Tier 2: e2e.multinode            → (2, 0)
```

Heavy tests start only after lightweight tests drain the queue. Use this when you need first results quickly (e.g., CI smoke runs, quick feedback loops).

#### Step 3: Recommended Workers

```python
scheduler.recommended_workers()  # == pool.total_gpu_slots()
```

Printed to stdout at session start. Pass this value as `-n` for optimal parallelism:

```bash
# Session prints: [rocm-test] Recommended: -n 8
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --remote-node host.yaml -n 8 -v
```

If GPU tests are present but `-n` is not set, the scheduler emits a warning:
```
[rocm-test] WARNING: 24 GPU test(s) will run SEQUENTIALLY on a 8-GPU node.
           Add -n 8 to run in parallel (one test per GPU).
```

#### Step 4: Runtime Collection (Optional)

`--collect-runtimes PATH` accumulates `(nodeid, duration_secs, outcome)` for every completed test call and writes a JSON file at session end:

```json
{
  "session": {
    "start_ts": "2026-05-12T02:00:00Z",
    "policy": "resource-most",
    "total_tests": 42
  },
  "tests": [
    {"nodeid": "tests/e2e/stack_validation/test_hip_runtime.py::test_hip_device_count", "duration_secs": 2.341, "outcome": "PASSED"},
    ...
  ]
}
```

This is **informational only** — the JSON is not fed back into scheduling in the current release. Use it to audit actual runtimes against declared `runtime.*` marker weights and tune them.

---

### 4.3 VRAM Headroom Filtering

Both mechanisms respect `--vram-headroom-gb` (default 2.0 GB). The rule applied per GPU:

```
assignable = (total_vram_gb - headroom_gb) >= test_gpu_vram_requirement
```

`@pytest.mark.gpu_vram(N)` declares the minimum VRAM in GB that a test needs. Tests without this marker have no VRAM requirement and can run on any GPU.

**Architecture VRAM reference:**

| GPU | Architecture | VRAM |
|---|---|---|
| RX 7900 XTX | gfx1100 | 24 GB |
| MI300X | gfx942 | 192 GB |

A test marked `@pytest.mark.gpu_vram(20)` is safe on both; a test marked `@pytest.mark.gpu_vram(100)` is **MI300X-only**. The scheduler will not assign it to gfx1100 nodes automatically.

**Increase headroom for long-running tests** that accumulate VRAM across steps:
```bash
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --vram-headroom-gb 4.0 -n 4 -v
```

---

### 4.4 CLI Combinations Reference

| Scenario | Command Pattern | Active Mechanism |
|---|---|---|
| Local single GPU, DryRun | `pytest tests/ --no-gpu` | Neither (no-op) |
| Local single GPU | `pytest tests/ -m "hw.gpu"` | Neither (1 GPU, no distribution needed) |
| Local multi-GPU, no xdist | `pytest tests/ -m "hw.gpu"` | Smart Sharding |
| Local multi-GPU, xdist | `pytest tests/ -m "hw.gpu" -n 4` | DynamicScheduler |
| Remote fleet, default policy | `pytest tests/ --remote-node host.yaml -n 8` | DynamicScheduler (resource-most) |
| Remote fleet, fast-feedback | `pytest tests/ --remote-node host.yaml -n 8 --schedule-policy resource-least` | DynamicScheduler (resource-least) |
| Remote fleet + VRAM guard | `pytest tests/ --remote-node host.yaml -n 8 --vram-headroom-gb 4.0` | DynamicScheduler + VRAM filter |
| Audit runtimes after nightly | `pytest tests/ --remote-node host.yaml -n 8 --collect-runtimes output/runtimes.json` | DynamicScheduler + telemetry |

---

### 4.5 Marker Interactions

| Marker | Read by | Effect |
|---|---|---|
| `@pytest.mark.hw.multi_gpu` | `DynamicScheduler._multi_gpu_count()` | Default `gpu_count = 2`; assigns `multi_gpu_*` xdist_group |
| `@pytest.mark.gpu_count(N)` | `DynamicScheduler._multi_gpu_count()` | Overrides count; N > 1 triggers multi-GPU group assignment |
| `@pytest.mark.e2e.multinode` | `DynamicScheduler._is_multinode()` | Assigns `multinode_*` xdist_group; Tier 0 in resource-most |
| `@pytest.mark.gpu_vram(N)` | Both mechanisms + `GpuAllocator` | Filters GPU assignment; scheduler + allocator both enforce |

**Always declare `runtime.*` explicitly** — no category profile injects it, and a missing or wrong weight directly degrades scheduling quality. If a declared `runtime.fast` test actually takes 20 minutes, both the scheduler and the CI gate will misbehave.

---

### 4.6 When the Scheduler Is a No-op

`DynamicScheduler` skips entirely when `config._node_pool is None`, which happens when:
- `--no-gpu` is passed (MockGpuDetector, DryRunExecutor)
- `hw.cpu_only` tests run without a GPU fleet

In these cases, pytest's default collection order applies. `SmartShardManager` also skips — no GPU topology is available.

---

### 4.7 Debugging Scheduling Decisions

Enable `log_level = "debug"` in `rocm-test.toml` or set `ROCM_TEST_FRAMEWORK_LOG_LEVEL=debug` to see per-item xdist_group assignment:

```
DEBUG scheduling_plugin [resource-most]: 24 items — 4 multi-resource, 20 single-gpu
DEBUG xdist_group assigned: tests/e2e/rocprim/test_multi_gpu_hmm.py::test_hmm_alloc_2gpu → multi_gpu_2_0
DEBUG xdist_group assigned: tests/e2e/rocprim/test_multi_gpu_hmm.py::test_hmm_migration_2gpu → multi_gpu_2_1
```

Use `--collect-only -q` to preview the sorted test order without running:
```bash
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --remote-node host.yaml --collect-only -q --no-gpu
```

---

## 5. Debugging Standards

### Log Levels

Set `log_level` in `rocm-test.toml` or via `ROCM_TEST_FRAMEWORK_LOG_LEVEL`:

| Level | Effect |
|---|---|
| `normal` (default) | Framework info, test RUNNING/PASS/FAIL banners, plugin summaries |
| `debug` | Enables `stream_stderr` on all executors — live subprocess stderr to console |
| `verbose` | Enables both `stream_stdout` and `stream_stderr` — full live subprocess output |

### Correlation IDs

Every log line carries a `run_id + test_id + phase` tuple. `run_id` is a UUID generated at session start (via `run_ctx` fixture); `test_id` is the pytest node ID. Use these to trace failures through `session.log`, Allure steps, and executor log files for a single test.

### Per-Test Executor Logs

Any executor supports logging subprocess output to a file via `log_path`:
```python
result = target_executor.run(
    "./my_kernel --benchmark",
    log_path="output/artifacts/executor-logs/test_foo__kernel.log",
)
```
For background daemons via `start_background()`:
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
- GPU state dump (`amd-smi --json` output)
- Kernel module list (`lsmod | grep amd`)
- Full stdout + stderr of the failed command (timestamped)

Always check the Allure step for the failed attempt first — the artifact is attached there, not just in the session log.

### DryRun Mode for Logic Debugging

Use `--no-gpu` to run the full plugin stack, fixture chain, and test code without touching real hardware:
```bash
pytest tests/e2e/stack_validation/test_hip_runtime.py --no-gpu -v --tb=long
```
`DryRunExecutor` always returns `exit_code=0, stdout="DRY_RUN=1\nRESULT_OK"`. Use this to debug fixture wiring, marker resolution, and plugin interactions before running on GPU hardware.

### Health Check Failures

`HEALTH_FAIL` outcomes (distinct from `FAIL`) indicate a hardware-side issue. Check:
1. `output/artifacts/gpu-info-<node>.log` — GPU state snapshot at session start
2. The Allure step for the health check that failed — includes `amd-smi --json` snapshot
3. `[gpu]` thresholds in `rocm-test.toml` — may need tuning for high-load test environments

---

## 6. Contribution Quality Gates

All of these are enforced by the PostToolUse hook and CI pipeline. A PR will not be merged if any gate fails.

### Required Marker Compliance
Every `test_*` function must carry at least one marker from each required dimension:
- `hw.*` — `gpu` / `multi_gpu` / `cpu_only`
- `ci.*` — `pr` / `nightly` / `weekly` / `smoke_e2e`
- `layer.*` — `driver` / `runtime` / `math_lib` / `ml_framework` / `debug_stack`

`runtime.*` must always be declared explicitly — no category profile injects it.

### Marker Values Source of Truth
All valid values live in `framework/markers/taxonomy.py → MARKER_SCHEMA`. Adding a new marker value anywhere else will cause `MarkerLinter` to reject the file. Add to `MARKER_SCHEMA` first.

### Strong Assertion Requirement
GPU tests must assert beyond `exit_code == 0`. Use `parse_metric()` from `framework.common.helpers` to extract a numeric value from `result.stdout` and assert a threshold. DryRun (`hw.cpu_only`) tests are exempt.

### Structural Conventions
- Module docstring with numbered `Validates:` list — required on every test file
- Inline scripts as triple-quoted module-level constants (`_SCRIPT_NAME = '''...'''`), not inside functions
- `allure_reporter.step(...)` wrapping every `target_executor.run()` call
- `Copyright Advanced Micro Devices, Inc. / SPDX-License-Identifier: MIT` header on every file

### Code Style
- Line length: 120 characters (`black`, `ruff`, `pylint` all configured)
- Google-style docstrings on all modules and public functions
- Imports: Standard Library → Third-Party → Local (`framework.*`)
- Type hints on all function signatures in `framework/` (enforced by `mypy`)

### CI Gate Compatibility
- `ci.pr` tests must be: `runtime.fast` (< 5 min), DryRun-safe or GPU-safe on PR runners
- `ci.pr` + `runtime.medium` is a **hard conflict** — medium tests must be `ci.nightly` or higher
- Tests that download models, large datasets, or require network access must NOT be `ci.pr`
- Soak tests must be `ci.weekly`

### Forbidden Patterns (checked by hook and CI)
```
subprocess.run() / subprocess.Popen()  →  use target_executor.run()
time.sleep()                           →  remove; health checks handle readiness
HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES → injected by executor automatically
from framework.plugins / import framework.plugins → use fixture injection only
nodes_fixture                          →  does not exist; use target_executor
```

### Adding a New Executor, Plugin, or Marker Value
1. **New executor:** Implement `AbstractExecutor` interface in `framework/executors/my_executor.py` → expose a fixture in an appropriate plugin → register the plugin in `conftest.py → pytest_plugins` → add a DryRun-compatible test under `tests/dry_run/` to validate it in CI.
2. **New plugin:** Create `framework/plugins/my_plugin.py` following the existing pattern → add to `conftest.py` (ordering matters: `markers_plugin` must stay first) → document in `CLAUDE.md`.
3. **New marker value:** Add to `framework/markers/taxonomy.py → MARKER_SCHEMA` → update `docs/markers.md` → `MarkerLinter` will immediately accept it in test files.
