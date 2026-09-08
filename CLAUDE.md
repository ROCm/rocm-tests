# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

`rocm-test` is an AMD ROCm End-to-End Test Framework. It validates the full ROCm software stack — kernel driver, HIP runtime, compute libraries, and ML frameworks — on AMD GPU hardware. Tests run against real hardware (nightly/weekly) or in DryRun mode (PR validation, no GPU required).

---

## Quick Start

```bash
git clone https://github.com/ROCm/rocm-tests.git
cd rocm-tests
python3 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt   # uv is optional; pip install works too
```

Verify the wiring before writing code:

```bash
pytest tests/ --collect-only -q --no-gpu   # must collect without errors
pytest tests/ -m "ci.pr" --no-gpu -v       # run the PR gate
```

---

## Common Commands

| Goal | Command |
|---|---|
| Run a single test (no GPU) | `pytest tests/dry_run/test_config_loader.py -v` |
| Collect only (preview) | `pytest tests/e2e/ -m "hw.gpu and ci.nightly" --collect-only -q --no-gpu` |
| Run nightly on local GPU | `pytest tests/e2e/ -m "hw.gpu and ci.nightly" --gpu-arch gfx942 -v` |
| Run against a remote fleet | `pytest tests/e2e/ -m "hw.gpu and ci.nightly" --remote-node host.yaml -n 4 -v` |
| Pre-install ROCm on fleet | `pytest tests/e2e/ --remote-node host.yaml --pre-install rocm=6.4.0 -n 4 -v` |
| Pre-install OS packages | `pytest tests/e2e/ --remote-node host.yaml --pre-install pkg=curl,wget -v` |
| Pre-install PyTorch (auto) | `pytest tests/e2e/ --remote-node host.yaml --pre-install "pytorch=mode=auto" -v` |
| Lint | `ruff check framework tests && black --check --diff framework tests` |
| Auto-fix formatting | `ruff check --fix framework tests && black framework tests` |
| Type check | `mypy framework --show-error-codes` |
| Security scan | `bandit -r framework -c pyproject.toml` |
| Build docs | `mkdocs build --strict --site-dir build/site` |

**Style:** PEP 8, 120-character line length (`black`, `ruff`, `pylint`). Google-style docstrings on all modules.

**Never commit secrets.** Use `ROCM_TEST_*` environment variables for webhook URLs, API tokens, and credentials.

---

## Architecture

### Layer Stack

```
conftest.py (root)
    ├── MarkDecorator.__getattr__ patch   # enables @pytest.mark.ci.pr dotted syntax
    └── pytest_plugins → framework/plugins/*.py   (registration order matters)
            ├── markers_plugin.py       # FIRST: category-profile marker injection
            ├── session_plugin.py       # SECOND: framework_config, run_ctx, _attach_test_log
            ├── gpu_plugin.py           # --no-gpu/--gpu-arch/--mock-gpu, dry_run_executor
            ├── remote_node_plugin.py   # --remote-node/--gpu-acquire-timeout, NodePool, target_executor
            ├── scheduling_plugin.py    # --schedule-policy/--collect-runtimes/--vram-headroom-gb
            ├── executor_plugin.py      # --container-mode/--container-image, cpu_executor/container_executor
            ├── os_plugin.py            # os_adapter/platform_name fixtures, os.* marker skip hook
            ├── health_plugin.py        # Pre/post GPU health gates (temp, ECC, VRAM, clocks)
            ├── artifacts_plugin.py     # allure_reporter, artifacts_fixture; GPU state dump on failure
            ├── prereqs_plugin.py       # Session-level driver/ROCm version checks
            ├── retry_plugin.py         # --retry-count, retry_fixture, @pytest.mark.retry
            ├── reports_plugin.py       # Allure label mapping, terminal summary
            ├── builder_plugin.py       # --rock-dir/--compiler-build-dir, compile_binary, ld_path
            └── install_plugin.py       # --pre-install rocm=X/pkg=X/pytorch=..., parallel fleet install

framework/
    config/       # rocm-test.toml → env vars → CLI flags cascade (FrameworkConfig dataclasses)
    common/       # ExecutionResult, executor_log_path(), gpu_monitor_log_path(), workspace_layout
    executors/    # AbstractExecutor + backends: DryRun, Cpu, Local, Ssh, Container, NodeExecutorGroup
    nodes/        # NodePool fleet manager: NodeSpec, NodeSlot, GpuFileLock, PendingTracker
    scheduling/   # DynamicScheduler, SchedulePolicy — resource-aware xdist scheduling
    builder/      # BinaryBuilder — hipcc compilation with xdist locking + incremental builds
    gpu/          # GpuDetector, MockGpuDetector, GpuAllocator, GpuDrainChecker, GpuBackgroundMonitor
    markers/      # MARKER_SCHEMA taxonomy, MarkerLinter
    reporting/    # AllureReporter, step(), attach_text(), report_metric()
    os_adapter/   # Linux + Windows GPU enumeration behind one interface
    rocm/libs/    # ROCm library helpers: hip.py, rccl.py, amd_smi.py, stack.py

tests/
    common/                  # Shared utilities — NOT test files (excluded via norecursedirs)
        factories.py         # fake_gpu_info(), fake_execution_result() — unit test helpers
        spirv.py             # assert_spirv_offload_bundle() — SPIR-V validation helper
        ml_provisioning/     # PyTorch provisioning engine (see PyTorch Workflow below)
        criu/                # CRIU checkpoint-restore build + runtime helpers (see CRIU below)
    dry_run/                 # PR-gate tests: config, DryRun, scheduling (no GPU required, ci.pr)
    e2e/
        compiler/            # hipcc and SPIR-V compilation tests
        hip_runtime/         # HIP driver API, multi-stream, IPC, multiprocess tests
        hip_directed/        # Catch2-based HIP directed tests
        hipblaslt/           # hipBLASLt GEMM heuristic and shape-boundary tests
        hwq_heuristic/       # GPU hardware queue heuristic tests
        rccl/                # RCCL collective communication tests
        rocm_examples/       # ROCm official examples suite
        rocm_libs/           # rocsolver, rocblas, montecarlo_weather tests
        rocprim/             # rocPRIM primitives and multi-GPU HMM tests
        hpc/quda/            # QUDA lattice QCD tests
        recovery/criu/       # GPU process checkpoint-restore tests (uses tests/common/criu/)
```

### Plugin Load Order — Ordering Rules

Two hard constraints govern `pytest_plugins` order:

1. **`markers_plugin` must be first.** It injects `hw.*`, `ci.*`, and `layer.*` markers from `CATEGORY_PROFILES` during `pytest_collection_modifyitems`. `scheduling_plugin` and `gpu_plugin` both read those markers and must see them fully applied.

2. **`session_plugin` must be second.** It provides `framework_config` and `run_ctx` — foundational session fixtures consumed by `remote_node_plugin`, `executor_plugin`, `health_plugin`, `builder_plugin`, and `install_plugin`. Plugin fixtures are globally visible regardless of directory layout; conftest-defined fixtures are scoped to tests below that conftest's directory.

---

## Marker System

Every test function must carry **at least one marker from each required dimension**.

| Dimension | Required | Values |
|---|---|---|
| `hw.*` | YES | `gpu`, `multi_gpu`, `cpu_only` |
| `ci.*` | YES | `pr`, `nightly`, `weekly` |
| `layer.*` | YES | `runtime`, `math_lib` |
| `runtime.*` | no¹ | `fast` (<5 min), `medium` (<30 min), `soak` (hours) |
| `os.*` | no | `linux` |
| `e2e.*` | no | `stack`, `multinode` |

¹ Not linter-enforced, but **always declare it** — omitting it disables smart-sharding runtime weights.

**Authoritative source:** `framework/markers/taxonomy.py → MARKER_SCHEMA`. Add new values there first; never add them only in test files.

**Dotted syntax** (`@pytest.mark.ci.nightly`) is enabled by a `MarkDecorator.__getattr__` patch in `conftest.py`.

**Parametric markers** (not dimension-enforced):

- `@pytest.mark.gpu_vram(16)` — minimum VRAM in GB; `GpuAllocator` filters GPUs accordingly
- `@pytest.mark.gpu_count(4)` — GPUs to acquire; read by `multi_gpu_fixture` and `multi_node_fixture`
- `@pytest.mark.container_image("rocm/pytorch:6.3")` — per-test container image override
- `@pytest.mark.retry(count=N)` — per-test retry count; handled by `retry_plugin`

### Category Profiles — Auto-Injected Markers

`markers_plugin` injects markers at collection time for tests under profile directories **only when the test function has no existing marker in that dimension** (function-level always wins). `runtime.*` is absent from all profiles — declare it explicitly on every test.

> **Source of truth:** `CATEGORY_PROFILES` in `framework/markers/taxonomy.py`.

| Directory | Auto-injected markers |
|---|---|
| `tests/e2e/compiler/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `os.linux` |
| `tests/e2e/hip_runtime/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `os.linux` |
| `tests/e2e/hipblaslt/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `os.linux` |
| `tests/e2e/hwq_heuristic/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `os.linux` |
| `tests/e2e/rocprim/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `os.linux` |
| `tests/e2e/rocm_libs/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `os.linux` |

**Minimum valid test:**

```python
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
def test_example(dry_run_executor):
    result = dry_run_executor.run("echo RESULT_OK")
    assert result.ok
    assert "RESULT_OK" in result.stdout
```

---

## Executor Hierarchy

All GPU tests receive a `NodeExecutorGroup` from `target_executor`. Test code never sees the underlying executor type.

| Executor | Role | When active |
|---|---|---|
| `DryRunExecutor` | Synthetic stub; never shells out | `--no-gpu` / `hw.cpu_only` |
| `CpuExecutor` | Real subprocess, no GPU env | `hw.cpu_only` tests needing real commands |
| `LocalExecutor` | Local subprocess + `ROCR_VISIBLE_DEVICES` | Local `hw.gpu` and `hw.multi_gpu` |
| `ContainerExecutor` | Docker/Podman with AMD GPU passthrough | `--container-mode` |
| `SshExecutor` | SSH + `ROCR_VISIBLE_DEVICES` injection | Remote `hw.gpu` and `hw.multi_gpu` |
| `NodeExecutorGroup` | Uniform container wrapping 1 or N executors | All GPU tests |

**Fixture decision guide — use `target_executor` for all GPU tests:**

| Markers | `target_executor` yields | Test code |
|---|---|---|
| `hw.gpu` | `NodeExecutorGroup(1 exec)` | `target_executor.run(cmd)` |
| `hw.multi_gpu` + `gpu_count(N)` | `NodeExecutorGroup(1 exec, ROCR=0,1,...)` | `target_executor.run(cmd)` |
| `e2e.multinode` + `gpu_count(N)` | `NodeExecutorGroup(N execs)` | `for e in target_executor: e.run(cmd)` |
| `--no-gpu` (any) | `NodeExecutorGroup(DryRunExecutor)` | `target_executor.run(cmd)` |

**Never set `ROCR_VISIBLE_DEVICES` in test code.** Always go through `target_executor`.

---

## Fixtures Reference

### Framework / Session (`session_plugin`)

- `framework_config` — merged `FrameworkConfig` (session-scoped); priority: code defaults → `rocm-test.toml` → `ROCM_TEST_*` env → CLI flags
- `run_ctx` — unique run ID + start timestamp (session-scoped)
- `_attach_test_log` — autouse; attaches per-test executor log to Allure after every test

### Executors (function-scoped unless noted)

- `target_executor` — use for all GPU tests; dispatches by `hw.*`/`e2e.*` markers and CLI flags; yields `NodeExecutorGroup`
- `dry_run_executor` — `DryRunExecutor`; synthetic, no subprocess; for `hw.cpu_only` / PR gate
- `cpu_executor` — `CpuExecutor`; real subprocess, no GPU env
- `container_executor` — `ContainerExecutor` with AMD GPU passthrough; use `probe()` / `exec_in()` directly

### GPU / Hardware

- `node_pool` — session-scoped `NodePool`; `None` when `--no-gpu`
- `gpu_arch` — session-scoped `str | None`; reads `--gpu-arch`
- `health_fixture` — `GpuHealthChecker` from `health_plugin`; pre/post GPU checks per test

### Builder (session-scoped)

- `rock_dir` — TheRock/ROCm install path; resolved from `--rock-dir`, `ROCK_DIR`, `ROCM_TEST_THEROCK_ROCK_DIR`, or `rocm-test.toml`
- `compiler_build_dir` — binary output dir (default `output/test-binaries/`)
- `compile_binary` — `BinaryBuilder` factory; compiles `.cpp` → binary via `hipcc`; xdist-safe
- `cmake_build_dir` — CMake-based build factory; for `.hip` sources and multi-target builds
- `ld_path` — `{"LD_LIBRARY_PATH": "{rock_dir}/lib:..."}` dict for TheRock-linked binaries
- `arch_lib_path` — callable; resolves arch-specific library sub-path (e.g. for hipBLASLt tensile)

### Tests-level (from `tests/conftest.py`)

- `pytorch_env` — provisioned PyTorch environment; function-scoped with session-level sanity cache
- `torch_python` — Python executable inside the provisioned PyTorch venv
- `require_torch` — calls `pytest.fail` (not skip) when `pytorch_env` is not usable
- `require_torch_tunableop` — skips when `torch.cuda.tunable` is absent from the build
- `workload_scale` — `"smoke"` (default) or `"full"`; reads `ROCM_TEST_WORKLOAD_SCALE` env var
- `mock_gpu_info` / `mock_ok_result` / `mock_fail_result` — synthetic objects for unit tests

---

## Building Tests — Compiler Patterns

### Pattern A: Single C++ source, `hipcc`

Use `compile_binary` from `builder_plugin` for a single `.cpp` file:

```python
# tests/e2e/my_area/conftest.py
import pytest

@pytest.fixture(scope="session")
def my_kernel(compile_binary):
    return compile_binary(
        src="tests/e2e/my_area/src/kernel.cpp",
        output_name="kernel",
        subdir="my_area",        # → output/test-binaries/my_area/kernel
        arch="gfx942",           # optional; None = hipcc auto-detects
    )
```

### Pattern B: CMake build (`.hip` sources, multi-target, GTest)

Use `cmake_build_dir` from `builder_plugin` when `compile_binary` is insufficient — `.hip` file extensions require `enable_language(HIP)`, or multiple binaries share one `CMakeLists.txt`:

```python
# tests/e2e/my_area/conftest.py
import os
import pytest

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

`cmake_build_dir` automatically passes `-DROCM_PATH`, `-DCMAKE_PREFIX_PATH`, and `-DCMAKE_CXX_COMPILER`. Use `find_rocm_clangpp()` from `tests/common/spirv.py` to probe the three canonical `clang++` locations under `rock_dir`.

### Pattern C: External repo + CMake (`external_build`)

For third-party projects that require cloning before building (e.g. RCCL tests, QUDA):

```python
@pytest.fixture(scope="session")
def rccl_binary(rock_dir, compiler_build_dir, framework_config, external_build, cmake_executor):
    build_timeout = float(framework_config.therock.build_timeout_secs)
    clone = external_build.clone_repo(
        "https://github.com/ROCm/rccl-tests.git",
        "rccl/rccl-tests",
        ref="develop",
        timeout=build_timeout,
    )
    # then cmake_build_dir or subprocess cmake...
```

### CompileSpec registry (compiler area pattern)

The `tests/e2e/compiler/` area uses a `CompileSpec` dataclass registry so all binary metadata lives in one place:

```python
# tests/e2e/compiler/conftest.py
from dataclasses import dataclass

@dataclass
class CompileSpec:
    src: str
    output_name: str
    subdir: str = "compiler"
    std: str = "c++17"
    flags: str = ""
    include_dirs: list[str] = field(default_factory=list)

_SPECS = {
    "hip_app": CompileSpec(src="tests/e2e/compiler/src/hip_app.cpp", output_name="hip_app"),
}

def _build(name: str, compile_binary) -> str:
    spec = _SPECS[name]
    return compile_binary(spec.src, spec.output_name, subdir=spec.subdir, flags=spec.flags or None)

@pytest.fixture(scope="session")
def hip_app_binary(compile_binary):
    return _build("hip_app", compile_binary)
```

---

## Shared Test Utilities (`tests/common/`)

`tests/common/` is on `sys.path` and excluded from collection (`norecursedirs`). Import directly:

```python
from tests.common.factories import fake_gpu_info, fake_execution_result
from tests.common.spirv   import assert_spirv_offload_bundle
from tests.common.ml_provisioning.fixtures import ensure_pytorch_env
from tests.common.criu.fixtures             import ensure_criu_runtime, ensure_criu_runtime_target
from tests.common.criu.steps                import criu_dump, criu_restore, attach_criu_log
```

### `factories.py` — Unit test data

```python
gpu   = fake_gpu_info(arch="gfx1100", vram_mb=16384)
result = fake_execution_result(exit_code=0, stdout="RESULT_OK\nTHROUGHPUT=12.5\n")
```

### `spirv.py` — SPIR-V validation

```python
assert_spirv_offload_bundle(target_executor, rock_dir, binary_path, label="my_kernel")
```

Calls `llvm-objdump --offloading` and asserts `amdgcnspirv` is present in the output.

---

## PyTorch Provisioning

PyTorch on ROCm is provisioned by `tests/common/ml_provisioning/`. It installs `torch`, `torchvision`, and `torchaudio` wheels into a managed venv on the execution node. The coordinator process never imports PyTorch.

### Fixtures (defined in `tests/conftest.py`)

```python
@pytest.mark.runtime.medium
def test_pytorch_workload(require_torch, torch_python, target_executor, ld_path):
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld_path['LD_LIBRARY_PATH']} {torch_python} my_workload.py"
    )
    assert result.ok, result.stderr
```

| Fixture | Behavior on failure |
|---|---|
| `pytorch_env` | Returns `FrameworkProvisionResult`; `pytest.fail` if unusable |
| `require_torch` | `pytest.fail` — unusable PyTorch is a hard failure, not a skip |
| `require_torch_tunableop` | `pytest.skip` — TunableOp absence is a build-variant, not an error |
| `torch_python` | Returns the Python executable in the managed venv |

### Installation modes

| Mode | Channel | Fallback |
|---|---|---|
| `auto` (default) | `multiarch → family` | Tries multiarch first; falls back to family v2 |
| `multiarch` | Single multi-arch index, `torch[device-gfxNNN]` extras | Single candidate |
| `family` | Per-arch v2 index (`<base>/<gfx_family>/`) | Single candidate |
| `staging` | Pre-promotion multi-arch index | Single candidate; never auto-selected |

### CLI pre-install

```bash
# Auto: production wheel path, GPU detected at runtime
pytest ... --pre-install "pytorch=mode=auto"

# Multi-arch with explicit GPU target
pytest ... --pre-install "pytorch=mode=multiarch,device=gfx942"

# Exact torch pin (disables fallback)
pytest ... --pre-install "pytorch=mode=multiarch,device=gfx942,torch=2.14.0a0+rocm7.12.0a20260716"

# Family v2 channel
pytest ... --pre-install "pytorch=mode=family,gfx_family=gfx94X-dcgpu"
```

Full install option reference and the three-phase fixture flow are documented in `tests/common/ml_provisioning/README.md`.

---

## CRIU Checkpoint-Restore

`tests/common/criu/` provides CRIU build, install, and runtime helpers for checkpoint-restore tests (`tests/e2e/recovery/criu/`). Reuse these helpers for any new area that needs GPU process checkpoint-restore.

### Entry points

```python
# Host/SSH path (baremetal + remote): clone via external_build, build on host
from tests.common.criu.fixtures import ensure_criu_runtime
criu_prefix = ensure_criu_runtime(external_build, cmake_executor, framework_config)

# Container/target path: installer self-clones + builds inside the target
from tests.common.criu.fixtures import ensure_criu_runtime_target
criu_prefix = ensure_criu_runtime_target(target_executor, framework_config)
```

Both return the shell command prefix (`sudo -n env PATH=... criu`) and auto-install CRIU + the amdgpu plugin when missing. Set `ROCM_TEST_CRIU_AUTO_INSTALL=0` to disable auto-install; `ROCM_TEST_CRIU_VERSION=<tag>` to pin the version (default `v4.1`).

### Runtime helpers (`tests/common/criu/steps.py`)

```python
from tests.common.criu.steps import criu_dump, criu_restore, attach_criu_log, kill_pid

dump_result    = criu_dump(executor, criu_prefix, workdir, pid)
restore_result = criu_restore(executor, criu_prefix, workdir)

assert "OK" in dump_result.stdout
assert "PID_GONE" in dump_result.stdout   # process removed after dump
assert "RESTORE_OK" in restore_result.stdout

attach_criu_log(executor, workdir, "dump.log")
```

### Prerequisites

- Passwordless `sudo` on the test node (checked at fixture setup; suite skips cleanly if absent)
- C toolchain, `libprotobuf-dev`, `libdrm-dev`, and related build deps (auto-installed by `installer.py`)
- Network access for `git clone` on the first run (or set `CRIU_REPO` to a local mirror)

### Standalone pre-installation

Run `installer.py` manually on a fleet node to avoid auto-install overhead at test time:

```bash
python3 tests/common/criu/installer.py v4.1
# or with an existing checkout:
python3 tests/common/criu/installer.py --src-dir /path/to/criu v4.1
```

### Extending CRIU tests

When adding a new CRIU test area:

1. Import `ensure_criu_runtime` or `ensure_criu_runtime_target` in the area's `conftest.py`.
2. Declare a session-scoped fixture that calls the helper and returns the `criu` prefix string.
3. Use `criu_dump` / `criu_restore` / `attach_criu_log` from `steps.py` in test functions.
4. Never call `criu` directly — always use the prefix string returned by `ensure_criu_runtime*`.

---

## Remote Fleet Configuration

`--remote-node host.yaml` enables `NodePool`. GPU detection runs once at session start (parallel for remote nodes). xdist workers receive the topology from the master — no redundant SSH calls.

```yaml
# host.yaml
HOST_IDX_1:
  HOSTNAME: gpu-node-01.example.com
  USERNAME: ci
  SSH_KEY:  ~/.ssh/ci_rsa
HOST_IDX_2:
  HOSTNAME: gpu-node-02.example.com
  USERNAME: ci
  SSH_KEY:  ~/.ssh/ci_rsa
```

**Fleet CLI flags:**

| Flag | Default | Plugin |
|---|---|---|
| `--remote-node PATH` | — | `remote_node_plugin` |
| `--gpu-acquire-timeout N` | 180 s | `remote_node_plugin` |
| `--gpu-health-metrics METRICS` | — | `remote_node_plugin` |
| `--monitor-gpu` | off | `remote_node_plugin` |
| `--no-gpu` | off | `gpu_plugin` |
| `--gpu-arch ARCH` | — | `gpu_plugin` |
| `--rocm-config PATH` | auto-find `rocm-test.toml` | `gpu_plugin` |
| `--schedule-policy` | `resource-most` | `scheduling_plugin` |
| `--collect-runtimes PATH` | — | `scheduling_plugin` |
| `--vram-headroom-gb GB` | 2.0 | `scheduling_plugin` |
| `--container-mode` | off | `executor_plugin` |
| `--container-image IMAGE` | — | `executor_plugin` |
| `--retry-count N` | 0 | `retry_plugin` |
| `--rock-dir PATH` | — | `builder_plugin` |
| `--compiler-build-dir PATH` | `output/test-binaries/` | `builder_plugin` |
| `--pre-install rocm=X` / `pkg=X` / `pytorch=...` | — | `install_plugin` |

---

## rocm-test.toml Config Reference

```toml
[framework]
log_level     = "normal"        # "quiet" / "normal" / "verbose"
run_id_prefix = "rocm-test"
artifact_dir  = "output/artifacts/"
session_log   = "output/logs/session.log"

[gpu]
detection         = "auto"      # "auto" / "kfd" / "amd-smi"
max_temp_celsius  = 90
max_ecc_errors    = 0
min_vram_free_mb  = 512
health_metrics    = ["temp", "vram", "util", "ecc", "clock"]
monitor_metrics   = ["temp", "vram", "util", "ecc", "clock"]
monitor_interval_secs = 15.0
monitor_duration_secs = 0.0     # 0 = stop when test ends

[frameworks]
default_mode      = "auto"      # PyTorch install mode: auto / multiarch / family / staging
multiarch_index   = "https://rocm.nightlies.amd.com/whl-multi-arch/"
family_index_base = "https://rocm.nightlies.amd.com/v2"
staging_index     = "https://rocm.nightlies.amd.com/whl-staging-multi-arch/"
requirements_pytorch = "tests/common/ml_provisioning/requirements-pytorch.txt"

[therock]
rock_dir     = ""               # override via --rock-dir / ROCK_DIR / ROCM_TEST_THEROCK_ROCK_DIR
rocm_version = ""               # optional nightly date hint e.g. "7.14.0a20260624"
build_dir    = "output/test-binaries/"

[results]
upload_mode = "auto"
local_dir   = "output/results/"
sqlite_db   = "output/rocm_test.db"

[reporting]
allure_results_dir = "output/artifacts/allure-results/"
history_depth      = 5
```

---

## CI Workflows

| Workflow | Trigger | GPU |
|---|---|---|
| `pre-commit.yml` | Every PR | No — DryRun tests, lint, marker lint, MkDocs strict build |
| `e2e-nightly.yml` | UTC 03:00 daily + `workflow_dispatch` | Yes — `gfx942` or `amdgpu_family` input |

---

## Agent Skills

Three slash commands are available in Claude Code. The marker-lint hook in `.claude/settings.json` runs `MarkerLinter` automatically every time a file is written or edited under `tests/` — violations surface at write time, not at PR review.

| Command | When to use |
|---|---|
| `/creator` | Generate a complete, marker-compliant test from a GPU feature description or requirements doc |
| `/refiner [review-as <persona>] <file>` | Review (4-persona or single), detect flakiness, extend with edge cases |
| `/porter <source-file>` | Port an external script, shell test, or non-compliant pytest into `rocm-tests` |

**Typical workflow:**

```bash
/creator
# → describe the feature; agent reads framework + nearest test area before generating

pytest tests/e2e/my_area/ --collect-only -q --no-gpu   # validate collection

/refiner tests/e2e/my_area/test_my_feature.py          # four-persona review
```

**Invariants — things Claude must never do:**

- Never set `ROCR_VISIBLE_DEVICES` in test code.
- Never add a marker value only in a test file — add it to `MARKER_SCHEMA` first.
- Never duplicate framework logic — consume it from `framework/`.
- Never commit secrets or credentials.
- Never move `markers_plugin` away from position 1 or `session_plugin` away from position 2 in `pytest_plugins`.
- Never import `torch` on the coordinator process — run all PyTorch code via `target_executor.run(f"{torch_python} ...")`.
