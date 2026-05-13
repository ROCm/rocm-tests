# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rocm-test` is an AMD ROCm System End-to-End Test Framework. It validates the full ROCm software stack (kernel driver → HIP runtime → compute libraries → ML frameworks) on AMD GPU hardware, extensible to firmware. 

---

## Build & Test Commands

**Install dependencies:**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt   # runtime deps + lint/type/docs tools
```

> Note: add 'uv' as prefix to expedite installation.

**Run a single test file:**
```bash
pytest tests/e2e/stack_validation/test_hip_runtime.py -v
```

**Preview matched tests without executing:**
```bash
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --collect-only -q --no-gpu
```

**Run nightly GPU tests for a specific architecture:**
```bash
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --gpu-arch gfx942 --alluredir=build/allure-results -v
```

**Generate a lightweight HTML report (pytest-html — no Allure CLI required):**
```bash
pytest tests/ --no-gpu --html=build/report.html --self-contained-html -v
```
> `--self-contained-html` bundles CSS/JS into one file. Requires `pytest-html<4` (already pinned in `requirements.txt` — v4 removed that flag).

**Run against a remote GPU fleet:**
```bash
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --remote-node host.yaml -n 4 -v
```

**Pre-install ROCm and packages on fleet nodes before tests:**
```bash
pytest tests/e2e/ -m "ci.nightly" --remote-node host.yaml \
  --pre-install rocm=6.4.0 --pre-install pkg=libssl-dev -n 4 -v
```

**Lint (line length is 120):**
```bash
ruff check framework tests
black --check --diff framework tests
mypy framework --show-error-codes
pylint framework --fail-under=9.5
```

**Auto-fix formatting:**
```bash
ruff check --fix framework tests && black framework tests
```

**Security scan:**
```bash
bandit -r framework -c pyproject.toml
pip-audit -r requirements.txt
```

**Build docs:**
```bash
mkdocs build --strict --site-dir build/site
mkdocs serve   # live preview at http://localhost:8000
```

---

## Coding Standards

**Style:** PEP 8, 120-character line length (configured in `black`, `ruff`, `pylint`). Google-style docstrings on all modules; `mkdocstrings` generates the API docs site.

**Test naming:** Files must match `test_*.py`. Test functions must begin with `test_`. Classes use `PascalCase`.

**Imports:** Group as Standard Library → Third-Party → Local Framework (`framework.*`).

**Marker requirements:** Every test function **must** carry at least one marker from each required dimension:

Markers act as flexible metadata tags that empowers to curate test execution: defining what tests to run, where to run (platform), and when (pre-commit, nightly, weekly). Structured hierarchically—applying default markers at the test directory level while allowing specific test cases to overwrite or add context-specific markers

| Dimension   | Required | Values |
|-------------|----------|--------|
| `hw.*`      | YES | `gpu`, `multi_gpu`, `cpu_only` |
| `ci.*`      | YES | `pr`, `nightly`, `weekly`, `smoke_e2e` |
| `layer.*`   | YES | `driver`, `runtime`, `math_lib`, `ml_framework`, `debug_stack` |
| `runtime.*` | no¹ | `fast` (<5 min), `medium` (<30 min), `longevity` (<2 hr), `soak` (hours) |
| `os.*`      | no | `linux`, `windows`, `wsl`, `both` |
| `e2e.*`     | no | `stack`, `multinode`, `app`, `upgrade` |

¹ Not enforced by the linter (`REQUIRED_DIMENSIONS = {"hw", "ci", "layer"}`), but always declare it explicitly — omitting it disables smart-sharding runtime weights.

Marker values are defined in `framework/markers/taxonomy.py` → `MARKER_SCHEMA`. **Never add new values only in test files** — add them to `MARKER_SCHEMA` first.

Dotted syntax (`@pytest.mark.ci.pr`) is enabled by a `MarkDecorator.__getattr__` patch in `conftest.py`.

**Parametric markers** (not linted as dimensions):
- `@pytest.mark.gpu_vram(16)` — minimum VRAM in GB for GPU allocation
- `@pytest.mark.gpu_count(4)` — number of GPUs to acquire
- `@pytest.mark.retry(count=2)` — per-test retry count (overrides `--retry-count`)
- `@pytest.mark.container_image("rocm/pytorch:6.3")` — per-test container image override

**Minimum valid test:**
```python
@pytest.mark.ci.pr
@pytest.mark.layer.runtime
@pytest.mark.hw.cpu_only
def test_example(dry_run_executor):
    result = dry_run_executor.run("echo RESULT_OK")
    assert result.ok
```

**GPU environment:** Never set `ROCR_VISIBLE_DEVICES` in test code — always go through `target_executor`.

**Config:** Never commit secrets. Use `ROCM_TEST_NOTIFICATIONS_WEBHOOK_URL` and similar env vars.

---

## Architecture Context

### Layer Stack

```
conftest.py (root)
    └── pytest_plugins → framework/plugins/*.py   (registration order matters — markers_plugin MUST be first)
            ├── markers_plugin.py       # FIRST: category-profile marker injection
            ├── gpu_plugin.py           # --no-gpu/--gpu-arch/--mock-gpu, gpu_fixture
            ├── remote_node_plugin.py   # --remote-node/--gpu-acquire-timeout, NodePool, target_executor
            ├── scheduling_plugin.py    # --schedule-policy/--collect-runtimes/--vram-headroom-gb
            ├── executor_plugin.py      # --container-mode/--container-image, executor fixtures
            ├── os_plugin.py            # os_adapter/platform_name fixtures, os.* marker skip hook
            ├── health_plugin.py        # Pre/post GPU health gates (temp, ECC, VRAM, clocks)
            ├── baseline_plugin.py      # Regression compare vs baselines/*.yaml
            ├── artifacts_plugin.py     # Allure attachment on failure
            ├── prereqs_plugin.py       # Session-level driver/ROCm version checks
            ├── retry_plugin.py         # --retry-count, retry_fixture, @pytest.mark.retry
            ├── reports_plugin.py       # Allure label mapping, terminal summary, --allure-log-name/--allure-db
            ├── builder_plugin.py       # --rock-dir/--compiler-build-dir, compile_binary/ld_path fixtures
            └── install_plugin.py       # --pre-install rocm=X/pkg=X, parallel pre-session node install

framework/
    config/       # rocm-test.toml → env vars → code defaults cascade (FrameworkConfig dataclasses)
    common/       # ExecutionResult, parse_metric(), Outcome, classify(), retry decorator
    executors/    # AbstractExecutor + concrete backends (see Executor Hierarchy below)
    nodes/        # NodePool fleet manager: NodeSpec, NodeSlot, MultiGpuSlots, GpuFileLock, PendingTracker
    scheduling/   # DynamicScheduler, SchedulePolicy — resource-aware xdist scheduling
    builder/      # BinaryBuilder — hipcc compilation with xdist locking + incremental builds
    gpu/          # GpuDetector, MockGpuDetector, GpuAllocator, GpuDrainChecker, GpuBackgroundMonitor
    markers/      # MARKER_SCHEMA taxonomy, MarkerLinter
    reporting/    # AllureReporter, step(), attach_text(), report_metric()
    os_adapter/   # Linux + Windows GPU enumeration behind one interface
    rocm/libs/    # ROCm library helpers: hip.py, rccl.py, amd_smi.py, stack.py

tests/
    common/       # Test data factories (fake_gpu_info, fake_execution_result) — NOT test files
    dry_run/      # Config and DryRun tests (no GPU required, ci.pr)
    e2e/
        compiler/               # hipcc compilation tests
        concurrent_collectives/ # RCCL concurrent collective stress tests
        hwq_heuristic/          # GPU hardware queue heuristic tests
    e2e/performance/baselines/  # Per-arch YAML baselines (not test files)
```

### Executor Hierarchy

All GPU tests receive a `NodeExecutorGroup` from `target_executor`. Tests never see the underlying executor type.

| Executor | Role | When used internally |
|---|---|---|
| `DryRunExecutor` | Synthetic stub; never shells out | `--no-gpu` / `hw.cpu_only` |
| `CpuExecutor` | Real subprocess, no GPU env | `hw.cpu_only` tests needing real commands |
| `LocalExecutor` | Local subprocess; injects `ROCR_VISIBLE_DEVICES` | Local `hw.gpu` and `hw.multi_gpu` |
| `ContainerExecutor` | Docker/Podman with AMD GPU passthrough | `--container-mode` |
| `SshExecutor` | Raw SSH | Used by `SshGpuExecutor` and `NodePool` — not in tests |
| `SshGpuExecutor` | SSH + `ROCR_VISIBLE_DEVICES` injection | Remote `hw.gpu` and `hw.multi_gpu` |
| `LabeledExecutor` | Wraps any executor; prefixes lines with `[test\|node\|GPU]` | Created by `NodeSlot.make_executor()` |
| `NodeExecutorGroup` | Uniform container returned by all GPU fixtures | Always: wraps 1 or N `LabeledExecutor`s |

**Fixture decision guide — use `target_executor` for all GPU tests:**

| Markers on test | `target_executor` yields | Test code |
|---|---|---|
| `hw.gpu` | `NodeExecutorGroup(1 exec)` | `target_executor.run(cmd)` |
| `hw.multi_gpu` + `gpu_count(N)` | `NodeExecutorGroup(1 exec, ROCR=0,1,...)` | `target_executor.run(cmd)` |
| `e2e.multinode` + `gpu_count(N)` | `NodeExecutorGroup(N execs, 1 per node)` | `for e in target_executor: e.run(cmd)` |
| `--no-gpu` (any) | `NodeExecutorGroup(DryRunExecutor)` | `target_executor.run(cmd)` |

**Background processes** — run a daemon alongside a test and capture its output:
```python
with cpu_executor.start_background(
    "rocm-smi --showmetrics --interval=2",
    log_path="output/artifacts/executor-logs/test__monitor.log",
) as monitor:
    result = target_executor.run("./my_kernel")
    assert result.ok
    assert monitor.is_alive

stopped = monitor.stop_result   # ExecutionResult with daemon's captured output
```
`DryRunExecutor.start_background()` returns a `NoOpBackgroundProcess` (same API, never alive).

**BinaryBuilder** (`compile_binary` fixture) — compiles HIP/C++ via `hipcc` in a CPU-only subprocess; xdist-safe via file locking:
```python
@pytest.fixture(scope="session")
def my_kernel(compile_binary):
    return compile_binary(
        src="tests/e2e/myarea/src/kernel.cpp",
        output_name="kernel",
        subdir="myarea",   # → output/test-binaries/myarea/kernel
        arch="gfx942",     # optional; None = hipcc auto-detects
    )
```

### Category Profiles (Auto-Injected Markers)

`markers_plugin.py` injects markers at collection time for tests under these directories **if no function-level marker exists for that dimension** (function-level always wins):

| Directory | Auto-injected markers |
|---|---|
| `tests/e2e/compiler` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` |
| `tests/e2e/concurrent_collectives` | `hw.multi_gpu`, `layer.math_lib`, `ci.nightly`, `e2e.stack`, `os.linux` |
| `tests/e2e/hwq_heuristic` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `e2e.stack`, `os.linux` |

`runtime.*` is intentionally absent from all profiles — declare it explicitly on every test function.

> **Note:** Paths for `ml_frameworks`, `multi_gpu`, `stack_validation`, and `debug_stack` are reserved in `taxonomy.py` (`CATEGORY_PROFILES`) for future test areas — those directories do not yet exist under `tests/e2e/`.

### Dynamic Scheduling & Smart Sharding

Two complementary mechanisms control test ordering and GPU assignment — both are automatic and require no manual test grouping.

**`DynamicScheduler`** (`framework/scheduling/dynamic_scheduler.py`, activated by `scheduling_plugin`):

- Runs during `pytest_collection_modifyitems`; no-op when `--no-gpu` is active.
- **Step 1 — xdist_group assignment:** Each multinode test gets `xdist_group = "multinode_N"`; each multi-GPU test gets `xdist_group = "multi_gpu_{count}_{N}"` — both unique per test so separate workers can run different multi-GPU tests in parallel. Single-GPU tests get no group (xdist worksteal).
- **Step 2 — Sort by `--schedule-policy`:**
  - `resource-most` (default): multinode → multi-GPU DESC (by `gpu_count`) → single-GPU. Multi-GPU workers block inside fixture acquisition; single-GPU tests fill free slots emergently.
  - `resource-least`: single-GPU → multi-GPU ASC → multinode. Maximises time-to-first-result.
- **VRAM headroom:** `--vram-headroom-gb` (default 2.0) reserves headroom per GPU; `@pytest.mark.gpu_vram(N)` tests skip GPUs where `total_vram_gb − headroom < N`.
- **Recommended `-n`:** call `DynamicScheduler.recommended_workers()` = total GPU slots across all nodes; printed at session start; pass as `-n` for optimal parallelism.
- **Runtime collection:** `--collect-runtimes PATH` writes `{nodeid, duration_secs, outcome}` JSON at session end — informational only, not used for scheduling.

**Decision guide — which to use:**

| Setup | Mechanism | Command |
|---|---|---|
| Single node, multiple GPUs, no xdist | Smart Sharding | `pytest tests/ -m "hw.gpu"` |
| Single node, multiple GPUs, xdist | DynamicScheduler | `pytest tests/ -m "hw.gpu" -n 4` |
| Multi-node fleet | DynamicScheduler | `pytest tests/ --remote-node host.yaml -n 4` |
| `--no-gpu` / DryRun | Neither (no-op) | `pytest tests/ --no-gpu` |

**Marker interaction:** `runtime.*` feeds Smart Sharding weights — always declare it explicitly. `@pytest.mark.gpu_vram(N)` and `@pytest.mark.gpu_count(N)` are read by both mechanisms for filtering and group assignment.

### Fixtures

**Framework / session:**
- `framework_config` — merged `FrameworkConfig` (session-scoped)
- `run_ctx` — unique run ID + timestamp (session-scoped)
- `os_adapter` — `AbstractOsAdapter`: `list_gpu_device_paths()`, `is_module_loaded()`, `load_kernel_module()`, `get_platform_name()` (session-scoped)
- `platform_name` — `"linux"` / `"windows"` / `"wsl"` string (session-scoped)

**Executors (function-scoped unless noted):**
- `target_executor` — **use for all GPU tests**; yields `NodeExecutorGroup`; dispatches based on `hw.*`/`e2e.*` markers and `gpu_vram`/`gpu_count`
- `dry_run_executor` — `DryRunExecutor`; synthetic, no subprocess; for `hw.cpu_only` / PR gate tests
- `cpu_executor` — `CpuExecutor`; real subprocess, no GPU env; for `hw.cpu_only` tests needing real commands
- `container_executor` — `ContainerExecutor` with AMD GPU passthrough; use `probe()` / `exec_in()` directly
- `session_executor` — legacy single-GPU fixture; prefer `target_executor` for new tests
- `remote_pool` — legacy `RemoteNodePool`; prefer `target_executor` with `e2e.multinode`

**GPU / hardware:**
- `node_pool` — Session-scoped `NodePool`; `None` when `--no-gpu` is active
- `gpu_fixture` — acquires a real or mock GPU; runs health checks in setup/teardown
- `multi_gpu_fixture` — explicit N-GPU from ONE node; yields `NodeExecutorGroup`; prefer `target_executor`
- `multi_node_fixture` — explicit per-node; yields `NodeExecutorGroup`; prefer `target_executor`
- `health_fixture` — `GpuHealthChecker` configured from `[gpu]` thresholds in `rocm-test.toml`

**Builder (session-scoped):**
- `rock_dir` — resolved TheRock/ROCm install path (from `--rock-dir`, `ROCK_DIR`, or `rocm-test.toml`)
- `compiler_build_dir` — binary output dir (default `output/test-binaries/`)
- `compile_binary` — `BinaryBuilder` factory; compiles `.cpp` → binary via `hipcc`
- `ld_path` — `{"LD_LIBRARY_PATH": "{rock_dir}/lib:..."}` dict for TheRock-linked binaries

**Retry / reporting:**
- `retry_fixture` — `RetryHelper`; use `.run(executor, cmd)` for manual retry; configured by `@pytest.mark.retry(count=N)` > `--retry-count` > default 1 attempt; marks test `flaky` in Allure on late success
- `outcome_fixture` — exposes the classified `Outcome` post-test; auto-detects pass/fail/skip; attaches label to Allure
- `baseline_fixture` — regression comparison against per-arch YAML baselines
- `artifacts_fixture` / `allure_reporter` — Allure attachment on failure
- `mock_gpu_info` / `mock_ok_result` / `mock_fail_result` — synthetic test objects (from `tests/conftest.py`)

### Remote Fleet Configuration

`--remote-node host.yaml` enables the `NodePool` fleet manager. GPU detection runs once at session start (in parallel for remote nodes). xdist workers receive topology from the master — no redundant SSH.

```yaml
HOST_IDX_1:
  HOSTNAME: gpu-node-01.example.com
  USERNAME: ci
  SSH_KEY:  ~/.ssh/ci_rsa
  # GPU_ARCH: gfx942   # optional: filter GPUs by arch
HOST_IDX_2:
  HOSTNAME: gpu-node-02.example.com
  USERNAME: ci
  SSH_KEY:  ~/.ssh/ci_rsa
```

> Passwords not to be stored statically, obtain from secrets/vault or from ~/.env during execution

**All CLI flags by plugin:**

| Flag | Default | Plugin |
|---|---|---|
| `--remote-node PATH` | — | `remote_node_plugin` |
| `--gpu-acquire-timeout N` | 180 s | `remote_node_plugin` |
| `--gpu-health-metrics METRICS` | — | `remote_node_plugin` |
| `--monitor-gpu` | off | `remote_node_plugin` |
| `--gpu-drain-secs SECS` | 0.5 | `remote_node_plugin` |
| `--gpu-drain-timeout SECS` | 30 | `remote_node_plugin` |
| `--no-gpu` | off | `gpu_plugin` |
| `--gpu-arch ARCH` | — | `gpu_plugin` |
| `--schedule-policy {resource-most,resource-least}` | `resource-most` | `scheduling_plugin` |
| `--collect-runtimes PATH` | — | `scheduling_plugin` |
| `--vram-headroom-gb GB` | 2.0 | `scheduling_plugin` |
| `--container-mode` | off | `executor_plugin` |
| `--container-image IMAGE` | — | `executor_plugin` |
| `--container-runtime {docker,podman}` | `docker` | `executor_plugin` |
| `--retry-count N` | 0 | `retry_plugin` |
| `--allure-log-name NAME` | — | `reports_plugin` |
| `--allure-db N` | 0 | `reports_plugin` |
| `--html PATH` | — | `pytest-html` (lightweight alternative to Allure) |
| `--self-contained-html` | off | `pytest-html` (bundles CSS/JS; unavailable in v4+) |
| `--rock-dir PATH` | — | `builder_plugin` |
| `--compiler-build-dir PATH` | `output/test-binaries/` | `builder_plugin` |
| `--pre-install rocm=X` / `pkg=X` | — | `install_plugin` |
| `--rocm-config PATH` | auto-find `rocm-test.toml` | `gpu_plugin` |

### rocm-test.toml Config Sections

```toml
[framework]
log_level     = "normal"       # "debug"/"verbose" enables stream_stdout on executors
run_id_prefix = "rocm-test"
artifact_dir  = "output/artifacts/"
session_log   = "output/artifacts/session.log"   # aggregate per-test log across all xdist workers

[gpu]
detection             = "auto"
max_temp_celsius      = 90
max_ecc_errors        = 0
min_vram_free_mb      = 512
health_metrics        = ["temp", "vram", "util", "ecc", "clock"]   # point-in-time snapshots
monitor_metrics       = ["temp", "vram", "util", "ecc", "clock"]   # continuous background poller
monitor_interval_secs = 15.0
monitor_duration_secs = 0.0

[therock]
rock_dir  = ""              # path to TheRock/ROCm install; also --rock-dir / ROCK_DIR env
build_dir = "output/test-binaries/"
# build_timeout_secs / build_inactivity_timeout_secs come from code defaults (7200 / 600)

[baselines]
regression_pct = 5.0
baseline_dir   = "tests/performance/baselines/"

[reporting]
allure_results_dir = "output/artifacts/allure-results/"
history_depth = 5          # number of prior runs kept by --allure-db

[results]
upload_mode = "auto"
local_dir   = "output/results/"
sqlite_db   = "output/rocm_test.db"
```

### CI Workflows

| Workflow | Trigger | GPU needed |
|---|---|---|
| `pre-commit.yml` | Every PR | No (runs black, ruff, mypy, pylint) |
| `e2e-nightly.yml` | UTC 03:00 daily; `workflow_call`; `workflow_dispatch` | Yes (`gfx90a` default; `amdgpu_family` input) |

---

## Agent Skills

For deep architectural guidance, test authoring workflows, and agent-specific personas, **always refer to the documentation in `.claude/`**.

| Command | When to Use |
|---|---|
| `/creator` | Generate a complete, marker-compliant test from a GPU feature description or requirements doc |
| `/refiner [review-as <persona>] <file>` | Review (4-persona or single), detect flakiness, and extend an existing test with edge cases |
| `/porter <source-file>` | Port an external script, shell test, or non-compliant pytest into rocm-tests |

The marker-lint hook in `.claude/settings.json` runs `MarkerLinter` automatically on every file written or edited under `tests/` — violations surface at write time, not at PR review.

**Typical workflow:**
```bash
/creator                   # describe the feature; agent generates a marker-compliant file
pytest tests/e2e/... --collect-only -q --no-gpu   # validate collection
/refiner tests/e2e/...     # four-persona review before opening a PR
```
