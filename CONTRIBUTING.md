# Contributing to rocm-tests

---

## Quickstart

### Step 1 — Clone and install

```bash
git clone https://github.com/ROCm/rocm-tests.git
cd rocm-tests
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Faster install with uv (optional)
uv pip install -r requirements-dev.txt
```

### Step 2 — Dry run

```bash
# Collect and lint all tests (DryRun — no GPU required)
pytest tests/ --collect-only -q --no-gpu

# Run PR-gate tests
pytest tests/ -m "ci.pr" --no-gpu -v
```

### Step 3 — Run on real hardware

```bash
# Smoke suite — fast PR-gate tests on real GPU
pytest tests/e2e/ -m "hw.gpu and ci.pr" -v

# Full nightly matrix for a specific architecture
pytest tests/e2e/ -m "hw.gpu and ci.nightly" --gpu-arch gfx942 -v
```

**Common pytest flags:**

| Flag | Effect |
|---|---|
| `-m "<expression>"` | Select by marker (`hw.gpu and ci.nightly`, `not hw.gpu`) |
| `--no-gpu` | DryRun mode — no GPU required |
| `--collect-only -q` | Preview matched tests without running |
| `-x` | Stop after the first failure |
| `--gpu-arch gfx942` | Filter GPUs by architecture |
| `--remote-node host.yaml` | Run against a remote GPU fleet |
| `-n 4` | Parallel xdist workers |

---

## Marker System

Every test function **must** carry at least one marker from each required dimension:

| Dimension | Required | Values |
|---|---|---|
| `hw.*` | YES | `gpu`, `multi_gpu`, `cpu_only` |
| `ci.*` | YES | `pr`, `nightly`, `weekly` |
| `layer.*` | YES | `runtime`, `math_lib` |
| `runtime.*` | no¹ | `fast` (<5 min), `medium` (<30 min), `soak` (hours) |
| `os.*` | no | `linux` |
| `e2e.*` | no | `stack`, `multinode` |

¹ Not linter-enforced but always declare it — omitting it disables smart-sharding runtime weights.

**Authoritative source:** `framework/markers/taxonomy.py → MARKER_SCHEMA`. Never add new
marker values only in test files — add them to `MARKER_SCHEMA` first.

**Dotted syntax** (`@pytest.mark.ci.pr`) is enabled by a `MarkDecorator.__getattr__` patch
in `conftest.py`. IDE linters may flag it — this is expected.

**Parametric markers** (not dimension-enforced):
- `@pytest.mark.gpu_vram(16)` — minimum VRAM in GB for GPU allocation
- `@pytest.mark.gpu_count(4)` — number of GPUs to acquire
- `@pytest.mark.container_image("rocm/pytorch:6.3")` — per-test container image override

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

## Two-Step Test Registration

### Existing test area (most common)

If your test belongs to an existing `tests/e2e/<domain>/` directory that already has a
`CATEGORY_PROFILES` entry, just add your test file — required dimension markers are
injected automatically. You only need to declare `@pytest.mark.runtime.<value>` explicitly.
Also existing default markers from category profiles can be overwritten from tests.

Profiled directories and their auto-injected markers (Example below):

| Directory | Auto-injected markers |
|---|---|
| `tests/e2e/compiler/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `os.linux` |
| `tests/e2e/rocprim/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `os.linux` |

### New test area

1. Add the directory profile to `framework/markers/taxonomy.py → CATEGORY_PROFILES`:

   ```python
   "tests/e2e/my_feature": [
       "hw.gpu", "layer.runtime", "ci.nightly", "os.linux",
   ],
   ```

2. Place your `test_*.py` in `tests/e2e/my_feature/`. Add any new marker values to
   `MARKER_SCHEMA` in the same `taxonomy.py` file if needed.

3. If your tests run compiled binaries, add a `conftest.py` in the new directory with a
   session-scoped build fixture:

   ```python
   # tests/e2e/my_feature/conftest.py
   import pytest
   from tests.common._cmake_build import cmake_build

   @pytest.fixture(scope="session")
   def my_binary(rock_dir, gpu_arch, compiler_build_dir):
       build_dir = cmake_build(
           src="tests/e2e/my_feature/src",
           build_dir=f"{compiler_build_dir}/my_feature",
           rocm_path=rock_dir,
           gpu_arch=gpu_arch,
       )
       return f"{build_dir}/my_binary"
   ```

   For single `.cpp` files without CMake, use `compile_binary` directly:

   ```python
   @pytest.fixture(scope="session")
   def my_binary(compile_binary):
       return compile_binary(
           src="tests/e2e/my_feature/src/kernel.cpp",
           output_name="kernel",
           subdir="my_feature",
       )
   ```

   Tests that only use `target_executor.run()` on pre-installed system binaries need no
   `conftest.py` — all framework fixtures are available automatically.

---

## Fixture Discovery

Framework fixtures are loaded automatically via `conftest.py` → `pytest_plugins`. You never
need to import them — declare them as function parameters.

**Framework / session:**

| Fixture | Purpose |
|---|---|
| `framework_config` | Merged `FrameworkConfig` (session-scoped) |
| `run_ctx` | Unique run ID + timestamp (session-scoped) |
| `os_adapter` | GPU device enumeration and kernel module interface |
| `platform_name` | `"linux"` / `"windows"` |

**Executors — use `target_executor` for all GPU tests:**

| Fixture | Purpose |
|---|---|
| `target_executor` | **Primary GPU fixture.** Yields `NodeExecutorGroup`. Dispatches automatically based on `hw.*`/`e2e.*` markers and `--no-gpu`/`--container-mode`/`--remote-node` flags. |
| `dry_run_executor` | Synthetic stub, never shells out. For `hw.cpu_only` and PR gate tests. |
| `cpu_executor` | Real subprocess, no GPU env. For `hw.cpu_only` tests needing real commands. |
| `container_executor` | Docker/Podman with AMD GPU passthrough. Use `probe()`/`exec_in()` directly. |

**Fixture decision guide:**

| Markers on test | `target_executor` yields | Test code |
|---|---|---|
| `hw.gpu` | `NodeExecutorGroup(1 exec)` | `target_executor.run(cmd)` |
| `hw.multi_gpu` + `gpu_count(N)` | `NodeExecutorGroup(1 exec, ROCR=0,1,...)` | `target_executor.run(cmd)` |
| `e2e.multinode` + `gpu_count(N)` | `NodeExecutorGroup(N execs, 1 per node)` | `for e in target_executor: e.run(cmd)` |
| `--no-gpu` (any) | `NodeExecutorGroup(DryRunExecutor)` | `target_executor.run(cmd)` |

**Never set `ROCR_VISIBLE_DEVICES` in test code** — always go through `target_executor`.
Note: Support to opt gpu_index and pinning from testcase shall be enabled shortly.

**Builder (session-scoped):**

| Fixture | Purpose |
|---|---|
| `rock_dir` | Path to ROCm install (from `--rock-dir`, `ROCK_DIR`, or `rocm-test.toml`) |
| `compiler_build_dir` | Binary output directory (default `output/test-binaries/`) |
| `compile_binary(src, output_name, subdir, arch=None)` | Compiles a `.cpp` source via `hipcc`; xdist-safe via file lock. `subdir` places the binary under `output/test-binaries/<subdir>/`. Use `cmake_build()` from `tests/common/_cmake_build.py` for `.hip` sources or multi-target CMake builds. |
| `ld_path` | `{"LD_LIBRARY_PATH": "..."}` dict for TheRock-linked binaries |
| `gpu_arch` | `str | None` from `--gpu-arch` CLI flag |

---

## How `target_executor` Works

When your test calls `target_executor.run("./my_binary")`, three things happen automatically:

```
test function
    │
    ▼
target_executor fixture
    │  reads hw.* / e2e.* markers + CLI flags
    │
    ├─ --no-gpu ──────────────────► DryRunExecutor    (synthetic, no subprocess)
    ├─ --container-mode ──────────► ContainerExecutor  (Docker/Podman + GPU passthrough)
    ├─ --remote-node host.yaml ───► SshExecutor        (SSH + ROCR_VISIBLE_DEVICES)
    └─ (default) ─────────────────► LocalExecutor      (subprocess + ROCR_VISIBLE_DEVICES)
            │
            ▼
    NodeExecutorGroup
        │  wraps 1 or N executors; uniform .run() interface
        ▼
    ExecutionResult
        .ok         → True if exit code 0
        .stdout     → captured output
        .stderr     → captured error output
        .exit_code  → raw integer
```

`ROCR_VISIBLE_DEVICES` is injected automatically — never set it in test code.
DryRun mode returns synthetic `"RESULT_OK\n"` for every command: it validates test
structure (markers, fixture wiring, collection), not functional correctness.

---

## Porting External Tests

Apply this transformation table to every external pattern:

| External pattern | rocm-tests equivalent | Notes |
|---|---|---|
| `subprocess.run(["./bin", "--check"])` | `result = target_executor.run("./bin --check")` | Drop `check=True`; use `result.ok` |
| `subprocess.Popen(cmd, ...)` | `target_executor.run(cmd)` | Executor handles Popen internally |
| `os.environ["ROCR_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Injected automatically |
| `os.environ["HIP_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Same |
| `result.returncode == 0` | `assert result.ok` | Plus a sentinel stdout check |
| `hipcc source.cpp -o binary` | `compile_binary()` in `conftest.py` | Session-scoped, xdist-safe |
| `.hip` source file | `cmake_build()` from `tests.common._cmake_build` | `hipcc` cannot handle `.hip` |
| `cmake && cmake --build` | `cmake_build()` in `conftest.py` | Preserve CMake; wrap in session fixture |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic |
| `time.sleep(N)` | **Remove entirely** | Health checks handle GPU readiness |
| `sys.exit(1)` on missing dep | `pytest.skip("reason")` | Graceful skip vs session abort |

For C++ gtest programs, keep `EXPECT_*`/`ASSERT_*` in the binary and assert `result.ok` +
`"PASSED" in result.stdout` from Python — do not translate gtest assertions to Python.

For `.hip` sources or multi-target CMake builds, use `cmake_build()` from
`tests.common._cmake_build` (not `compile_binary`). See `.claude/agents/porter.md` for
complete C++ gtest/CMake templates.

---

## AI Agent Commands

| Command | When to Use |
|---|---|
| `/creator` | Generate a complete, marker-compliant test from a GPU feature description |
| `/refiner <file>` | Four-persona review + flakiness detection + edge-case extension |
| `/porter <source-file>` | Port a shell script, raw Python, or non-compliant pytest into rocm-tests |

---

## Coding Standards

- **Style:** PEP 8, 120-character line length
- **Docstrings:** Google-style; concise intent description — no Args blocks restating type hints
- **Test naming:** Files match `test_*.py`; functions begin with `test_`; classes use `PascalCase`
- **Imports:** Standard Library → Third-Party → Local Framework (`framework.*`)
- **Secrets:** Never commit. Use `ROCM_TEST_*` environment variables.

**Lint commands:**

```bash
ruff check framework tests          # linting
black --check --diff framework tests # formatting
mypy framework --show-error-codes   # type checking
pylint framework --fail-under=9.5   # pylint

# Auto-fix
ruff check --fix framework tests && black framework tests
```
