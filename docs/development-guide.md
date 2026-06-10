# Development Guide — Writing and Contributing Test Cases

This guide walks you through everything you need to add a new test case to `rocm-tests`,
from environment setup to opening a pull request.

---

## 1. Set Up Your Environment

```bash
git clone https://github.com/ROCm/rocm-tests.git
cd rocm-tests
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Faster install with uv (optional)
uv pip install -r requirements-dev.txt
```

Verify everything is wired up before writing any code:

```bash
# Collect and lint all tests (no GPU required)
pytest tests/ --collect-only -q --no-gpu

# Run the sanity tests
pytest tests/ -m "ci.nightly" --no-gpu -v
```

---

## 2. Understand the Marker System

Every test function **must** carry at least one marker from each required dimension. Markers
control where and when a test runs and power smart GPU scheduling.

| Dimension | Required | Values |
|---|---|---|
| `hw.*` | YES | `gpu`, `multi_gpu`, `cpu_only` |
| `ci.*` | YES | `pr`, `nightly`, `weekly` |
| `layer.*` | YES | `runtime`, `math_lib` |
| `runtime.*` | no¹ | `fast` (<5 min), `medium` (<30 min), `soak` (hours) |
| `os.*` | no | `linux` |

¹ Not linter-enforced but **always declare it** — omitting it disables smart-sharding runtime weights.

**Authoritative source:** `framework/markers/taxonomy.py → MARKER_SCHEMA`. Never add new
marker values only in test files — add them to `MARKER_SCHEMA` first.

**Dotted syntax** (`@pytest.mark.ci.nightly`) is enabled by a `MarkDecorator.__getattr__` patch
in `conftest.py`. IDE linters may flag it — this is expected.

**Parametric markers** (not dimension-enforced):

- `@pytest.mark.gpu_vram(16)` — minimum VRAM in GB for GPU allocation
- `@pytest.mark.gpu_count(4)` — number of GPUs to acquire
- `@pytest.mark.container_image("rocm/pytorch:6.3")` — per-test container image override

**Minimum valid test:**

```python
@pytest.mark.ci.nightly
@pytest.mark.hw.cpu_only
@pytest.mark.runtime.fast
def test_example(dry_run_executor):
    result = dry_run_executor.run("echo RESULT_OK")
    assert result.ok
    assert "RESULT_OK" in result.stdout
```

---

## 3. Choose Where to Place Your Test

### Adding to an existing test area (most common)

If a `tests/e2e/<domain>/` directory already exists and has a `CATEGORY_PROFILES` entry in
`framework/markers/taxonomy.py`, drop your `test_*.py` file there. Required dimension
markers (`hw.*`, `ci.*`, `layer.*`) are injected automatically — you only need to declare
`@pytest.mark.runtime.<value>` explicitly. You can still override any auto-injected marker
by declaring it on your function.

Example auto-injected profiles:

| Directory | Auto-injected markers |
|---|---|
| `tests/e2e/hip_runtime/` | `hw.gpu`, `layer.runtime`, `ci.nightly`, `os.linux` |
| `tests/e2e/rocprim/` | `hw.gpu`, `layer.math_lib`, `ci.nightly`, `os.linux` |

### Creating a new test area

1. **Register the directory profile** in `framework/markers/taxonomy.py → CATEGORY_PROFILES`:

   ```python
   "tests/e2e/my_feature": [
       "hw.gpu", "layer.runtime", "ci.nightly", "os.linux",
   ],
   ```

2. **Place your test file** at `tests/e2e/my_feature/test_my_feature.py`.
   Add any new marker values to `MARKER_SCHEMA` in the same `taxonomy.py` if needed.

3. **Add a build fixture** if your tests run compiled binaries. Create
   `tests/e2e/my_feature/conftest.py`:

   ```python
   # For a single .cpp file — use compile_binary directly
   import pytest

   @pytest.fixture(scope="session")
   def my_binary(compile_binary):
       return compile_binary(
           src="tests/e2e/my_feature/src/test.cpp",
           output_name="test_binary",
           subdir="my_feature",   # → output/test-binaries/my_feature/test_binary
       )
   ```

   For `.hip` sources or multi-target CMake builds, use `cmake_build()` from
   `tests.common._cmake_build`:

   ```python
   import os
   import pytest
   from tests.common._cmake_build import cmake_build

   @pytest.fixture(scope="session")
   def my_binary(rock_dir, gpu_arch, compiler_build_dir):
       build_dir = cmake_build(
           src="tests/e2e/my_feature/src",
           build_dir=os.path.join(compiler_build_dir, "my_feature"),
           rocm_path=rock_dir,
           gpu_arch=gpu_arch,
       )
       return os.path.join(build_dir, "my_binary")
   ```

   Tests that only call `target_executor.run()` on pre-installed system binaries need
   **no** `conftest.py` — all framework fixtures are available automatically.

---

## 4. Write Your Test

Use `target_executor` for all GPU tests. The fixture automatically picks the right backend
(`LocalExecutor`, `SshExecutor`, `ContainerExecutor`, or `DryRunExecutor`) based on CLI
flags and markers — you never touch `ROCR_VISIBLE_DEVICES`.

```
test function
    │
    ▼
target_executor fixture
    │  reads hw.* / e2e.* markers + CLI flags
    │
    ├─ --no-gpu ──────────────────► DryRunExecutor    (synthetic; no subprocess)
    ├─ --container-mode ──────────► ContainerExecutor  (Docker/Podman + GPU passthrough)
    ├─ --remote-node host.yaml ───► SshExecutor        (SSH + ROCR_VISIBLE_DEVICES)
    └─ (default) ─────────────────► LocalExecutor      (subprocess + ROCR_VISIBLE_DEVICES)
            │
            ▼
    NodeExecutorGroup
        .run(cmd)  →  ExecutionResult
                          .ok         → True if exit code 0
                          .stdout     → captured output
                          .stderr     → captured error output
                          .exit_code  → raw integer
```

**Single GPU test example:**

```python
@pytest.mark.runtime.fast
def test_hip_device_query(target_executor, my_binary):
    result = target_executor.run(f"{my_binary} --device-query")
    assert result.ok, f"device query failed:\n{result.stderr}"
    assert "Device count" in result.stdout
```

**Multi-GPU test example:**

```python
@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count(2)
@pytest.mark.runtime.medium
def test_allreduce_two_gpus(target_executor, my_binary):
    result = target_executor.run(f"{my_binary} --op allreduce --count 2")
    assert result.ok, result.stderr
```

**Fixture decision guide:**

| Markers on test | `target_executor` yields | Test code |
|---|---|---|
| `hw.gpu` | `NodeExecutorGroup(1 exec)` | `target_executor.run(cmd)` |
| `hw.multi_gpu` + `gpu_count(N)` | `NodeExecutorGroup(1 exec, ROCR=0,1,...)` | `target_executor.run(cmd)` |
| `e2e.multinode` + `gpu_count(N)` | `NodeExecutorGroup(N execs, 1 per node)` | `for e in target_executor: e.run(cmd)` |
| `--no-gpu` (any) | `NodeExecutorGroup(DryRunExecutor)` | `target_executor.run(cmd)` |

**Never set `ROCR_VISIBLE_DEVICES` in test code** — always go through `target_executor`.

---

## 5. Validate Your Test Locally

```bash
# Check markers and collection (no GPU required)
pytest tests/e2e/my_feature/ --collect-only -q --no-gpu

# Run against real GPU
pytest tests/e2e/my_feature/ -m "hw.gpu and ci.nightly" -v

# Full nightly matrix for a specific architecture
pytest tests/e2e/my_feature/ -m "hw.gpu and ci.nightly" --gpu-arch gfx942 -v
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

## 6. Key Fixtures Reference

Framework fixtures are loaded automatically via `conftest.py → pytest_plugins`. Declare them
as function parameters — no imports needed.

**Executors:**

| Fixture | Purpose |
|---|---|
| `target_executor` | **Primary GPU fixture.** Yields `NodeExecutorGroup`. Dispatches automatically based on markers and CLI flags. |
| `dry_run_executor` | Synthetic stub, never shells out. For `hw.cpu_only` and PR gate tests. |
| `cpu_executor` | Real subprocess, no GPU env. For `hw.cpu_only` tests needing real commands. |
| `container_executor` | Docker/Podman with AMD GPU passthrough. Use `probe()`/`exec_in()` directly. |

**Builder (session-scoped):**

| Fixture | Purpose |
|---|---|
| `rock_dir` | Path to ROCm install (`--rock-dir`, `ROCK_DIR`, or `rocm-test.toml`) |
| `compiler_build_dir` | Binary output dir (default `output/test-binaries/`) |
| `compile_binary(src, output_name, subdir, arch=None)` | Compiles `.cpp` via `hipcc`; xdist-safe. Use `cmake_build()` for `.hip` or multi-target builds. |
| `ld_path` | `{"LD_LIBRARY_PATH": "..."}` for TheRock-linked binaries |
| `gpu_arch` | `str \| None` from `--gpu-arch` |

**Framework / session:**

| Fixture | Purpose |
|---|---|
| `framework_config` | Merged `FrameworkConfig` |
| `run_ctx` | Unique run ID + timestamp |
| `os_adapter` | GPU device enumeration and kernel module interface |
| `platform_name` | `"linux"` / `"windows"` |

---

## 7. AI-Assisted Test Authoring

Claude Code skills are available to accelerate test creation and review:

| Command | When to Use |
|---|---|
| `/creator` | Generate a complete, marker-compliant test from a GPU feature description |
| `/refiner <file>` | Four-persona review + flakiness detection + edge-case extension |
| `/porter <source-file>` | Port a shell script, raw Python, or non-compliant pytest into rocm-tests |

**Typical workflow:**

```bash
/creator                   # describe the feature; agent generates a compliant file
pytest tests/e2e/... --collect-only -q --no-gpu   # validate collection
/refiner tests/e2e/...     # four-persona review before opening a PR
```

---

## 8. Porting Existing Tests

Apply this transformation table when porting external test scripts:

| External pattern | rocm-tests equivalent | Notes |
|---|---|---|
| `subprocess.run(["./bin", "--check"])` | `result = target_executor.run("./bin --check")` | Drop `check=True`; use `result.ok` |
| `os.environ["ROCR_VISIBLE_DEVICES"] = "0"` | **Remove entirely** | Injected automatically |
| `result.returncode == 0` | `assert result.ok` | Plus a sentinel stdout check |
| `hipcc source.cpp -o binary` | `compile_binary()` in `conftest.py` | Session-scoped, xdist-safe |
| `.hip` source file | `cmake_build()` from `tests.common._cmake_build` | `hipcc` cannot handle `.hip` |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic |
| `time.sleep(N)` | **Remove entirely** | Health checks handle GPU readiness |
| `sys.exit(1)` on missing dep | `pytest.skip("reason")` | Graceful skip vs session abort |

For C++ gtest programs, keep `EXPECT_*`/`ASSERT_*` in the binary and assert `result.ok` +
`"PASSED" in result.stdout` from Python — do not translate gtest assertions to Python.

---

## 9. Coding Standards

- **Style:** PEP 8, 120-character line length
- **Docstrings:** Google-style; concise intent — no Args blocks restating type hints
- **Test naming:** Files match `test_*.py`; functions begin with `test_`; classes use `PascalCase`
- **Imports:** Standard Library → Third-Party → Local Framework (`framework.*`)
- **Secrets:** Never commit. Use `ROCM_TEST_*` environment variables. See [SECURITY.md](../SECURITY.md).

**Lint (must pass before merge — run by `pre-commit.yml` CI):**

```bash
black --check --diff framework tests   # formatting
ruff check framework tests             # linting
mypy framework --show-error-codes      # type checking
pylint framework --fail-under=9.5      # quality score

# Auto-fix locally
ruff check --fix framework tests && black framework tests
```

**Security (run locally before opening a PR):**

```bash
bandit -r framework -c pyproject.toml   # Python SAST
pip-audit -r requirements.txt           # CVE scan of pinned dependencies
```

CodeQL (`security-extended` query suite) runs automatically on every PR to `main` via
`.github/workflows/codeql.yml` — no local setup required.
