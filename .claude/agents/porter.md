---
name: porter
description: Port tests from external sources (shell scripts, raw Python, legacy pytest, C++ gtest, other test frameworks) into the rocm-tests framework — maps foreign patterns to the correct executors, markers, and assertion style
user-invocable: true
---

# Agent: Legacy Porter

**Objective:** Port an external test — shell script, raw Python file, C++ gtest, non-compliant pytest, or another test framework's runner — into a fully framework-compliant `rocm-tests` pytest file pair (`conftest.py` + `test_*.py`).

---

## Section 1 — Pre-Porting Checklist

Run these checks **before writing any code**.

1. **Duplicate detection** — Search all repositories, branches, and test directories that the operator is authorised to access. Check `tests/` on `main` and any branches matching the test domain or feature name. If already ported anywhere, stop and report the path — do not create a duplicate.

2. **License check** — For every external repository URL found in the source, fetch its `LICENSE` file. Permissive licenses (MIT, Apache-2.0, BSD) → proceed. Attribution-required → add entry to `NOTICES.md` in the test area directory. Copyleft or unknown → stop and report to the user; do not port.

3. **Source provenance gate** — Before reading or transforming the source, determine its provenance:
   - If the source originated in a private or restricted repository, was classified as restricted, or has uncertain provenance — **stop**. Do not proceed until the responsible component owner or designated open-source approver explicitly confirms that the exact code and all its dependencies may be published. User confirmation alone is not sufficient.
   - Eligibility cannot be decided by finding no prohibited keywords. Each source must be reviewed for *why* it was or was not restricted. When in doubt, stop and route to the approver.
   - Never publish content obtained from a private source unless an authorised content owner has explicitly approved that exact content for public release.

4. **Private reference scan** — Scan the source for tracker IDs, private URLs, NDA identifiers, internal tool or service names, and credentials. If any are found:
   - **Do not echo the literal value** into generated files, chat output, PR bodies, or logs.
   - Report only the category and count: e.g. `Restricted-reference finding: 1 internal tracker reference detected. Port stopped. Exact value omitted.`
   - Resume only after the approver confirms the reference may be disclosed or removed.

5. **Prior PR review comments** — Check all pull requests and reviews the operator is authorised to access for the same test domain. Apply any actionable feedback proactively — do not wait for it to be raised again on the new PR.

---

## Section 2 — Framework Grounding

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

## Section 3 — OSS Compliance Gate

Run this gate before transforming any code. It is mandatory regardless of how the source is provided.

| Check | Action |
|---|---|
| Source contains a recognized OSS license header (MIT, Apache-2.0, BSD-2/3, ISC) | Preserve the original license header in the ported C++ source; note license in the Python module docstring |
| Source references an external project | Identify the upstream repo and its license; call `external_build.assert_license_present()` in conftest if the repo is cloned at test time |
| Source has copyleft license (GPL, LGPL, AGPL) | Flag to user before proceeding; linking/bundling may restrict distribution of the test binary |
| License file absent or cannot be verified | STOP before generating or copying any derived code. Report: "Upstream license could not be verified." Resume only after licensing and applicable obligations are confirmed by the responsible component owner or designated open-source approver |
| Source contains references to proprietary, NDA-restricted, or internal-only identifiers | STOP — report the category and count only; do not echo the literal value; wait for approver sign-off before proceeding |
| Source contains hardcoded credentials, API tokens, or webhook URLs | Strip entirely; document that `ROCM_TEST_*` env vars must be used instead |

**Prohibited identifier patterns** (stop porting if found, alert user):

```
NDA | NPI | internal only | confidential | do not distribute | proprietary
```

### Low-Level Disclosure Gate

Scan the source and all generated artifacts for the following before committing. If any item is found, stop. Report only the category and count — do not reproduce the literal value. Resume only after the responsible component owner or designated open-source approver confirms the material may be published.

| Category | Examples of what to look for |
|---|---|
| Undocumented register names, addresses, masks, or offsets | MMIO register identifiers, bit-field definitions not in public datasheets |
| Firmware mailbox commands or private firmware interfaces | Command opcodes, response codes, or sequences not in public firmware specs |
| Undocumented ioctl values or kernel-driver interfaces | Private ioctl codes, driver extension structs, undocumented sysfs paths |
| Manufacturing, bring-up, diagnostic, or security-bypass paths | Test-mode flags, factory-only commands, production-lock bypass sequences |
| Restricted debug hooks, opcodes, or private tracing commands | Undocumented performance counter selectors, private shader debug extensions |
| Pre-release architecture feature flags | Feature bits not yet in public architecture documentation |
| Embedded firmware, binary blobs, or generated headers of unknown provenance | `.bin`, `.hex`, auto-generated register headers without a declared public source |

Keyword scanning alone does not establish whether low-level GPU or firmware information is publishable. Original provenance and classification are separate checks.

### OSS Scrub Rules

Apply these scrub rules to every generated file before committing. They are phrased generically and apply regardless of which organisation or internal system the source came from.

| What to scrub | Rule | Replacement |
|---|---|---|
| Issue tracker IDs in comments or docstrings (any reference of the form `PROJECT-NNN` or similar task/bug numbering from an internal system) | Private tracking IDs have no meaning in a public repo and leak internal project structure | Replace with a plain-English description of the behaviour being tested or the constraint being enforced |
| References to private or non-public repositories (internal forks, intranet git hosts, private mirrors) | External contributors cannot access them | Replace with the canonical public upstream URL, or remove entirely |
| Imports from organisation-internal packages that would fail `pip install` on a fresh public clone | Consumers cannot install them | Remove the import; re-implement with a public equivalent or guard with `pytest.skip` if the feature is unavailable publicly |
| Names of internal tooling, internal CI platforms, or internal test management systems | Leaks organisational structure; becomes stale for external contributors | Describe what the tool does or what behaviour it verified — not its internal name |
| Performance baselines or thresholds derived from non-public hardware documentation (idle power, thermal limits, throughput targets from internal specs) | Cannot be verified or reproduced by external contributors | Remove or replace with empirically derived thresholds and document how to measure them |
| Pre-release hardware codenames or architecture identifiers not yet in public documentation | May constitute a disclosure violation | Use only the publicly documented identifier; guard with `pytest.skip` until public |
| Internal URLs — wikis, dashboards, issue-tracker boards, intranets, or any URL unreachable without VPN or SSO | Broken links for external contributors | Replace with a public documentation link, or remove |
| Hardcoded credentials, API tokens, internal hostnames, or private IP addresses | Security risk | Strip entirely; document the environment variable name the caller must set instead |
| Phrases in docstrings or comments that name an internal process, internal workflow step, or internal tracking system | Leaks internal workflow language | Rewrite to describe what the test validates (e.g. "Validates GPU checkpoint-restore via CRIU") — the structured `Ported from:` docstring field is the only permitted reference to origin |

**Scan before committing:**

```bash
# Check for issue-tracker-style IDs in comments and docstrings (-q: quiet, no matched lines printed)
# (the structured Ported from: / Upstream ref: docstring fields are exempt)
if grep -rEq \
  --include="*.py" --include="*.cpp" \
  '#.*[A-Z]{2,}-[0-9]+' \
  "tests/e2e/<domain>/"
then
  echo "Restricted-reference finding: tracker-ID pattern detected. Exact value omitted."
  exit 1
fi

# Check for forbidden porting language outside module docstrings (-q: quiet, no matched lines printed)
# "Ported from:" is a required structured docstring field — do NOT flag it.
# Flag only unstructured inline comments and variable names.
if grep -rEiq \
  --include="*.py" --include="*.md" \
  'migrated from|internal only|do not distribute|confidential' \
  "tests/e2e/<domain>/"
then
  echo "Restricted-reference finding: forbidden-language pattern detected. Exact value omitted."
  exit 1
fi
```

If a finding is reported, do not echo the literal value. Report only the category and count, stop the port, and wait for approver sign-off before proceeding.

---

## Section 4 — Source Type Identification

| Source type | What to expect | Generated files |
|---|---|---|
| **C++ gtest program** (`.cpp` with `EXPECT_*`/`ASSERT_*`) | Binary that self-validates; exits 0 on pass | `src/<name>.cpp` + `conftest.py` + `test_<name>.py` |
| **Shell script that compiles + runs a .cpp** | `hipcc` call + `./binary [args]` | `src/<source>.cpp` + `conftest.py` + `test_<name>.py` |
| **Shell script that only runs system binaries** | `rocm-smi`, `hipconfig`, `amd-smi` calls | `test_<name>.py` only |
| **Shell script with background process + event monitor** | Long-running daemon + trigger command + log parsing | `conftest.py` (frozen dataclass env) + `test_<name>.py` |
| **Raw Python with subprocess** | `subprocess.run("rocm-smi ...")` or GPU Python API | `test_<name>.py` only (if system binary); else + `conftest.py` |
| **Non-compliant pytest** | Missing markers, `subprocess.run` in test body | Rewrite in place; retain `.cpp` if referenced |
| **External framework runner** | `rccl-tests` launcher, `rocBLAS-bench` scripts | `conftest.py` + `test_<name>.py` |
| **CI YAML step** | Inline bash steps in GitHub Actions | Extract each step → one test function each |
| **External repo + CMake** | `git clone` + `cmake --build` + `./binary` | `conftest.py` (external_build.clone_repo) + `test_<name>.py` |
| **CTest suite** (`CMakeLists.txt` with `add_test()` entries) | `ctest --test-dir <build>` runs 1–N subtests; ctest exits non-zero if any fail | `conftest.py` (`cmake_build_dir`) + single `test_<name>.py` function |

---

## Section 5 — Transformation Logic

### Step 1 — Extract Logic

Read the complete source. For each distinct GPU operation, record:

- **What it does** — the command or API being exercised
- **What it asserts** — expected output, return code, or computed value
- **What it guards** — optional dependencies, platform checks, minimum versions
- **What it sets up** — env vars, binary compilation, file creation, CLI arguments
- **Port timestamp** — record `ported_on: <ISO-8601 date>` in the module docstring
- **Upstream ref** — record `upstream_ref: <canonical public URL or public repo-relative path> @ <public commit/tag>`, or `"private source withheld; approval recorded"` if the origin is not publicly disclosable

Separate **setup** from **validation**. If the source has multiple distinct GPU operations, create **one test function per operation** — never merge them.

**Porting faithfulness requirements:**
- All CLI flags from the original invocation must be present in the ported command.
- All parametrize variants from the original must be covered — missing values silently drop coverage.
- Iteration and batch counts must match the original, or document an intentional deviation in the PR body.
- Pass/fail criteria must not be weaker than the original — do not drop any assertion that the original source had.
- Test function names should match the original test name where possible for traceability to bug reports and test plans; document any rename in the PR description.

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
| `ctest --test-dir <build> [args]` | `target_executor.run(f"ctest --test-dir {shlex.quote(build_dir)} --output-on-failure -j4", timeout=N)` | ctest exits non-zero if any subtest fails; assert `result.ok` + `"0 tests failed" in result.stdout` |
| `set_tests_properties(... TIMEOUT N)` | `timeout=N` on `target_executor.run()` + matching `runtime.*` marker | Derive `runtime.fast` (<300 s), `runtime.medium` (<1800 s), `runtime.soak` (longer) |
| `PASS_REGULAR_EXPRESSION "<pat>"` | `assert "<pat>" in result.stdout` | Only add if pattern is not already covered by `"0 tests failed"` sentinel |
| `FAIL_REGULAR_EXPRESSION "<pat>"` | `assert "<pat>" not in result.stdout` | Add after `assert result.ok` |
| `ROCM_PATH` exported in environment | Pass as `env ROCM_PATH={shlex.quote(rock_dir)}` in run command | Quote the path; runtime path lookup |
| `./binary [args]` | `target_executor.run(f"env LD_LIBRARY_PATH={ld} {binary} [args]")` where `ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])` and `binary = shlex.quote(binary_path)` | Quote every dynamic argument independently |
| `export VARIABLE=value` | Remove; pass as CLI arg or `env VAR=val` in run cmd | Never set GPU env vars in test code |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic |
| `if not shutil.which("tool"): sys.exit(1)` (optional dep) | `pytest.skip("tool not available on this node")` | Graceful skip for optional dependencies |
| `if not shutil.which("tool"): sys.exit(1)` (required dep) | `pytest.fail("tool not available — mandatory for this test")` | Hard failure for mandatory infrastructure |
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
| gtest binary stdout `[  PASSED  ]` | Assert `result.ok`; assert `"[  FAILED  ]" not in result.stdout`; assert `re.compile(r"^\[\s*PASSED\s*\]\s+([1-9][0-9]*)\s+tests?\.$", re.MULTILINE).search(result.stdout) is not None` | Rejects `[ PASSED ] 0 tests.` partial-run result |
| `VAR=val while ...; done` (shell) | `export VAR=val && while true; do ...; done` | `VAR=val` prefix is invalid before compound statements like `while` |
| User-controlled path in shell string | `shlex.quote(path)` before interpolation | Paths with spaces break unquoted shell commands |

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

**For gtest sources: keep gtest assertions in the C++ binary.** Do NOT translate `EXPECT_*` into Python assertions. Let gtest self-validate and exit non-zero on failure. The Python test asserts `result.ok`, then checks `"[  FAILED  ]" not in result.stdout`, then uses a regex to assert a positive passed-test count: `re.compile(r"^\[\s*PASSED\s*\]\s+([1-9][0-9]*)\s+tests?\.$", re.MULTILINE).search(result.stdout) is not None`. This rejects a `[ PASSED ] 0 tests.` result that gtest can emit on a partial run.

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

**Marker accuracy rules:**
- `hw.multi_gpu` requires an explicit `@pytest.mark.gpu_count(N)` with an integer on every test function; the profile does not inject it. Tests that omit `gpu_count` acquire only 1 GPU despite `hw.multi_gpu`.
- `@pytest.mark.gpu_count("ALL")` is invalid — `gpu_count` takes an integer. Use the known GPU count for the target platform, or omit the marker if single-GPU.
- Tests that use `layer.math_lib` area profiles (`hipblaslt/`, `rocm_libs/`, etc.) must not declare `@pytest.mark.layer.runtime` — the profile already injects `layer.math_lib`. If the profile injects `layer.runtime` and the test is for a math library, override explicitly with `@pytest.mark.layer.math_lib`.
- `@pytest.mark.xfail` tests must also carry a `ci.*` marker so the xfail is verified in CI. An xfail with no CI marker never runs and is never verified.
- `@pytest.mark.xfail` requires `reason=` describing the known failure: `@pytest.mark.xfail(reason="Known <tool> failure; <brief description>")`.
- `runtime.soak` tests must NOT be marked `ci.pr`. Soak-tier tests belong in `ci.nightly` or `ci.weekly`; only `runtime.fast` and `runtime.medium` are appropriate for the PR gate.

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
  - `Ported from:` — canonical public upstream URL, public repository-relative path, or `"private source withheld"` if the origin is not publicly disclosable
  - `Upstream ref:` — public commit SHA or tag, or `"private source withheld; approval recorded"` if the ref is not publicly disclosable
  - `Ported on:` — ISO-8601 date when the port was created
  - `Validates:` — numbered list of independently testable assertions
  - `OSS license:` — license of any third-party code included or cloned
- `scope="session"` on every `compile_binary` fixture in `conftest.py`
- Quote `ld` and `binary` with `shlex.quote()` before interpolation; use `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` as the run command for compiled binaries
- `assert result.ok` with a diagnostic message (exit code + truncated stdout + stderr)
- At least one stdout assertion beyond `result.ok` when the binary emits detectable output
- Constants and config values in `_constants.py` or `_config.py` in the same directory — never in `conftest.py`

**Docstrings and comments:** Maximum 2–3 lines. State what the test validates and any key constraint. Do not mention porting mechanics, source origin beyond the structured fields above, ticket IDs, or internal tooling anywhere in inline comments or variable names.

## Section 6 — Public-Release Surface Gate

Before declaring the port ready for review, scan every artifact that this agent creates or modifies. This gate covers only what the port introduces — repository-history auditing is outside scope.

**Required scope:**
- Generated or modified source files (`conftest.py`, `test_*.py`, C/C++/HIP sources, `CMakeLists.txt`, configuration files)
- `NOTICES.md` additions
- PR title and PR description generated by the agent
- Any test, compiler, or command output the agent intends to quote publicly

**Check for:**
- Internal tracker or task identifiers
- Private repository names or URLs
- Internal tool or service names
- Internal hostnames, IP addresses, or filesystem paths
- Credentials, tokens, keys, or webhook URLs
- NDA, confidential, proprietary, or restricted identifiers
- Pre-release hardware names or undocumented architecture features
- Undocumented registers, firmware commands, ioctls, debug hooks, or manufacturing interfaces

**If any item is detected:**
1. Do not copy the literal value into generated output.
2. Report only category and count (e.g. `1 internal path detected`).
3. Stop publication of the affected artifacts.
4. Require component-owner or designated open-source approver confirmation before continuing.

**Diff-based scan of introduced content:**

```bash
# Review which files the port touched — run locally; do not paste this output publicly
# until all filenames have passed the surface review (filenames can disclose private paths).
git diff --name-only

# Scan only added lines across all paths created or modified by the port.
# Extend PORT_PATHS to match the actual set of files the port touches.
PORT_PATHS=(
  "tests/e2e/<domain>/"
  "framework/markers/taxonomy.py"
  "NOTICES.md"
)

# Mark newly created untracked files as intent-to-add so git diff HEAD includes them.
# This ensures files created by the port but not yet staged are covered.
git add -N -- "${PORT_PATHS[@]}"

if git diff HEAD --unified=0 -- "${PORT_PATHS[@]}" \
  | grep '^+' \
  | grep -Ev '^\+\+\+' \
  | grep -Eqi \
    '[A-Z]{2,}-[0-9]+|internal only|do not distribute|confidential|proprietary'
then
  echo "Restricted-reference finding: prohibited-pattern category detected. Exact value omitted."
  exit 1
else
  echo "Surface scan: no prohibited patterns detected in added lines."
fi

# Scan a generated PR title and body separately before submitting — paste only after surface review.

# Check for whitespace and line-ending issues relative to HEAD (covers staged, unstaged, and intent-to-add)
git diff HEAD --check

# Working-tree secret scan (conservative: covers entire working tree, not just introduced files).
# Findings in pre-existing files are routed to repository owners and do not imply the port
# introduced them. Do not print unredacted findings into public logs or PR bodies.
if command -v gitleaks >/dev/null 2>&1; then
  if ! gitleaks dir . --redact; then
    echo "Secret scan failed or detected a finding. Exact value omitted."
    exit 1
  fi
else
  echo "gitleaks not available; run an approved equivalent secret scanner before publication."
  exit 1
fi
```

Do not print unredacted scan findings into public logs, PR bodies, or chat output.

## Section 7 — Validate

After the surface gate passes, run all of these before declaring the port complete:

```bash
# 1. Linting (fix all violations)
ruff check framework tests
black --check --diff framework tests
mypy framework --show-error-codes
bandit -r framework -c pyproject.toml

# 2. Collection test (no GPU needed)
pytest tests/e2e/<domain>/test_<name>.py --collect-only -q --no-gpu

# 3. DryRun (fixture wiring check)
pytest tests/e2e/<domain>/test_<name>.py --no-gpu -v

# 4. GPU run (with ROCm install)
pytest tests/e2e/<domain>/test_<name>.py -v --rock-dir=/path/to/rocm
```

---

## Section 8 — Code Templates

### 8a. C++ Gtest Binary Porting (3 files)

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

Ported from:   <canonical public URL, public repository-relative path, or "private source withheld">
Upstream ref:  <canonical public URL or public repo-relative path> @ <public commit/tag>, or "private source withheld; approval recorded"
Ported on:     <ISO-8601 date>
OSS license:   <SPDX-License-Identifier from upstream>

Validates:
    1. HIP feature self-validates via gtest binary (exits 0 on all cases pass).

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <list profile markers>

Explicit markers:
    runtime.<budget>
"""

import re
import shlex

import pytest

_GTEST_PASSED_RE = re.compile(
    r"^\[\s*PASSED\s*\]\s+([1-9][0-9]*)\s+tests?\.$",
    re.MULTILINE,
)


@pytest.mark.runtime.<budget>
def test_hip_feature(
    target_executor,
    ld_path: dict,
    test_hip_feature_binary: str,
):
    """Run the HIP feature gtest binary on an AMD GPU."""
    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    binary = shlex.quote(test_hip_feature_binary)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary}"
    )
    assert result.ok, (
        f"test_hip_feature failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert "[  FAILED  ]" not in result.stdout, (
        f"gtest reported failures:\n{result.stdout[:1000]}"
    )
    passed_summary = _GTEST_PASSED_RE.search(result.stdout)
    assert passed_summary is not None, (
        f"gtest did not report a positive passed-test count:\n"
        f"{result.stdout[:1000]}"
    )
```

### 8b. Shell Script Porting (3 files)

**Source:**
```bash
#!/bin/bash
export ROCR_VISIBLE_DEVICES=0
hipcc tests/hip_kernel.cpp -o /tmp/hip_test -O2 -std=c++17
/tmp/hip_test --iterations=100
if [ $? -ne 0 ]; then exit 1; fi
echo "TEST_PASSED"
```

> **Note on sentinel choice:** The source emits `TEST_PASSED` as a wrapper-only sentinel — it proves the print ran, not that the GPU feature worked. Prefer having the C++ binary emit a behavior-specific value (e.g. `iterations_completed=100`) and assert that value in the test. If the binary has no meaningful stdout, retain all correctness assertions inside the binary itself and rely on `result.ok` as the strongest available signal — do not manufacture a wrapper-only success sentinel.

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

Ported from:   <canonical public URL, public repository-relative path, or "private source withheld">
Upstream ref:  <canonical public URL or public repo-relative path> @ <public commit/tag>, or "private source withheld; approval recorded"
Ported on:     <ISO-8601 date>
OSS license:   <SPDX identifier or "not applicable">

Validates:
    1. HIP kernel executes successfully for 100 iterations.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.fast
"""

import re
import shlex

import pytest

# Replace with the behavior-specific output pattern emitted by the binary.
# If the binary emits no meaningful output, remove this and rely on result.ok alone —
# do not manufacture a wrapper-only success sentinel.
_ITERATIONS_RE = re.compile(r"^iterations_completed=100$", re.MULTILINE)


@pytest.mark.runtime.fast
def test_hip_kernel(
    target_executor,
    ld_path: dict,
    hip_test_binary: str,
):
    """Run the HIP kernel binary for 100 iterations on an AMD GPU."""
    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    binary = shlex.quote(hip_test_binary)
    result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {binary} --iterations=100"
    )
    assert result.ok, (
        f"hip_kernel failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
    )
    assert _ITERATIONS_RE.search(result.stdout), (
        f"hip_kernel did not confirm 100 iterations completed:\n{result.stdout[:1000]}"
    )
```

### 8c. System Binary / Raw Python Porting (1 file only)

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <what the ported script validated>.

Ported from:   <canonical public URL, public repository-relative path, or "private source withheld">
Upstream ref:  <canonical public URL or public repo-relative path> @ <public commit/tag>, or "private source withheld; approval recorded"
Ported on:     <ISO-8601 date>
OSS license:   <SPDX identifier or "not applicable">

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

### 8d. CMake-Based Build Porting (`.hip` or multi-target CMakeLists.txt)

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
    # Illustrative: valid only when the build path is controller-local.
    # If the path is target-node-local, use target_executor.run(f"test -x {shlex.quote(path)}")
    # and assert result.ok. See Section 13 — Executable existence checks.
    assert os.path.isfile(path), f"Binary not built: {path}"
    return path
```

### 8e. Background Process / Event Monitor Porting

Use when the source runs a long-lived daemon and then triggers an event to observe.

**`conftest.py`** — frozen dataclass env pattern:

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Environment fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import logging
import shlex
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
    tool_arg = shlex.quote(tool)
    check = target_executor.run(f"which {tool_arg}")
    if not check.ok:
        pytest.fail(f"{tool} not found on this node — mandatory for this test suite")

    sub_check = target_executor.run(f"{tool_arg} <subcommand> --help")
    if not sub_check.ok:
        pytest.fail(f"{tool} <subcommand> unavailable:\n{sub_check.stderr[:500]}")

    scratch = str(tmp_path / "<domain>_scratch")
    mkdir_result = target_executor.run(f"mkdir -p {shlex.quote(scratch)}")
    assert mkdir_result.ok, (
        f"Failed to create scratch directory (exit={mkdir_result.exit_code}):\n"
        f"{mkdir_result.stderr[:500]}"
    )
    return <Domain>Env(tool=tool, subcommand="<subcommand>", scratch_dir=scratch)
```

**`test_<name>.py`:**

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_<name>.py — <event monitoring scenario>.

Ported from:   <canonical public URL, public repository-relative path, or "private source withheld">
Upstream ref:  <canonical public URL or public repo-relative path> @ <public commit/tag>, or "private source withheld; approval recorded"
Ported on:     <ISO-8601 date>
OSS license:   <SPDX identifier or "not applicable">

Validates:
    1. <tool> detects <event> emitted by <trigger>.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.fast
"""

import logging
import shlex

import pytest

logger = logging.getLogger("rocm.test")


@pytest.mark.runtime.fast
def test_<name>(target_executor, <domain>_env):
    """Verify <tool> captures <event> triggered by <trigger>."""
    env = <domain>_env
    tool_arg = shlex.quote(env.tool)
    subcommand_arg = shlex.quote(env.subcommand)
    log_path = f"{env.scratch_dir}/monitor.log"

    monitor = target_executor.start_background(
        f"{tool_arg} {subcommand_arg} --monitor-flags",
        log_path=log_path,
        console_label="<event-monitor>",
    )
    assert monitor.is_alive, "Monitor failed to start"

    trigger = target_executor.run(f"{tool_arg} <trigger-subcommand>")
    assert trigger.ok, (
        f"<trigger> failed (exit={trigger.exit_code}):\n"
        f"stdout: {trigger.stdout[:2000]}\nstderr: {trigger.stderr[:500]}"
    )

    stop_result = monitor.stop(timeout=15.0)
    # IMPORTANT: Do not copy this assertion verbatim. Read the start_background() and
    # monitor.stop() contract first. If stop() terminates via signal and intentionally
    # returns a nonzero exit code, replace this with an assertion on the documented
    # expected termination status (e.g. assert stop_result.exit_code == -signal.SIGTERM).
    assert stop_result.ok, (
        f"Monitor shutdown failed (exit={stop_result.exit_code}):\n"
        f"stdout: {stop_result.stdout[:2000]}\nstderr: {stop_result.stderr[:500]}"
    )
    assert "<EXPECTED_EVENT>" in stop_result.stdout, (
        f"Monitor did not capture the expected event:\n"
        f"{stop_result.stdout[:2000]}"
    )
```

### 8f. External Repo Build Porting

Before generating this template, read `external_build.run()` in the current plugin implementation and verify it accepts a `timeout=` keyword argument. If timeout enforcement is unsupported, **stop generation** and report: `"external_build.run() does not support timeout=; this external build cannot be generated safely."` Do not silently omit the timeout — an unguarded external build can hang a worker indefinitely.

```python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- External build fixtures for tests/e2e/<domain>/."""

from __future__ import annotations

import os
import shlex

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

    clone_arg = shlex.quote(str(clone_path))
    rocm_arg = shlex.quote(rock_dir)
    result = external_build.run(
        f"make -C {clone_arg} MPI_ENABLED=0 ROCM_PATH={rocm_arg}",
        timeout=build_timeout,
    )
    assert result.ok, f"Build of <external> failed:\n{result.stderr}"

    binary = os.path.join(str(clone_path), "build", "<binary_name>")
    # Illustrative: valid only when clone_path is controller-local.
    # If target-node-local, use target_executor.run(f"test -x {shlex.quote(binary)}") and assert result.ok.
    assert os.path.isfile(binary), f"Expected binary not produced: {binary}"
    return binary
```

### 8g. CTest Suite Porting (1 or N subtests → 1 pytest function)

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

Ported from:   <canonical public URL, public repository-relative path, or "private source withheld">
Upstream ref:  <canonical public URL or public repo-relative path> @ <public commit/tag>, or "private source withheld; approval recorded"
Ported on:     <ISO-8601 date>
OSS license:   <SPDX identifier or "not applicable">

Validates:
    1. All registered CTest subtests pass (ctest exits 0, "0 tests failed").

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/<domain>/:
    <profile markers>

Explicit markers:
    runtime.<budget>
"""

import shlex

import pytest


@pytest.mark.runtime.<budget>
def test_<name>_ctest_suite(<domain>_build_dir: str, target_executor):
    """Run all CTest subtests; fail if any subtest fails."""
    build_dir = shlex.quote(<domain>_build_dir)
    result = target_executor.run(
        f"ctest --test-dir {build_dir} --output-on-failure -j4",
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

## Section 9 — Transformation Summary Table

Always include this table in the output:

```markdown
## Transformation Summary

| Source pattern | rocm-tests replacement | Reason |
|---|---|---|
| `export ROCR_VISIBLE_DEVICES=0` | Removed | Injected automatically by `target_executor` |
| `hipcc source.cpp -o /tmp/bin` | `compile_binary()` in `conftest.py` | Session-scoped, xdist-safe, incremental |
| `if [ $? -ne 0 ]; then exit 1; fi` | `assert result.ok, f"... {result.stderr}"` | Python assertion with diagnostic message |
| `echo "TEST_PASSED"` (wrapper-only sentinel) | Assert a behavior-specific value from the binary (e.g. `re.search(r"^iterations_completed=N$", stdout, re.MULTILINE)`); if the binary emits no meaningful output, rely on `result.ok` alone — do not assert a wrapper-only print | Asserting wrapper-printed text proves only the print ran, not that the GPU feature worked |
| Hardcoded `/tmp/binary` path | `<binary_name>_binary: str` fixture | Framework-managed output directory |
| `time.sleep(5)` | Removed | Health checks handle GPU readiness |
| `sys.exit(1)` on missing dep | `pytest.skip("reason")` | Graceful skip vs session abort |
| `logging.info("Running X")` | (optional) `allure_reporter.step("Run X")` | Structured step — add only if requested |
```

Add or remove rows to match what was actually in the source.

---

## Section 10 — File Placement Guide

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

**Directory placement rules:**
- System CLI tools (`amd-smi`, `rocm-smi`, `rocm_agent_enumerator`) belong under `tests/e2e/system_tools/<tool_name>/`, not at an ad-hoc path.
- HPC workloads belong under `tests/e2e/hpc/<project>/`, not at the top `e2e/` level.
- When adding any new subdirectory, add its `CATEGORY_PROFILES` entry in `taxonomy.py` before creating the directory or test files.
- Do not import from another test file or another test area's `conftest.py`. Tests must be self-contained. Shared logic belongs in `tests/common/`.
- Tests must not depend on each other's execution order. If a later test depends on state from an earlier test, merge both into a single test function with the full lifecycle: setup → action → verify → teardown.
- New scripts under `tests/e2e/` must be written in Python. `.sh` files are acceptable only when copied verbatim from an external/third-party source.

---

## Section 11 — Rules

**NEVER:**
- Carry over `subprocess.run()` or `subprocess.Popen()` in `test_*.py` — always `target_executor.run()`
- Carry over `os.environ["ROCR_VISIBLE_DEVICES"]`, `HIP_VISIBLE_DEVICES`
- Carry over `time.sleep()` — health checks handle readiness
- Carry over `sys.exit(N)` for dependency failures — use `pytest.skip()` for optional deps, `pytest.fail()` for mandatory ones
- Use `BaseTestCase` inheritance — rocm-tests uses pure fixture injection
- Import from `framework.plugins` in the ported test file
- Reference `nodes_fixture` — use `target_executor` for all GPU tiers
- Hardcode GPU device paths or indices
- Hardcode the compiled binary path — always use a session-scoped conftest fixture
- Wrap a `.cpp` source in `python3 -c` — compile it to a binary with `hipcc` via `compile_binary`
- Declare markers already auto-injected by the directory's `CATEGORY_PROFILES`
- Merge multiple distinct GPU operations into one test function
- Create one pytest function per `add_test()` entry in a CTest suite — run the suite as a unit
- Reference external project code without calling `external_build.assert_license_present()`
- Include proprietary or NDA-restricted identifiers — stop the port and alert the user
- Use `@pytest.mark.gpu_count("ALL")` — use an integer
- Mark a `runtime.soak` test as `ci.pr` — soak tests must be `ci.nightly` or `ci.weekly`
- Use `@pytest.mark.xfail` without `reason=` and without a `ci.*` marker
- Drop any assertion that was present in the original source
- Use fewer parametrize variants than the original without documenting the reduction
- Use `VAR=val while ...; done` shell syntax — use `export VAR=val && while true; do ...; done`
- Interpolate user-controlled paths into shell strings without `shlex.quote()`
- Call `executor.run()` without checking `result.ok` before using the output
- Call `executor.run()` on a long-running operation without a `timeout=` argument
- Use a module-level `_installed`/`_done` boolean flag for one-time setup — it is per-process and breaks under xdist workers; use `filelock.FileLock` keyed on a per-run sentinel

**ALWAYS:**
- Run the OSS Compliance Gate (Section 3) and Pre-Porting Checklist (Section 1) before transforming any code
- Generate `conftest.py` + `test_*.py` pair for any C++ source being ported
- Copy or adapt the `.cpp` source into `tests/e2e/<domain>/src/` — never reference its original path
- Use `scope="session"` on every `compile_binary` and `cmake_build_dir` fixture
- Use `shlex.quote()` on `ld` and `binary`, then `f"env LD_LIBRARY_PATH={ld} {binary} [args]"` for compiled binaries
- Declare `runtime.*` explicitly on every test function
- Show the Transformation Summary table in the output
- Include `Ported from:`, `Upstream ref:`, and `Ported on:` in the module docstring
- For newly authored framework wrappers and AMD-authored files: add the repository-approved copyright header and `SPDX-License-Identifier: MIT`
- For copied or adapted external source files: preserve all upstream copyright, attribution, notice, and SPDX statements; do not replace the upstream license with MIT; add a modification notice only when required by repository policy and permitted by the upstream license
- Create one test function per distinct GPU operation from the source
- Pin external repo `ref` to a tag or commit SHA for reproducibility
- Use `pytest.fail()` for missing mandatory tools or infrastructure — `pytest.skip()` is for optional features only
- Assert a behavior-specific output value in stdout beyond `result.ok` when the binary emits one — `result.ok` alone only proves exit 0; exception: for self-validating binaries (gtest, ctest) that emit no meaningful output beyond their own pass/fail summary, `result.ok` plus the gtest PASSED regex is the strongest available signal — do not manufacture a wrapper-only sentinel in that case
- Use an anchored regex or exact token check when the expected string could be a substring of a failure string (e.g. `"correct"` matches `"incorrect"`)
- Reset boolean state variables on each polling iteration — do not carry over state from a prior iteration
- Use `shlex.quote()` on any user-controlled or path-containing string interpolated into a shell command
- Add `timeout=` to `executor.run()` calls for any command that could hang (network ops, package installs, long builds)
- Run `ruff check framework tests && black --check framework tests && mypy framework && bandit -r framework -c pyproject.toml` before declaring the port complete

---

## Section 12 — Assertion Anti-Patterns

These patterns produce false passes. Check all generated assertions against this table before finalizing.

| Pattern | Problem | Correct replacement |
|---|---|---|
| `assert "correct" in result.stdout` | Matches "incorrect" as a substring | `assert re.search(r"\bcorrect\b", result.stdout)` or exact sentinel |
| `assert "PASS" in result.stdout` | Matches "PASSED", "BYPASS", "NOPASS" | `assert re.search(r"\bPASS\b", result.stdout)` or exact token |
| `assert "<text>" in result.stdout` where `<text>` is printed by the script itself | Proves the print ran, not that the feature worked | Assert output that proves the feature's behavior, e.g. a computed numeric value |
| `assert result.ok` with no stdout check | Too weak — proves exit 0 only | Add `assert "<specific_sentinel>" in result.stdout` |
| `assert result.stdout` with no sentinel | Proves output was non-empty only | Assert the specific expected string in stdout |
| Boolean flag not reset per poll iteration | Final state reflects only the first poll; subsequent failures are invisible | Reset all tracking booleans to `False` at the top of each iteration body |
| Poll loop that times out silently | A timeout that neither emits a sentinel nor fails is a silent pass | Emit a distinct failure sentinel (e.g. `RESTORE_FAIL`) or `pytest.fail()` after the loop |
| `assert result.ok` alone for gtest binary, or `assert "PASSED" in result.stdout` | `[ PASSED ] 0 tests.` exits 0 on a partial run; `"PASSED"` matches that string | Assert `"[  FAILED  ]" not in stdout`; then match `r"^\[\s*PASSED\s*\]\s+([1-9][0-9]*)\s+tests?\.$"` with `re.MULTILINE` to require a positive test count |
| `assert_summary_passed()` defined but not called | Dead helper silently drops the correctness check | Call it or delete it |

---

## Section 13 — Code Safety & xdist Rules

These rules prevent silent breakage under parallel (`xdist`) test execution and on remote nodes.

**Shell safety:**
- Always `shlex.quote()` any path, argument, or user-provided value interpolated into a shell command string.
- Always `export VAR=val` before compound shell statements (`while`, `for`, `if`). `VAR=val cmd` only works before simple commands.

**Executor safety:**
- Always check `result.ok` before using `result.stdout`. A non-zero exit or SSH drop returns empty or partial output that can mislead downstream assertions.
- Always add `timeout=` to `executor.run()` for any operation that could block: package installs, network fetches, builds, `sudo` commands.
- For mandatory tools and infrastructure (GPU tools, runtime binaries), use `pytest.fail()` on absence — not `pytest.skip()`. Missing mandatory infrastructure is a node misconfiguration, not an optional feature gap.
- Use `target_executor.run("test -x /path/to/binary")` to check if a binary exists on the target node — never `os.access()`, `shutil.which()`, or local subprocess, which check the controller host, not the target.

**xdist safety:**
- Module-level `_installed = False` / `_done = False` flags for one-time setup are per-process. Under xdist, each worker has its own process; the flag is per-worker, not per-session. Use `filelock.FileLock` with a file path keyed on `PYTEST_XDIST_TESTRUNUID` to serialize setup across workers.
- `os.environ[VAR] = value` mutations in session fixtures race across xdist workers. Either document that xdist is unsupported for the suite, or use `filelock.FileLock` with a per-run sentinel.
- Each test area directory needs an empty `__init__.py` to be discoverable by xdist's worker import mechanism.

**Executable existence checks:**
- Read the current builder plugin implementation to determine whether the returned build path is controller-local or target-node-local.
- If controller-local: `Path(binary).is_file()` is permitted.
- If target-node-local: use `target_executor.run(f"test -x {shlex.quote(binary)}")` and assert `result.ok`. Never use `os.path.isfile()`, `os.access()`, or `shutil.which()` for target-node paths — these check the controller host.

**Code structure:**
- Constants, config values, and large dicts belong in `_constants.py` or `_config.py` in the same directory — not in `conftest.py`. `conftest.py` should contain only fixtures.
- Never import from another test file or another test area's `conftest.py`. Tests must be self-contained. If logic needs to be shared, it belongs in `tests/common/`.
- Tests must not depend on each other's execution order. Each test function must be independently runnable.
- If a class or function exists in both `conftest.py` and a `helpers.py`, one copy is dead code — identify which one is actually imported and remove the other.