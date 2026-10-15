# OS Package Dependencies — Developer Guide

`os_deps_plugin` lets test authors declare OS package dependencies directly on
test functions and classes. The framework installs them automatically before the
test runs, using the native package manager of the target node — without any
manual intervention or environment-specific configuration.

---

## How It Works

```
Test declares @OS_DEPS("cmake", "libnuma-dev")
         │
         ▼
Framework reads declaration before test body runs
         │
         ▼
Probes node: are these packages already installed?
         │
    ┌────┴────┐
   YES       NO
    │         │
    │         ▼
    │    Detects package manager (apt / dnf / zypper / ...)
    │         │
    │         ▼
    │    Installs missing packages
    │         │
    └────┬────┘
         │
         ▼
Test body runs — packages are guaranteed present
```

---

## Quick Start

```python
from framework.os_adapter.deps import OS_DEPS

# Simplest form — same package name on all distros
@OS_DEPS("cmake", "git", "curl")
@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
def test_hip_compile(target_executor):
    result = target_executor.run("cmake --version")
    assert result.ok
```

That's it. The framework detects the distro, probes whether cmake/git/curl are installed,
and installs anything missing before `test_hip_compile` runs.

---

## Declaration Styles

There are three places you can declare dependencies.

### On the test function (most common)

```python
@OS_DEPS("cmake", "git")
def test_something(target_executor):
    ...
```

### On the test class (applies to every method)

```python
class TestROCMCompiler:
    __OS_DEPS__ = OS_DEPS("cmake", "libdrm-dev")

    def test_compile_simple(self, target_executor):
        ...

    def test_compile_with_flags(self, target_executor):
        ...   # cmake and libdrm-dev are installed for this one too
```

### Inside the test body (conditional)

Use this when you don't know at write time what you'll need:

```python
def test_conditional_deps(target_executor):
    result = target_executor.run("./detect_features.sh")

    if "NUMA" in result.stdout:
        OS_DEPS().install("libnuma-dev")

    if "DRM" in result.stdout:
        OS_DEPS().install({"apt": "libdrm-dev", "dnf": "libdrm-devel", "*": "libdrm-dev"})
```

### Resolution priority

When more than one level is set, the most specific one wins:

```
Function-level  ──► wins over ──► Class-level  ──► wins over ──► Module-level
```

---

## Package Specification Forms

### Plain string — same name everywhere

```python
@OS_DEPS("cmake", "git", "wget")
```

### Version pin

```python
@OS_DEPS("cmake==3.26")
```

The framework translates `==` to the right syntax per PM:

| Package spec   | apt            | dnf/yum      | zypper       | pip          |
|----------------|----------------|--------------|--------------|--------------|
| `cmake`        | cmake          | cmake        | cmake        | cmake        |
| `cmake==3.26`  | cmake=3.26\*   | cmake-3.26   | cmake=3.26   | cmake==3.26  |

### Cross-distro name mapping

Many packages have different names on different distros. Use a dict:

```python
@OS_DEPS({"apt": "libssl-dev", "dnf": "openssl-devel", "*": "libssl-dev"})
```

The `"*"` key is the fallback for any package manager not explicitly listed.

You can also pass a **list** of packages per manager:

```python
@OS_DEPS({
    "apt": ["libssl-dev", "libdrm-dev"],
    "dnf": ["openssl-devel", "libdrm-devel"],
    "*": "libssl-dev",
})
```

### Custom installer (pip, snap, conda, etc.)

For Python packages or anything outside the OS package manager:

```python
@OS_DEPS("onnxruntime-rocm", command="pip")

@OS_DEPS(
    "torch==2.1",
    command="pip",
    extra_args=["--index-url", "https://download.pytorch.org/whl/rocm6.0"],
)

@OS_DEPS("node --classic", command="snap")
```

### Custom repository source

```python
# Same source URL for all package managers
@OS_DEPS("my-pkg", src="https://my-repo.example.com/apt/")

# Different URL per package manager
@OS_DEPS(
    {"apt": "libdrm-dev", "dnf": "libdrm-devel", "*": "libdrm-dev"},
    sources={
        "apt": "deb https://apt.custom-repo.example.com/ stable main",
        "dnf": "https://dnf.custom-repo.example.com/rhel/",
    },
)
```

---

## Return Value — `InstallResult`

`OS_DEPS().install()` returns an `InstallResult` you can assert on:

```python
result = OS_DEPS().install("cmake", "libnuma-dev")

assert result.ok                           # True if no error
assert "cmake" in result.all_packages      # installed or already present
assert "cmake" in result.installed         # only if actually installed this run
assert "libnuma-dev" in result.already_present  # only if probe found it
assert result.dry_run is False             # True when running with --no-gpu
```

| Field             | Type        | Meaning                                              |
|-------------------|-------------|------------------------------------------------------|
| `ok`              | bool        | True unless a RuntimeError was raised                |
| `installed`       | list[str]   | Packages that were missing and got installed         |
| `already_present` | list[str]   | Packages already on the node — probe passed          |
| `all_packages`    | list[str]   | Union of installed + already_present                 |
| `dry_run`         | bool        | True when running in DryRun / --no-gpu mode          |
| `command`         | str \| None | Custom installer used ("pip", "snap", …) or None     |

---

## What Happens When a Package Is Missing

Control what the framework does when a required package is absent:

```python
# Default — install it
@OS_DEPS("cmake")

# Skip the test if missing (optional dependency)
@OS_DEPS("libcuda-dev", on_missing="skip")

# Hard fail if missing (mandatory dependency)
@OS_DEPS("hipcc", on_missing="fail")

# Log a warning only, continue the test
@OS_DEPS("optional-tool", on_missing="probe")
```

You can also set a default for the whole CI run with an environment variable:

```bash
ROCM_TEST_PKG_ON_MISSING=skip pytest ...    # skip all tests with missing deps
ROCM_TEST_PKG_ON_MISSING=install pytest ... # install everything (default)
```

---

## Platform Support

The framework auto-detects the package manager on the execution node.

```
Execution node
      │
      ▼
Probe: command -v apt-get? command -v dnf? command -v zypper? ...
      │
      ├── Linux
      │     apt / apt-get  ── Ubuntu, Debian
      │     dnf            ── RHEL 8+, Fedora, Rocky, AlmaLinux
      │     yum            ── RHEL 7, CentOS (fallback after dnf)
      │     zypper         ── openSUSE, SLES
      │     apk            ── Alpine Linux
      │     pacman         ── Arch Linux
      │
      └── Windows
            winget  ── built-in since Windows 10 1809
            choco   ── Chocolatey (bootstrapped if missing)
            scoop   ── Scoop, user-mode (bootstrapped if missing)
```

Windows package managers are tried in preference order. If the first one
(`winget`) is missing, the framework bootstraps it automatically before
proceeding. The same applies to `choco` and `scoop`.

### Sudo handling

On standard fleet nodes (non-root with passwordless sudo), installs run as
`sudo -n apt-get install ...`. On Docker containers running as root (no sudo
binary), the framework detects this and runs `apt-get install ...` directly
without sudo. The sudo check is cached per executor so it only runs once per
session.

---

## DryRun Mode

When running with `--no-gpu` (CI pre-validation gate), no real installs happen.
The framework logs what it would have done:

```
INFO  os_deps: [DryRun] would install: cmake, libnuma-dev
```

The `InstallResult` returned has `dry_run=True` and empty `installed` / `already_present`
lists, so assertions that check `result.ok` still pass.

---

## How It Works Internally

```
pytest collects test_something
         │
         ▼
_os_deps_autouse fixture runs (autouse=True)
  │
  ├── reads test_fn.__OS_DEPS__      ← set by @OS_DEPS decorator
  ├── or reads TestClass.__OS_DEPS__ ← set by class attribute
  │
  ▼
_execute_install(executor, spec)
  │
  ├── DryRun? ──► echo log, return InstallResult(dry_run=True)
  │
  ├── command= set? ──► pip/snap/conda install, return result
  │
  └── OS PM path:
        │
        ▼
      _detect_backend(executor)  ← cached after first detection
        │
        ▼
      _probe_missing(executor, backend, pkgs)
        │   runs: bash -c '(dpkg-query -W cmake && echo OK:cmake || echo MISS:cmake); ...'
        │   single batched command → one log line
        │
        ├── all present ──► return InstallResult(already_present=[...])
        │
        └── some missing:
              │
              ▼
            _resolve_sudo_prefix(executor, backend)  ← cached
              │   probes: command -v sudo && sudo -n true
              │   root/no-sudo → ""   fleet node → "sudo -n "
              │
              ▼
            _run_install_cmd(executor, backend, missing, src, extra_args)
              │
              ▼
            return InstallResult(installed=[...], already_present=[...])

         ▼
test body runs
```

---

## Probe Before Install

The framework always checks whether a package is already installed before running
the package manager. This avoids unnecessary `apt-get update` round-trips (which
take 5–40 seconds on a cold cache) when a package is already present.

All package probes are batched into a **single shell command**:

```bash
bash -c '
  (dpkg-query -W cmake >/dev/null 2>&1 && echo OK:cmake || echo MISS:cmake);
  (dpkg-query -W git   >/dev/null 2>&1 && echo OK:git   || echo MISS:git);
  (dpkg-query -W curl  >/dev/null 2>&1 && echo OK:curl  || echo MISS:curl)
'
```

This produces one executor log line instead of one per package.

---

## Session-Level Deduplication

Within a single pytest session, each `(executor_id, package, source_url)` tuple
is tracked. If test A already installed `cmake` on `node-01`, test B running on
the same node gets the cached result — no second install attempt.

---

## Integration with the Framework

The plugin registers at position 15 in `conftest.py`, after `install_plugin`:

```
markers_plugin   (1) — must be first
session_plugin   (2) — must be second
...
install_plugin   (14) — fleet-wide pre-session install
os_deps_plugin   (15) — per-test install
```

The `--pre-install pkg=` CLI option (position 14) remains the right tool for
packages that every test on every node needs. Use `@OS_DEPS` for anything
that is test-specific or area-specific.

---

## Full Example

```python
from framework.os_adapter.deps import OS_DEPS
from framework.packages.abstract_installer import PackageSpec
import pytest


# ── Simple case: same package name on all distros ───────────────────────────

@OS_DEPS("cmake", "git")
@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
def test_hip_project_build(target_executor):
    result = target_executor.run("cmake --version")
    assert result.ok


# ── Cross-distro: different package name per PM ──────────────────────────────

@OS_DEPS({"apt": "libdrm-dev", "dnf": "libdrm-devel", "*": "libdrm-dev"})
@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
def test_drm_device_access(target_executor):
    result = target_executor.run("find /usr/include -name 'drm.h' | head -1")
    assert result.ok
    assert result.stdout.strip()


# ── Python package via pip ───────────────────────────────────────────────────

@OS_DEPS("onnxruntime-rocm", command="pip")
@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
def test_onnx_rocm_provider(target_executor):
    result = OS_DEPS().install("onnxruntime-rocm", command="pip")
    assert result.ok
    assert "onnxruntime-rocm" in result.all_packages

    result = target_executor.run(
        "python3 -c 'import onnxruntime; print(onnxruntime.__version__)'"
    )
    assert result.ok


# ── Class-level: shared deps for a whole test area ───────────────────────────

class TestROCMExamplesVulkan:
    __OS_DEPS__ = OS_DEPS(
        {"apt": "libvulkan-dev", "dnf": "vulkan-devel", "*": "libvulkan-dev"},
        {"apt": "libglfw3-dev",  "dnf": "glfw-devel",   "*": "libglfw3-dev"},
    )

    @pytest.mark.hw.gpu
    @pytest.mark.ci.nightly
    @pytest.mark.layer.runtime
    @pytest.mark.runtime.medium
    def test_vulkan_init(self, target_executor):
        result = target_executor.run("vulkaninfo --summary")
        assert result.ok

    @pytest.mark.hw.gpu
    @pytest.mark.ci.nightly
    @pytest.mark.layer.runtime
    @pytest.mark.runtime.medium
    def test_vulkan_compute(self, target_executor):
        result = target_executor.run("./vk_compute_test")
        assert result.ok


# ── Conditional install based on runtime output ───────────────────────────────

@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
def test_adaptive_deps(target_executor):
    features = target_executor.run("./detect_hardware_features.sh")

    if "NUMA" in features.stdout:
        r = OS_DEPS().install(
            {"apt": "libnuma-dev", "dnf": "numactl-devel", "*": "libnuma-dev"}
        )
        assert r.ok

    if "INFINIBAND" in features.stdout:
        r = OS_DEPS().install("libibverbs-dev", on_missing="skip")
        assert r.ok
```

---

## Reference

| Symbol | Where to import | Purpose |
|--------|----------------|---------|
| `OS_DEPS` | `from framework.os_adapter.deps import OS_DEPS` | Declare / install dependencies |
| `InstallResult` | `from framework.os_adapter import InstallResult` | Type returned by `install()` |

The full plugin implementation lives in `framework/plugins/os_deps_plugin.py`.
Copy that single file to use this feature in another repo.
