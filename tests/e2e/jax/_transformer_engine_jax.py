# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
_transformer_engine_jax.py -- Shared helpers for the Transformer Engine JAX suite.

This module holds the executor-agnostic logic for exercising the ROCm
Transformer Engine (TE) JAX unit tests and C++ unit tests on an AMD GPU node:

    * ``TeJaxConfig``           -- wheel filenames, artifact base URLs, and branch,
                                   populated from the environment.
    * ``build_wheel_urls``      -- construct full artifact URLs from bare filenames,
                                   detecting nightly vs release builds and the
                                   GPU-arch-specific JAX index.
    * ``build_rocm_envs`` /
      ``render_env_prefix``     -- assemble the ROCm environment and render it into
                                   an ``env KEY=VALUE`` command prefix.
    * ``parse_te_tests``        -- parse ctest / pytest result lines into a summary.
    * The ``install_*`` / ``run_*`` helpers -- drive the build and test commands.

Every helper that touches the GPU node takes a ``NodeExecutorGroup`` (``executor``)
and goes through ``executor.run(...)`` so it works transparently across local, SSH,
and container execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.executor_group import NodeExecutorGroup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TE_REPO_URL = "https://github.com/ROCm/TransformerEngine.git"
DEFAULT_TE_GIT_BRANCH = "dev"

# Clone / build / run wall-clock budgets (seconds). The suite builds TE from
# source and runs the full JAX unit tests at TEST_LEVEL=3, hence the long budgets.
CLONE_TIMEOUT = 1800.0
INSTALL_TIMEOUT = 3600.0
BUILD_TIMEOUT = 7200.0
TEST_TIMEOUT = 14400.0

# The TE JAX unit-test level exercised by ci/jax.sh (full coverage).
TEST_LEVEL = "3"

# Node-side paths (``$HOME`` is expanded by the shell on the target node).
TE_DIR = "$HOME/TransformerEngine"
VENV_DIR = "$HOME/venvs/te_jax_ut"
VENV_BIN = f"{VENV_DIR}/bin"
VENV_PYTHON = f"{VENV_BIN}/python"
VENV_SITE = f"{VENV_DIR}/lib/python3.12/site-packages"
PYTHON312 = "/usr/bin/python3.12"

# CMake pinned for the C++ unit-test build.
CMAKE_VERSION = "3.29.3"

# Python dependencies installed into the venv before the JAX and TE wheels.
VENV_PYTHON_DEPS = "setuptools wheel 'numpy>=2.0' 'scipy>=1.13' 'ml_dtypes>=0.5.0' opt_einsum cmake ninja pybind11"

# The three JAX wheels are GPU-arch-specific; the three TE artifacts are not.
JAX_WHEEL_KEYS = ("jax_whl", "jax_rocm_plugin_whl", "jax_rocm_plugin_native_whl")

# Result-line patterns.
JAX_UT_PATTERN = r"(?P<testcase>.*)\s+(?P<status>PASSED|SKIPPED|FAILED)"
CPP_UT_PATTERN = (
    r"Test\s+#\d+: (?P<testcase>[\w/.]+)\s+.*(?:\*\*\*.*)?"
    r"(?P<status>Failed|Passed|Skipped|Exception|Error)(?::)?\s*[\d\.]+\s+sec"
)

# Markers the C++ build output must contain before ``make test`` is worth running.
CPP_PREPROCESS_MARKER = "Successfully preprocessed all matching files"
CPP_BUILD_MARKER = "[100%] Built target test_operator"

# Supported GPU architectures (matches gfx94X and gfx950).
SUPPORTED_ARCH_PREFIX = "gfx94"
SUPPORTED_ARCH_EXACT = "gfx950"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeJaxConfig:
    """Wheel filenames, artifact base URLs, and branch for the TE JAX suite.

    Attributes:
        te_git_branch:              TransformerEngine branch to clone.
        jax_whl:                    ``jaxlib`` wheel filename.
        jax_rocm_plugin_whl:        JAX ROCm plugin wheel filename.
        jax_rocm_plugin_native_whl: JAX ROCm plugin native wheel filename.
        te_whl:                     Transformer Engine wheel filename.
        te_rocm_whl:                Transformer Engine ROCm wheel filename.
        te_rocm_jax_tar:            Transformer Engine ROCm JAX archive filename.
        jax_release_url:            Base URL for release JAX artifacts.
        jax_nightly_url:            Base URL for nightly JAX artifacts.
        te_release_url:             Base URL for release TE artifacts.
        te_nightly_url:             Base URL for nightly TE artifacts.
        gfx_family:                 GPU family used for the arch-specific JAX index.
    """

    te_git_branch: str = DEFAULT_TE_GIT_BRANCH
    jax_whl: str = ""
    jax_rocm_plugin_whl: str = ""
    jax_rocm_plugin_native_whl: str = ""
    te_whl: str = ""
    te_rocm_whl: str = ""
    te_rocm_jax_tar: str = ""
    jax_release_url: str = ""
    jax_nightly_url: str = ""
    te_release_url: str = ""
    te_nightly_url: str = ""
    gfx_family: str = ""

    @classmethod
    def from_env(cls) -> TeJaxConfig:
        """Build a config from ``ROCM_TEST_*`` environment variables.

        Returns:
            A populated ``TeJaxConfig`` (unset variables fall back to defaults).
        """

        def _env(name: str, default: str = "") -> str:
            return os.environ.get(name, default).strip()

        return cls(
            te_git_branch=_env("ROCM_TEST_TE_GIT_BRANCH", DEFAULT_TE_GIT_BRANCH),
            jax_whl=_env("ROCM_TEST_JAX_WHL"),
            jax_rocm_plugin_whl=_env("ROCM_TEST_JAX_ROCM_PLUGIN_WHL"),
            jax_rocm_plugin_native_whl=_env("ROCM_TEST_JAX_ROCM_PLUGIN_NATIVE_WHL"),
            te_whl=_env("ROCM_TEST_TE_WHL"),
            te_rocm_whl=_env("ROCM_TEST_TE_ROCM_WHL"),
            te_rocm_jax_tar=_env("ROCM_TEST_TE_ROCM_JAX_TAR"),
            jax_release_url=_env("ROCM_TEST_JAX_RELEASE_URL"),
            jax_nightly_url=_env("ROCM_TEST_JAX_NIGHTLY_URL"),
            te_release_url=_env("ROCM_TEST_TE_RELEASE_URL"),
            te_nightly_url=_env("ROCM_TEST_TE_NIGHTLY_URL"),
            gfx_family=_env("ROCM_TEST_GFX_FAMILY"),
        )

    @property
    def te_filenames(self) -> dict[str, str]:
        """Return the TE artifact filenames keyed by config attribute name."""
        return {
            "te_whl": self.te_whl,
            "te_rocm_whl": self.te_rocm_whl,
            "te_rocm_jax_tar": self.te_rocm_jax_tar,
        }

    @property
    def all_filenames(self) -> dict[str, str]:
        """Return every wheel/archive filename keyed by config attribute name."""
        return {
            "jax_whl": self.jax_whl,
            "jax_rocm_plugin_whl": self.jax_rocm_plugin_whl,
            "jax_rocm_plugin_native_whl": self.jax_rocm_plugin_native_whl,
            **self.te_filenames,
        }


@dataclass(frozen=True)
class WheelUrls:
    """Fully-resolved artifact URLs plus the JAX index and version.

    Attributes:
        urls:          Full artifact URLs keyed by config attribute name.
        jax_index_url: Arch-specific JAX index URL for ``--extra-index-url``.
        jax_version:   JAX version string extracted from the jaxlib filename.
    """

    urls: dict[str, str]
    jax_index_url: str
    jax_version: str


@dataclass(frozen=True)
class TeTestSummary:
    """Aggregate per-testcase result counts parsed from a suite run.

    Attributes:
        passed:  Number of PASSED test cases.
        failed:  Number of FAILED test cases.
        skipped: Number of SKIPPED test cases.
        error:   Number of ERROR / EXCEPTION test cases.
    """

    passed: int
    failed: int
    skipped: int
    error: int

    @property
    def total(self) -> int:
        """Total number of matched result lines."""
        return self.passed + self.failed + self.skipped + self.error

    @property
    def ran(self) -> bool:
        """True when at least one result line was matched."""
        return self.total > 0


REQUIRED_FILENAMES: dict[str, tuple[str, ...]] = {
    "te_jax_ut": (
        "te_whl",
        "te_rocm_whl",
        "jax_whl",
        "jax_rocm_plugin_whl",
        "jax_rocm_plugin_native_whl",
        "te_rocm_jax_tar",
    ),
    "te_jax_cpp_ut": ("te_whl", "te_rocm_whl"),
}


# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


def missing_filenames(config: TeJaxConfig, test_name: str) -> list[str]:
    """Return the ``--arg`` names required by *test_name* that are unset.

    Args:
        config:    The suite configuration.
        test_name: Either ``"te_jax_ut"`` or ``"te_jax_cpp_ut"``.

    Returns:
        Sorted list of missing argument flags (empty when all are present).
    """
    return sorted(f"--{name}" for name in REQUIRED_FILENAMES[test_name] if not getattr(config, name))


def is_supported_arch(arch: str) -> bool:
    """Return True when *arch* is a supported gfx94X or gfx950 architecture."""
    return arch.startswith(SUPPORTED_ARCH_PREFIX) or arch == SUPPORTED_ARCH_EXACT


def is_release_build(reference_filename: str) -> bool:
    """Return True when *reference_filename* is a release (non-nightly) artifact.

    Nightly artifacts embed an ``a<YYYYMMDD>`` date stamp in the filename.

    Args:
        reference_filename: Any wheel/archive filename to inspect.

    Returns:
        True for release builds, False for nightly builds.
    """
    return not bool(re.search(r"a\d{8}", reference_filename))


def extract_jax_version(jax_whl: str) -> str:
    """Extract the ``X.Y.Z`` JAX version from a jaxlib wheel filename.

    Args:
        jax_whl: The jaxlib wheel filename (e.g. ``jaxlib-0.8.2+rocm...whl``).

    Returns:
        The version string (e.g. ``"0.8.2"``).

    Raises:
        ValueError: If the version cannot be extracted from *jax_whl*.
    """
    match = re.search(r"jaxlib-([\d]+\.[\d]+\.[\d]+)", jax_whl)
    if not match:
        raise ValueError(f"Cannot extract JAX version from filename '{jax_whl}'; expected jaxlib-X.Y.Z+rocm...")
    return match.group(1)


def resolve_te_base(config: TeJaxConfig) -> str:
    """Return the TE artifact base URL for the detected build type.

    Args:
        config: The suite configuration.

    Returns:
        The TE base URL with any trailing slash removed.

    Raises:
        ValueError: If the matching base URL is not configured.
    """
    reference = next((name for name in config.te_filenames.values() if name), "")
    release = is_release_build(reference)
    te_base = config.te_release_url if release else config.te_nightly_url
    if not te_base:
        raise ValueError(f"TE {'release' if release else 'nightly'} base URL is not configured")
    return te_base.rstrip("/")


def build_wheel_urls(config: TeJaxConfig, gfx_family: str) -> WheelUrls:
    """Construct full artifact URLs from the configured bare filenames.

    Detects nightly vs release from the filenames, extracts the JAX version, and
    builds the GPU-arch-specific JAX index (``jax_base/<gfx_family>/``). JAX wheels
    are resolved against the arch index; TE artifacts against the flat TE base.

    Args:
        config:     The suite configuration.
        gfx_family: GPU family for the arch-specific JAX index.

    Returns:
        A ``WheelUrls`` with resolved URLs, JAX index URL, and JAX version.

    Raises:
        ValueError: If a required base URL, the GPU family, or the JAX version
            cannot be resolved.
    """
    filenames = config.all_filenames
    reference = next((name for name in filenames.values() if name), "")
    release = is_release_build(reference)

    jax_version = extract_jax_version(config.jax_whl)

    jax_base = config.jax_release_url if release else config.jax_nightly_url
    te_base = config.te_release_url if release else config.te_nightly_url
    if not jax_base:
        raise ValueError(f"JAX {'release' if release else 'nightly'} base URL is not configured")
    if not te_base:
        raise ValueError(f"TE {'release' if release else 'nightly'} base URL is not configured")
    if not gfx_family:
        raise ValueError("GPU family could not be determined; set ROCM_TEST_GFX_FAMILY")

    jax_arch_base = f"{jax_base.rstrip('/')}/{gfx_family}/"
    urls = {
        key: f"{(jax_arch_base if key in JAX_WHEEL_KEYS else te_base).rstrip('/')}/{filename}"
        for key, filename in filenames.items()
        if filename
    }
    return WheelUrls(urls=urls, jax_index_url=jax_arch_base, jax_version=jax_version)


def build_rocm_envs(rock_dir: str, libstdcxx_dir: str) -> dict[str, str]:
    """Assemble the ROCm environment for the venv build and test commands.

    Args:
        rock_dir:      TheRock/ROCm install path.
        libstdcxx_dir: GCC-13 ``libstdc++`` directory (empty when not found).

    Returns:
        Ordered mapping of environment variable names to values.
    """
    ld_lib = f"{libstdcxx_dir}:{rock_dir}/lib" if libstdcxx_dir else f"{rock_dir}/lib"
    return {
        "VIRTUAL_ENV": VENV_DIR,
        "HIP_PATH": rock_dir,
        "ROCM_PATH": rock_dir,
        "AMD_COMGR_NAMESPACE": "1",
        "LLVM_PATH": f"{rock_dir}/llvm",
        "HIP_DEVICE_LIB_PATH": f"{rock_dir}/lib/llvm/amdgcn/bitcode",
        "JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH": f"{rock_dir}/lib/llvm/amdgcn/bitcode",
        "JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH": f"{rock_dir}/lib/llvm/bin",
        "LD_LIBRARY_PATH": ld_lib,
    }


def render_env_prefix(env: dict[str, str], before_keys: set[str]) -> str:
    """Render an environment mapping into an ``env KEY=VALUE`` command prefix.

    Keys in *before_keys* are prepended to the node's existing value for that
    variable (``KEY="value:$KEY"``); all other keys are set verbatim.

    Args:
        env:         Ordered mapping of environment variables.
        before_keys: Variables whose value is prepended to the existing one.

    Returns:
        A shell command prefix beginning with ``env``.
    """
    parts = []
    for key, value in env.items():
        if key in before_keys:
            parts.append(f'{key}="{value}:${key}"')
        else:
            parts.append(f'{key}="{value}"')
    return "env " + " ".join(parts)


def parse_te_tests(output: str, pattern: str) -> TeTestSummary:
    """Parse ctest / pytest result lines into a ``TeTestSummary``.

    Each matched line contributes one testcase; the status is normalised so that
    ``PASSED``/``Passed`` count as passed, ``FAILED``/``Failed`` as failed,
    ``SKIPPED``/``Skipped`` as skipped, and ``Exception``/``Error`` as error. An
    unrecognised status counts as a failure.

    Args:
        output:  Combined stdout/stderr from the suite run.
        pattern: Regex with named groups ``testcase`` and ``status``.

    Returns:
        A ``TeTestSummary`` of the per-status counts.
    """
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
    for line in (output or "").splitlines():
        match = re.search(pattern, line.replace("\n", ""))
        if not match:
            continue
        status = match.group("status").upper()
        key = "ERROR" if status in ("EXCEPTION", "ERROR") else re.sub(r"(ED|PED)$", "", status)
        if key not in counts:
            key = "FAIL"
        counts[key] += 1
    return TeTestSummary(counts["PASS"], counts["FAIL"], counts["SKIP"], counts["ERROR"])


# ---------------------------------------------------------------------------
# Executor-driven detection helpers
# ---------------------------------------------------------------------------


def detect_gpu_arch(executor: NodeExecutorGroup) -> str:
    """Return the first ``gfxNNN`` architecture reported by ``rocminfo``.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.

    Returns:
        The architecture string (e.g. ``"gfx942"``), or ``""`` when unavailable.
    """
    result = executor.run("rocminfo")
    if not result.ok:
        return ""
    match = re.search(r"gfx[0-9a-fA-F]+", result.stdout)
    return match.group(0) if match else ""


def detect_gfx_family(executor: NodeExecutorGroup, override: str) -> str:
    """Return the configured GPU family, or fall back to the detected arch.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.
        override: Explicit family from configuration (wins when non-empty).

    Returns:
        The GPU family string, or ``""`` when neither is available.
    """
    if override:
        return override
    return detect_gpu_arch(executor)


# ---------------------------------------------------------------------------
# Executor-driven build helpers
# ---------------------------------------------------------------------------


def _assert_ok(result, what: str) -> None:
    """Assert *result* succeeded, raising with a diagnostic when it did not.

    Args:
        result: The ``ExecutionResult`` to check.
        what:   Human-readable description of the failed step.
    """
    assert result.ok, f"{what} (exit={result.exit_code}):\n{result.stderr[:500]}"


def install_cmake(executor: NodeExecutorGroup) -> None:
    """Install the pinned CMake into the system Python for the C++ build."""
    result = executor.run(f'python3 -m pip install "cmake=={CMAKE_VERSION}"', timeout=INSTALL_TIMEOUT)
    _assert_ok(result, f"CMake {CMAKE_VERSION} installation failed")


def install_ninja(executor: NodeExecutorGroup) -> None:
    """Install the ninja build tool into the system Python."""
    result = executor.run("python3 -m pip install ninja", timeout=INSTALL_TIMEOUT)
    _assert_ok(result, "ninja installation failed")


def clone_transformer_engine(executor: NodeExecutorGroup, branch: str) -> str:
    """Clone the TransformerEngine repository (with submodules) at *branch*.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.
        branch:   Git branch to check out.

    Returns:
        The node-side path to the cloned repository.
    """
    executor.run(f"rm -rf {TE_DIR}")
    result = executor.run(
        f"git clone --recurse-submodules -b {branch} {TE_REPO_URL} {TE_DIR}",
        timeout=CLONE_TIMEOUT,
    )
    _assert_ok(result, f"TransformerEngine clone failed for branch '{branch}'")
    return TE_DIR


def install_te_wheels_system(executor: NodeExecutorGroup, config: TeJaxConfig, te_base: str) -> None:
    """Install the TE wheels into the system Python so CMake can find TE.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.
        config:   The suite configuration (TE wheel filenames).
        te_base:  Resolved TE artifact base URL.
    """
    te_files = f"{te_base}/{config.te_whl} {te_base}/{config.te_rocm_whl}"
    result = executor.run(
        f"python3 -m pip install {te_files} --no-build-isolation",
        timeout=INSTALL_TIMEOUT,
    )
    _assert_ok(result, "TE wheel installation into system Python failed")


def install_python_toolchain(executor: NodeExecutorGroup) -> str:
    """Provision Python 3.12 and GCC-13, returning the GCC-13 libstdc++ directory.

    OS-package provisioning is best-effort (the image usually ships these); the
    hard gate is the presence of ``/usr/bin/python3.12`` afterwards.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.

    Returns:
        The GCC-13 ``libstdc++`` directory, or ``""`` when it cannot be located.
    """
    is_rhel = executor.run("test -f /etc/redhat-release").ok
    if is_rhel:
        executor.run(
            "sudo dnf install -y --nogpgcheck python3.12 python3.12-devel gcc-toolset-13-libstdc++-devel || true",
            timeout=INSTALL_TIMEOUT,
        )
    else:
        executor.run(
            "sudo apt-get install -y --allow-unauthenticated software-properties-common || true",
            timeout=INSTALL_TIMEOUT,
        )
        executor.run("sudo add-apt-repository ppa:deadsnakes/ppa -y || true", timeout=INSTALL_TIMEOUT)
        executor.run("sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y || true", timeout=INSTALL_TIMEOUT)
        executor.run("sudo apt-get update -y || true", timeout=INSTALL_TIMEOUT)
        executor.run(
            "sudo apt-get install -y --allow-unauthenticated python3.12 python3.12-venv "
            "python3.12-dev libstdc++-13-dev || true",
            timeout=INSTALL_TIMEOUT,
        )

    python312_present = bool(executor.run(f"test -x {PYTHON312} && echo ok").stdout.strip())
    assert python312_present, f"{PYTHON312} not found after OS package install"

    if is_rhel:
        find_cmd = (
            "find /opt/rh/gcc-toolset-13/root/usr/lib/gcc -name 'libstdc++.so.6' "
            "2>/dev/null | head -1 | xargs -r dirname"
        )
    else:
        find_cmd = "find /usr/lib/gcc -path '*/13/*' -name 'libstdc++.so.6' 2>/dev/null | head -1 | xargs -r dirname"
    return executor.run(find_cmd).stdout.strip()


def create_te_venv(executor: NodeExecutorGroup) -> None:
    """Create the Python 3.12 venv and install ``uv`` into the system Python.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.
    """
    executor.run(f"mkdir -p {os.path.dirname(VENV_DIR)}")
    result = executor.run(f"{PYTHON312} -m venv {VENV_DIR}", timeout=INSTALL_TIMEOUT)
    _assert_ok(result, "Python 3.12 venv creation failed")
    uv_result = executor.run("python3 -m pip install uv", timeout=INSTALL_TIMEOUT)
    _assert_ok(uv_result, "uv installation failed")


def install_ut_wheels(executor: NodeExecutorGroup, wheel_urls: WheelUrls, rocm_envs: dict[str, str]) -> None:
    """Install the Python deps, JAX wheels, and TE wheels into the venv.

    Args:
        executor:   ``NodeExecutorGroup`` from ``target_executor``.
        wheel_urls: Resolved artifact URLs, JAX index, and JAX version.
        rocm_envs:  ROCm environment mapping from :func:`build_rocm_envs`.
    """
    env_prefix = render_env_prefix(rocm_envs, {"LD_LIBRARY_PATH"})
    urls = wheel_urls.urls

    deps = executor.run(f"{env_prefix} uv pip install {VENV_PYTHON_DEPS}", timeout=INSTALL_TIMEOUT)
    _assert_ok(deps, "Python dependency installation into venv failed")

    jax_wheels = " ".join(urls[key] for key in JAX_WHEEL_KEYS)
    jax = executor.run(
        f"{env_prefix} uv pip install --extra-index-url {wheel_urls.jax_index_url} "
        f"{jax_wheels} jax=={wheel_urls.jax_version}",
        timeout=INSTALL_TIMEOUT,
    )
    _assert_ok(jax, "JAX wheel installation failed")

    te_files = " ".join(urls[key] for key in ("te_whl", "te_rocm_whl", "te_rocm_jax_tar"))
    te = executor.run(f"{env_prefix} uv pip install {te_files} --no-build-isolation", timeout=INSTALL_TIMEOUT)
    _assert_ok(te, "Transformer Engine wheel installation failed")


def teardown_venv(executor: NodeExecutorGroup) -> None:
    """Delete the venv created for the JAX unit tests (best-effort)."""
    executor.run(f"rm -rf {VENV_DIR}")


# ---------------------------------------------------------------------------
# Executor-driven run helpers
# ---------------------------------------------------------------------------


def run_jax_ut(executor: NodeExecutorGroup, te_dir: str, rocm_envs: dict[str, str]) -> TeTestSummary:
    """Run ``ci/jax.sh`` at TEST_LEVEL=3 in the venv and parse the results.

    Args:
        executor:  ``NodeExecutorGroup`` from ``target_executor``.
        te_dir:    Cloned TransformerEngine directory.
        rocm_envs: ROCm environment mapping from :func:`build_rocm_envs`.

    Returns:
        A ``TeTestSummary`` parsed from the ``jax.sh`` output.
    """
    run_envs = dict(rocm_envs)
    run_envs["TEST_LEVEL"] = TEST_LEVEL
    run_envs["JAXCI_PYTHON"] = VENV_PYTHON
    run_envs["PATH"] = VENV_BIN
    run_envs["PYTHONPATH"] = VENV_SITE
    env_prefix = render_env_prefix(run_envs, {"LD_LIBRARY_PATH", "PATH", "PYTHONPATH"})

    result = executor.run(f"cd {te_dir}/ci && {env_prefix} ./jax.sh", timeout=TEST_TIMEOUT)
    return parse_te_tests(f"{result.stdout}\n{result.stderr}", JAX_UT_PATTERN)


def run_jax_cpp_ut(executor: NodeExecutorGroup, te_dir: str) -> TeTestSummary:
    """Build and run the TE C++ unit tests, parsing the ctest results.

    Args:
        executor: ``NodeExecutorGroup`` from ``target_executor``.
        te_dir:   Cloned TransformerEngine directory.

    Returns:
        A ``TeTestSummary`` parsed from the ``make test`` output.
    """
    build_dir = f"{te_dir}/tests/cpp/build"
    executor.run(f"mkdir -p {build_dir}")
    build = executor.run(f"cd {build_dir} && cmake ../; make -j", timeout=BUILD_TIMEOUT)
    combined = f"{build.stdout}\n{build.stderr}"
    built = CPP_PREPROCESS_MARKER in combined and CPP_BUILD_MARKER in combined
    assert built, f"TE C++ build did not complete (exit={build.exit_code}); missing build markers:\n{combined[-2000:]}"

    result = executor.run(f"cd {build_dir} && make test", timeout=TEST_TIMEOUT)
    return parse_te_tests(f"{result.stdout}\n{result.stderr}", CPP_UT_PATTERN)
