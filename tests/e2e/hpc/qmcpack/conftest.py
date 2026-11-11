# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Clone and HIP-build QMCPACK for the functional deterministic suite.

QMCPACK is a third-party CMake project, so this uses the framework's
remote-transparent external-build primitives:

    * ``external_build.clone_repo``             -- idempotent git clone
    * ``external_build.assert_license_present`` -- provenance guard
    * ``install_packages``                      -- OS build dependencies
    * separate cmake configure, build, and install steps, with the ROCm
      environment injected via an ``env VAR=... cmd`` prefix rather than by
      mutating ``os.environ``.

``test_qmcpack_functional`` then runs ``ctest -R deterministic`` on each of
the four OpenMP+HIP configurations. This is the functional suite. It does
not collect or compare QMCPACK performance scores.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import pathlib
import re
import shlex
import time

from filelock import FileLock
import pytest

from framework.builder.binary_builder import resolve_parallel_jobs
from framework.executors.background_process import _blocking_stream_run
from tests.common.os_packages import detect_os, install_packages
from tests.e2e.hpc.qmcpack._workload import BUILD_TIMEOUT, CONFIGS, QMCPACK_REF, QMCPACK_URL

logger = logging.getLogger(__name__)

_BUILD_SENTINEL = "CTestTestfile.cmake"
_LOCK_DIR = pathlib.Path("output") / ".pkg-locks"
_RUN_ID = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getpid())

# Ubuntu names are installed as given. install_packages appends ``-devel`` on
# RHEL and SLES, so this list is the base names that suffix correctly.
_UBUNTU_PACKAGES = [
    "git",
    "cmake",
    "g++",
    "make",
    "ca-certificates",
    "python3",
    "libxml2-dev",
    "libhdf5-dev",
    "zlib1g-dev",
    "libopenblas-dev",
    "liblapack-dev",
    "libboost-dev",
    "libboost-all-dev",
    "libomp-dev",
    "libfftw3-dev",
]
_RPM_DEVEL_PACKAGES = [
    "boost",
    "lapack",
    "openblas",
    "hdf5",
    "libxml2",
    "zlib",
]
# These names must not go through install_packages: it would turn them into
# git-devel, cmake-devel, and fftw3-devel. RHEL's FFTW devel package is
# fftw-devel; SLES's is fftw3-devel.
_RPM_EXACT_PACKAGES = {
    "rhel": ["git", "cmake", "gcc-c++", "make", "python3", "fftw-devel"],
    "sles": ["git", "cmake", "gcc-c++", "make", "python3", "fftw3-devel"],
}


@dataclass(frozen=True)
class QmcpackBuild:
    """Result of a session-scoped QMCPACK HIP build.

    Attributes:
        build_dirs: One CMake build directory per complex/precision configuration.
        source_dir: Cloned QMCPACK checkout.
    """

    build_dirs: tuple[str, ...]
    source_dir: str


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _path_exists(path: str, cmake_executor) -> bool:
    """Return True when *path* is a file, locally or on the build node."""
    if cmake_executor is not None:
        return bool(cmake_executor.run(f"test -f {shlex.quote(path)}", timeout=30.0).ok)
    return os.path.isfile(path)


class _LocalShell:
    """Local stand-in for ``executor.run`` when the build is not remote."""

    def run(self, command: str, timeout: float = 600, stream: bool = False):
        return _blocking_stream_run(
            command=command,
            env=os.environ.copy(),
            cwd=None,
            timeout=timeout,
            stream_stdout=stream,
            stream_stderr=stream,
            log_path=None,
        )


def _shell(cmake_executor):
    """Return the build-node executor, or a local shell when the build is local."""
    return cmake_executor if cmake_executor is not None else _LocalShell()


def _install_exact(executor, family: str, packages: list[str]) -> None:
    """Install package names that must not receive the ``-devel`` suffix."""
    pkg_str = " ".join(packages)
    if family == "rhel":
        cmd = f"sudo dnf install -y --nogpgcheck {pkg_str}"
    else:
        cmd = f"sudo zypper install -y --no-gpg-checks {pkg_str}"
    logger.info("installing exact packages [%s]: %s", family, pkg_str)
    result = executor.run(cmd, timeout=600)
    if not result.ok:
        logger.warning(
            "exact package install [%s] returned non-zero (exit=%s) for [%s]\n%s",
            family,
            result.exit_code,
            pkg_str,
            (result.stderr or "")[:500],
        )


def _install_build_packages(executor) -> None:
    """Install QMCPACK's OS build dependencies once."""
    family = detect_os(executor)
    if family == "ubuntu":
        update = executor.run("sudo apt-get update -qq", timeout=600)
        if not update.ok:
            logger.warning("apt-get update failed:\n%s", (update.stderr or "")[:500])
        install_packages(executor, _UBUNTU_PACKAGES)
        return
    install_packages(executor, _RPM_DEVEL_PACKAGES)
    _install_exact(executor, family, _RPM_EXACT_PACKAGES[family])


def _resolve_compilers(executor, rock_dir: str) -> tuple[str, str]:
    """Return ``(cc, cxx)``. Prefer amdclang++. QMCPACK's OpenMP+HIP build uses it."""
    rock = shlex.quote(rock_dir)
    probe = (
        f"for bindir in {rock}/lib/llvm/bin {rock}/llvm/bin {rock}/bin; do "
        'if [ -x "$bindir/amdclang++" ]; then '
        'echo "CC=$bindir/amdclang"; echo "CXX=$bindir/amdclang++"; exit 0; fi; '
        'if [ -x "$bindir/clang++" ]; then '
        'echo "CC=$bindir/clang"; echo "CXX=$bindir/clang++"; exit 0; fi; '
        "done; exit 1"
    )
    result = executor.run(f"bash -lc {shlex.quote(probe)}", timeout=30)
    text = result.stdout or ""
    cc = next((ln.split("=", 1)[1].strip() for ln in text.splitlines() if ln.startswith("CC=")), "")
    cxx = next((ln.split("=", 1)[1].strip() for ln in text.splitlines() if ln.startswith("CXX=")), "")
    if not result.ok or not cxx:
        raise RuntimeError(
            f"ROCm clang not found under {rock_dir} (exit={result.exit_code}):\n{(result.stderr or '')[-1000:]}"
        )
    return cc, cxx


def _run_step(cmd: str, *, executor, timeout: float, label: str, log_path: str) -> None:
    """Run one cmake step and stream its output."""
    logger.info("QMCPACK %s -> %s", label, log_path)
    if not isinstance(executor, _LocalShell):
        result = executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"QMCPACK {label} failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return
    result = _blocking_stream_run(
        command=cmd,
        env=os.environ.copy(),
        cwd=None,
        timeout=timeout,
        stream_stdout=True,
        stream_stderr=True,
        log_path=log_path,
    )
    if not result.ok:
        raise RuntimeError(
            f"QMCPACK {label} failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


@pytest.fixture(scope="session")
def qmcpack_functional_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    node_pool,
    require_gpu_arch_for,
) -> QmcpackBuild:
    """Clone QMCPACK ``develop`` and configure, build, and install each HIP configuration."""
    if not rock_dir:
        pytest.fail("rock_dir is empty — pass --rock-dir or set ROCK_DIR to the ROCm install path.")

    resolved_gpu_arch = gpu_arch
    if not resolved_gpu_arch and node_pool:
        try:
            resolved_gpu_arch = next(gpu for alloc in node_pool._allocators.values() for gpu in alloc._pool).arch
        except (StopIteration, AttributeError):
            resolved_gpu_arch = None
    if not resolved_gpu_arch:
        require_gpu_arch_for("qmcpack")
    resolved_gpu_arch = resolved_gpu_arch.split(":", 1)[0]

    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = max(float(framework_config.therock.build_timeout_secs), BUILD_TIMEOUT)

    ref_label = _safe_ref_name(QMCPACK_REF)
    dest = pathlib.Path(compiler_build_dir) / "qmcpack" / f"qmcpack-{ref_label}"
    qmc_dir = str(external_build.clone_repo(url=QMCPACK_URL, dest=dest, ref=QMCPACK_REF, timeout=build_timeout))
    external_build.assert_license_present(qmc_dir)

    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        qmc_dir = str(cmake_executor.workspace_path_for(qmc_dir))
    else:
        qmc_dir = os.path.abspath(qmc_dir)

    executor = _shell(cmake_executor)
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    sentinel = _LOCK_DIR / f"qmcpack_{_RUN_ID}.done"
    with FileLock(str(_LOCK_DIR / f"qmcpack_{_RUN_ID}.lock")):
        if not sentinel.exists():
            _install_build_packages(executor)
            sentinel.touch()

    arch_label = _safe_ref_name(resolved_gpu_arch)
    build_dirs = tuple(f"{qmc_dir}/build_{name}-{arch_label}" for name, _complex, _mixed in CONFIGS)
    if all(_path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor) for build_dir in build_dirs):
        logger.info("QMCPACK: existing build trees under %s — skipping configure/build/install", qmc_dir)
        return QmcpackBuild(build_dirs=build_dirs, source_dir=qmc_dir)

    cc, cxx = _resolve_compilers(executor, rocm_path)
    jobs = resolve_parallel_jobs(remote_executor=cmake_executor)
    log_dir = os.path.join(framework_config.framework.artifact_dir, "qmcpack")
    os.makedirs(log_dir, exist_ok=True)
    path = f"{rocm_path}/bin:{rocm_path}/llvm/bin:{rocm_path}/lib/llvm/bin"
    lib = f"{rocm_path}/lib:{rocm_path}/lib64"
    env_prefix = (
        f"ROCM_PATH={shlex.quote(rocm_path)} "
        f"PATH={shlex.quote(path)}:$PATH "
        f"LD_LIBRARY_PATH={shlex.quote(lib)}:${{LD_LIBRARY_PATH:-}}"
    )
    logger.info("QMCPACK: configuring HIP builds (gpu_arch=%s, rocm=%s)", resolved_gpu_arch, rocm_path)
    # BUILD_TIMEOUT is the wall clock for every configure, build, and install
    # across the four configurations, not the cap of each step.
    build_deadline = time.monotonic() + BUILD_TIMEOUT

    def step_timeout() -> float:
        remaining = build_deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                "QMCPACK configure/build/install exceeded the "
                f"{BUILD_TIMEOUT:.0f}s budget shared by all four configurations"
            )
        return remaining

    for name, complex_, mixed in CONFIGS:
        build = f"{qmc_dir}/build_{name}-{arch_label}"
        install = f"{qmc_dir}/install_{name}-{arch_label}"
        fresh = executor.run(f"rm -rf {shlex.quote(build)} && mkdir -p {shlex.quote(build)}", timeout=60)
        if not fresh.ok:
            raise RuntimeError(f"Could not recreate {build}: {(fresh.stderr or '')[-500:]}")
        # Nexus generation during cmake can fail when a user-site package prints
        # DeprecationWarning into the configure Python subprocess.
        # QMC_OFFLOAD_MEM_ASSOCIATED is left unset: current QMCPACK only honors it
        # for CUDA or SYCL, and forces it off for this HIP build.
        configure_cmd = (
            f"env {env_prefix} PYTHONWARNINGS=ignore::DeprecationWarning "
            f"cmake -S {shlex.quote(qmc_dir)} -B {shlex.quote(build)} "
            "-DCMAKE_BUILD_TYPE=Release "
            f"-DROCM_PATH={shlex.quote(rocm_path)} "
            f"-DCMAKE_C_COMPILER={shlex.quote(cc)} "
            f"-DCMAKE_CXX_COMPILER={shlex.quote(cxx)} "
            "-DQMC_MPI=OFF "
            '"-DQMC_GPU=openmp;hip" '
            f"-DQMC_GPU_ARCHS={shlex.quote(resolved_gpu_arch)} "
            f"-DQMC_COMPLEX={complex_} "
            f"-DQMC_MIXED_PRECISION={mixed} "
            f"-DCMAKE_INSTALL_PREFIX={shlex.quote(install)}"
        )
        build_cmd = f"env {env_prefix} cmake --build {shlex.quote(build)} -j{jobs}"
        install_cmd = f"env {env_prefix} cmake --install {shlex.quote(build)}"
        _run_step(
            configure_cmd,
            executor=executor,
            timeout=step_timeout(),
            label=f"{name} configure",
            log_path=os.path.join(log_dir, f"{name}-configure.log"),
        )
        _run_step(
            build_cmd,
            executor=executor,
            timeout=step_timeout(),
            label=f"{name} build",
            log_path=os.path.join(log_dir, f"{name}-build.log"),
        )
        _run_step(
            install_cmd,
            executor=executor,
            timeout=step_timeout(),
            label=f"{name} install",
            log_path=os.path.join(log_dir, f"{name}-install.log"),
        )
    return QmcpackBuild(build_dirs=build_dirs, source_dir=qmc_dir)
