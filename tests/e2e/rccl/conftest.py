# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and prerequisite fixtures for RCCL E2E tests.

``rccl-tests`` is cloned at runtime from the rocm-systems monorepo using a
sparse checkout of ``projects/rccl-tests``.  Remote runs build on the GPU node
instead of staging large checkouts over SFTP.
"""

from __future__ import annotations

import glob
import logging
import os
import pathlib
import re
import shutil
import subprocess  # nosec B404

import pytest

logger = logging.getLogger(__name__)

# rccl-tests lives inside the ROCm/rocm-systems monorepo under projects/rccl-tests/.
# Canonical source: https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl-tests
_RCCL_TESTS_MONOREPO_URL = "https://github.com/ROCm/rocm-systems.git"
_RCCL_TESTS_SUBPATH = "projects/rccl-tests"
# Default to the live rocm-systems rccl-tests source so CI tracks the latest
# TheRock/ROCm headers. Pin a known-good ref with RCCL_TESTS_REF when needed.
_DEFAULT_RCCL_TESTS_REF = "develop"
_RCCL_TESTS_REF = os.environ.get("RCCL_TESTS_REF", _DEFAULT_RCCL_TESTS_REF)

# First-party MIT stub for the error-handling negative test.
_STUB_SRC = "tests/e2e/rccl/src/rccl_error_handling/main.cpp"

# First-party MIT RCCL unroll-factor validation sources (rccl_unroll_test).
_DUAL_KERNEL_SRC = "tests/e2e/rccl/src/rccl_unroll_test/rccl_dual_kernel_build_test.cpp"
_UNROLL_PERF_MATRIX_SRC = "tests/e2e/rccl/src/rccl_unroll_test/rccl_unroll_perf_matrix_test.cpp"

# First-party MIT RCCL concurrent-collectives stress harness (concurrent_collectives).
_CONCURRENT_COLLECTIVES_SRC = "tests/e2e/rccl/src/concurrent_collectives/concurrent_collectives.cpp"


def _openssh_client_install_cmd() -> str:
    """Return a distro-aware OpenSSH client install command for remote executors."""
    return (
        "bash -lc '"
        "set -e; "
        "if command -v ssh >/dev/null 2>&1; then exit 0; fi; "
        "if command -v apt-get >/dev/null 2>&1; then "
        "  apt-get update && apt-get install -y openssh-client; "
        "elif command -v dnf >/dev/null 2>&1; then "
        "  dnf install -y openssh-clients; "
        "elif command -v yum >/dev/null 2>&1; then "
        "  yum install -y openssh-clients; "
        "elif command -v zypper >/dev/null 2>&1; then "
        "  zypper --non-interactive install openssh; "
        "else "
        '  echo "no supported package manager found to install OpenSSH client" >&2; exit 1; '
        "fi; "
        "command -v ssh >/dev/null 2>&1'"
    )


def _local_openssh_client_install_cmds() -> list[list[str]] | None:
    """Return ordered local OpenSSH client install commands, if supported."""
    package_managers = (
        ("apt-get", [["apt-get", "update"], ["apt-get", "install", "-y", "openssh-client"]]),
        ("dnf", [["dnf", "install", "-y", "openssh-clients"]]),
        ("yum", [["yum", "install", "-y", "openssh-clients"]]),
        ("zypper", [["zypper", "--non-interactive", "install", "openssh"]]),
    )
    for binary, commands in package_managers:
        if shutil.which(binary):
            return commands
    return None


def _run_local_install_cmd(command: list[str], timeout: float) -> tuple[bool, str, str]:
    """Run a local package-manager install command without shell expansion."""
    # command is selected from a hard-coded package-manager allowlist.
    result = subprocess.run(  # nosec B603
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode == 0, result.stdout, result.stderr


def _ensure_openssh_client(cmake_executor) -> None:
    """Ensure OpenMPI's default rsh launcher can find ``ssh`` on the execution host."""
    if cmake_executor is not None:
        probe = cmake_executor.run("command -v ssh", timeout=15.0)
        if probe.ok and probe.stdout.strip():
            return
        logger.info("Installing OpenSSH client for RCCL MPI")
        result = cmake_executor.run(_openssh_client_install_cmd(), timeout=300.0)
        ok, stdout, stderr = result.ok, result.stdout, result.stderr
    else:
        if shutil.which("ssh"):
            return
        install_cmds = _local_openssh_client_install_cmds()
        if install_cmds is None:
            pytest.fail("RCCL MPI needs ssh; no supported package manager found to install OpenSSH client.")
        logger.info("Installing OpenSSH client for RCCL MPI")
        for install_cmd in install_cmds:
            ok, stdout, stderr = _run_local_install_cmd(install_cmd, 300.0)
            if not ok:
                break

    if not ok:
        pytest.fail(
            "RCCL MPI needs ssh; OpenSSH client install failed.\n" f"stdout: {stdout[-1000:]}\nstderr: {stderr[-1000:]}"
        )


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _build_rccl_tests(
    *,
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    mpi_enabled: bool,
    mpi_runtime=None,
) -> str:
    """Sparse-clone and build ``projects/rccl-tests``."""
    rocm_path = os.path.realpath(rock_dir)
    build_timeout = float(framework_config.therock.build_timeout_secs)

    ref_label = _safe_ref_name(_RCCL_TESTS_REF)
    build_flavour = "mpi" if mpi_enabled else "nompi"
    monorepo_dest = pathlib.Path(compiler_build_dir) / "rccl" / f"rocm-systems-{ref_label}-{build_flavour}"
    rccl_tests_dir = external_build.clone_repo(
        url=_RCCL_TESTS_MONOREPO_URL,
        dest=monorepo_dest,
        ref=_RCCL_TESTS_REF,
        timeout=build_timeout,
        sparse_subtree=_RCCL_TESTS_SUBPATH,
    )

    build_dir = rccl_tests_dir / "build"

    # License check (open-source provenance guard).
    external_build.assert_license_present(rccl_tests_dir)

    # Build step: rccl-tests reads ROCM_PATH/HIP_HOME and links RCCL from NCCL_HOME/RCCL_HOME.
    make_args = [
        f"MPI={1 if mpi_enabled else 0}",
        f"ROCM_PATH={rocm_path}",
        f"HIP_HOME={rocm_path}",
        f"NCCL_HOME={rocm_path}",
        f"RCCL_HOME={rocm_path}",
    ]

    if cmake_executor is not None:
        # clone_repo() returns an absolute path in the managed remote workspace.
        abs_build_dir = str(build_dir)

        # Idempotency check: probe a specific well-known binary rather than globbing.
        sentinel_binary = f"{abs_build_dir}/all_reduce_perf"
        existing = cmake_executor.run(f"test -f {sentinel_binary}", timeout=15.0)
        if existing.ok:
            logger.info("rccl-tests build (remote): binaries exist in %s — skipping make", abs_build_dir)
        else:
            try:
                external_build.make_build(
                    rccl_tests_dir,
                    make_args=make_args,
                    env=mpi_runtime.env if mpi_runtime else None,
                    timeout=build_timeout,
                )
            except RuntimeError as exc:
                _skip_known_rccl_tests_incompatibility(exc)

        # Verify binaries exist and collect their absolute paths for the return value.
        ls_result = cmake_executor.run(f"ls {abs_build_dir}/*_perf", timeout=15.0)
        perf_binaries = [line.strip() for line in ls_result.stdout.splitlines() if line.strip()]
        assert perf_binaries, f"rccl-tests build produced no *_perf binaries in {abs_build_dir} on remote"
        logger.info("rccl-tests built %d perf binaries (remote) in %s", len(perf_binaries), abs_build_dir)
        return abs_build_dir

    # Local build.
    existing_binaries = glob.glob(str(build_dir / "*_perf"))
    if existing_binaries:
        logger.info("rccl-tests build (local): binaries exist in %s — skipping make", build_dir)
    else:
        try:
            external_build.make_build(
                rccl_tests_dir,
                make_args=make_args,
                env=mpi_runtime.env if mpi_runtime else None,
                timeout=build_timeout,
            )
        except RuntimeError as exc:
            _skip_known_rccl_tests_incompatibility(exc)
    perf_binaries = glob.glob(str(build_dir / "*_perf"))
    assert perf_binaries, f"rccl-tests build produced no *_perf binaries in {build_dir}"
    logger.info("rccl-tests built %d perf binaries in %s", len(perf_binaries), build_dir)
    return str(build_dir.resolve())


def _skip_known_rccl_tests_incompatibility(exc: RuntimeError) -> None:
    """Skip when the selected rccl-tests source is newer than installed headers."""
    detail = str(exc)
    if "ncclCommProperties_t" in detail:
        pytest.skip(
            "rccl-tests source is incompatible with the installed RCCL/NCCL headers: "
            "the source references ncclCommProperties_t, but this ROCm build does not "
            "provide that type. Set RCCL_TESTS_REF to a matching rccl-tests ref, "
            "or update the RCCL install."
        )
    raise exc


@pytest.fixture(scope="session")
def rccl_tests_build(rock_dir: str, compiler_build_dir: str, framework_config, external_build, cmake_executor) -> str:
    """Build non-MPI ``rccl-tests`` clients for single-process ``-g N`` tests."""
    return _build_rccl_tests(
        rock_dir=rock_dir,
        compiler_build_dir=compiler_build_dir,
        framework_config=framework_config,
        external_build=external_build,
        cmake_executor=cmake_executor,
        mpi_enabled=False,
    )


@pytest.fixture(scope="session")
def rccl_tests_mpi_build(
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    mpi_runtime,
) -> str:
    """Build MPI-enabled ``rccl-tests`` clients for legacy ``mpirun`` RCCL_TESTS mode."""
    return _build_rccl_tests(
        rock_dir=rock_dir,
        compiler_build_dir=compiler_build_dir,
        framework_config=framework_config,
        external_build=external_build,
        cmake_executor=cmake_executor,
        mpi_enabled=True,
        mpi_runtime=mpi_runtime,
    )


@pytest.fixture(scope="session")
def rccl_perf_binaries(rccl_tests_build: str, cmake_executor) -> list[str]:
    """Return sorted ``*_perf`` binaries from local or remote rccl-tests builds."""
    if cmake_executor is not None:
        ls_result = cmake_executor.run(f"ls {rccl_tests_build}/*_perf", timeout=15.0)
        return sorted(line.strip() for line in ls_result.stdout.splitlines() if line.strip())
    return sorted(glob.glob(os.path.join(rccl_tests_build, "*_perf")))


@pytest.fixture(scope="session")
def rccl_error_handling_binary(compile_binary, rock_dir: str) -> str:
    """Compile the first-party RCCL error-handling stub."""
    return compile_binary(
        src=_STUB_SRC,
        output_name="rccl_error_handling",
        std="c++17",
        opt="-O2",
        include_dirs=["tests/e2e/rccl/src/rccl_error_handling"],
        extra_flags=[
            "-Wall",
            "-D__HIP_PLATFORM_AMD__",
            "-isystem",
            f"{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-lrccl",
            "-lamdhip64",
        ],
        subdir="rccl",
    )


@pytest.fixture(scope="session")
def rccl_dual_kernel_build_binary(compile_binary, rock_dir: str) -> str:
    """Compile the dual-kernel validation stub; it uses dlopen, so link ``-ldl``."""
    return compile_binary(
        src=_DUAL_KERNEL_SRC,
        output_name="rccl_dual_kernel_build_test",
        std="c++17",
        opt="-O2",
        include_dirs=["tests/e2e/rccl/src/rccl_unroll_test"],
        extra_flags=[
            "-Wall",
            "-D__HIP_PLATFORM_AMD__",
            "-isystem",
            f"{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-lamdhip64",
            "-ldl",
        ],
        subdir="rccl",
    )


@pytest.fixture(scope="session")
def rccl_unroll_perf_matrix_binary(compile_binary, rock_dir: str) -> str:
    """Compile the unroll-factor matrix stub; HIP runtime only, no RCCL link."""
    return compile_binary(
        src=_UNROLL_PERF_MATRIX_SRC,
        output_name="rccl_unroll_perf_matrix_test",
        std="c++17",
        opt="-O2",
        include_dirs=["tests/e2e/rccl/src/rccl_unroll_test"],
        extra_flags=[
            "-Wall",
            "-D__HIP_PLATFORM_AMD__",
            "-isystem",
            f"{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-lamdhip64",
        ],
        subdir="rccl",
    )


@pytest.fixture(scope="session")
def concurrent_collectives_binary(compile_binary, rock_dir: str) -> str:
    """Compile the concurrent-collectives stress harness."""
    return compile_binary(
        src=_CONCURRENT_COLLECTIVES_SRC,
        output_name="concurrent_collectives",
        std="c++17",
        opt="-O3",
        include_dirs=["tests/e2e/rccl/src/concurrent_collectives"],
        extra_flags=[
            "-Wall",
            "-Wno-unused-result",
            "-D__HIP_PLATFORM_AMD__",
            "-isystem",
            f"{rock_dir}/include",
            f"-L{rock_dir}/lib",
            "-lrccl",
            "-lpthread",
            "-lamdhip64",
        ],
        subdir="rccl",
    )


@pytest.fixture
def require_rccl(rock_dir: str, cmake_executor):
    """Skip when ``librccl`` is absent from the ROCm install."""
    if cmake_executor is not None:
        # Probe for the unversioned symlink; use ls glob via double-quoted bash -c so the
        # SSH executor's outer single-quote wrapping does not conflict with nested quotes.
        probe_cmd = (
            f"ls {rock_dir}/lib/librccl.so 2>/dev/null || " f"ls {rock_dir}/lib/librccl*.so* 2>/dev/null | head -1"
        )
        result = cmake_executor.run(
            f'bash -c "{probe_cmd}"',
            timeout=15.0,
        )
        if not result.ok or not result.stdout.strip():
            pytest.skip(f"librccl not found under {rock_dir}/lib on remote — RCCL not installed")
    else:
        if not glob.glob(os.path.join(rock_dir, "lib", "librccl*.so*")):
            pytest.skip(f"librccl not found under {rock_dir}/lib — RCCL not installed")


@pytest.fixture(scope="session")
def mpi_runtime(external_build, cmake_executor):
    """Return MPI launcher/env metadata, provisioning local OpenMPI if needed."""
    _ensure_openssh_client(cmake_executor)
    runtime = external_build.detect_mpi_runtime()
    if runtime is not None:
        return runtime

    version = os.environ.get("ROCM_TEST_RCCL_OPENMPI_VERSION") or os.environ.get("OPENMPI_VERSION") or "4.1.4"
    logger.info("MPI runtime not found; provisioning OpenMPI %s for RCCL tests", version)
    return external_build.provision_openmpi_runtime(version=version)


@pytest.fixture(scope="session")
def require_mpirun(mpi_runtime):
    """Backward-compatible prerequisite fixture for MPI-mode tests."""
    return mpi_runtime


# requested_gpu_count is provided by the shared suite-level conftest (tests/conftest.py).
