# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MPI runtime helpers shared by HPC test suites.

Provides distro-aware OpenSSH client installation, MPI runtime detection and
auto-provisioning, HPC environment-prefix construction, and an MPI launch
assertion. Shared across any suite that drives benchmarks via ``mpirun``
(rocHPL, RCCL-tests MPI mode, etc.).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess  # nosec B404

import pytest

logger = logging.getLogger(__name__)


def ensure_ssh_client(executor=None) -> None:
    """Install ``openssh-client`` when ``ssh`` is absent on the execution host.

    MPI's default ``rsh`` launcher requires ``ssh`` on the path. This function
    detects the first available package manager and installs the client package
    using a hard-coded allowlist of distro package manager commands — no shell
    expansion or user-supplied strings are passed to the subprocess.

    Args:
        executor: Remote executor; ``None`` means local.
    """
    if executor is not None:
        probe = executor.run("command -v ssh", timeout=15.0)
        if probe.ok and probe.stdout.strip():
            return
        logger.info("Installing OpenSSH client for MPI")
        result = executor.run(_remote_openssh_install_cmd(), timeout=300.0)
        if not result.ok:
            pytest.fail(
                "MPI needs ssh; OpenSSH client install failed on remote.\n"
                f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}"
            )
        return

    if shutil.which("ssh"):
        return

    install_cmds = _local_openssh_install_cmds()
    if install_cmds is None:
        pytest.fail("MPI needs ssh; no supported package manager found to install OpenSSH client.")

    logger.info("Installing OpenSSH client for MPI")
    ok = True
    stdout = stderr = ""
    for cmd in install_cmds:
        ok, stdout, stderr = _run_local_cmd(cmd, timeout=300.0)
        if not ok:
            break

    if not ok:
        pytest.fail(
            "MPI needs ssh; OpenSSH client install failed.\n" f"stdout: {stdout[-1000:]}\nstderr: {stderr[-1000:]}"
        )


def ensure_mpi_runtime(external_build, executor=None, version: str = "4.1.4"):
    """Detect an existing MPI runtime; provision OpenMPI when none is found.

    Discovery probes for a system ``mpirun`` and known OpenMPI prefix locations.
    When no runtime is found, provisions a private OpenMPI at *version* under the
    framework build directory so the test never depends on global package state.

    Args:
        external_build: Framework external-build helper.
        executor:       Remote executor; ``None`` means local.
        version:        OpenMPI version to provision when absent (default ``"4.1.4"``).

    Returns:
        An ``MpiRuntime`` dataclass with ``.launcher`` and ``.env`` attributes.
    """
    ensure_ssh_client(executor)
    runtime = external_build.detect_mpi_runtime()
    if runtime is not None:
        logger.info("MPI: using discovered runtime at %s", runtime.launcher)
        return runtime

    logger.info("MPI: no runtime found; provisioning OpenMPI %s", version)
    return external_build.provision_openmpi_runtime(version=version)


def build_hpc_env_prefix(rock_dir: str, mpi_runtime, ld_library_path: str = "") -> str:
    """Construct the ``env VAR=...`` shell prefix for HPC benchmark invocations.

    Injects ``ROCM_PATH``, ``LD_LIBRARY_PATH`` (combining MPI and ROCm libraries),
    ``MPI_HOME``, and ``PATH`` (MPI bin dir prepended) so that HPC binaries
    resolve both the ROCm runtime and MPI launcher at invocation time without
    modifying ``os.environ``.

    Args:
        rock_dir:          ROCm install root.
        mpi_runtime:       ``MpiRuntime`` returned by :func:`ensure_mpi_runtime`.
        ld_library_path:   Existing ``LD_LIBRARY_PATH`` value to extend (may be empty).

    Returns:
        Shell ``env VAR=value ...`` prefix string (no trailing space).
    """
    mpi_home = mpi_runtime.env.get("MPI_HOME", "")
    mpi_bin = os.path.dirname(mpi_runtime.launcher)
    mpi_lib = mpi_runtime.env.get("LD_LIBRARY_PATH", "")

    lib_parts = [p for p in (mpi_lib, f"{rock_dir}/lib", ld_library_path) if p]
    combined_ld = ":".join(lib_parts)

    parts = [
        f"ROCM_PATH={rock_dir}",
        f"PATH={mpi_bin}:{rock_dir}/bin:$PATH",
        f"LD_LIBRARY_PATH={combined_ld}:$LD_LIBRARY_PATH",
    ]
    if mpi_home:
        parts.insert(0, f"MPI_HOME={mpi_home}")

    return "env " + " ".join(parts)


def assert_mpi_launch_ok(result, label: str) -> None:
    """Assert that an ``mpirun`` invocation exited cleanly.

    Checks for a zero exit code and the absence of known MPI initialisation
    failure signatures (``MPI_Init``, rank-0 abort, and similar). A non-zero
    exit alone is insufficient: some MPI implementations emit the abort message
    before killing ranks, so the process may technically exit 1 or 139 without
    a meaningful MPI error in the output.

    Args:
        result: Execution result with ``.ok``, ``.stdout``, ``.stderr``, ``.exit_code``.
        label:  Human-readable test label for the failure message.

    Raises:
        AssertionError: When the launch is determined to have failed.
    """
    combined = f"{result.stdout or ''}\n{result.stderr or ''}"
    mpi_fatal_markers = (
        "MPI_Init",
        "MPI_INIT",
        "Fatal error in MPI",
        "rank 0 in job",
        "aborting MPI_COMM_WORLD",
        "An error occurred in MPI_Init",
    )
    mpi_errors = [m for m in mpi_fatal_markers if m in combined]
    assert result.ok, (
        f"{label} MPI launch failed (exit={result.exit_code}).\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-1000:]}"
    )
    assert not mpi_errors, (
        f"{label} MPI launch reported fatal MPI errors: {mpi_errors}\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-1000:]}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _remote_openssh_install_cmd() -> str:
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


def _local_openssh_install_cmds() -> list[list[str]] | None:
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


def _run_local_cmd(command: list[str], timeout: float) -> tuple[bool, str, str]:
    # command is selected from a hard-coded package-manager allowlist.
    result = subprocess.run(  # nosec B603
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return result.returncode == 0, result.stdout, result.stderr
