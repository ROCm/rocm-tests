# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared CRIU runtime helpers.

Two entry points make CRIU + the amdgpu plugin ready and return the ``sudo -n env PATH=... criu``
command prefix:

- :func:`ensure_criu_runtime` -- host/SSH path (session-scoped suites). Clones CRIU via
  ``external_build.clone_repo`` and builds it with :mod:`tests.common.criu.installer`.
- :func:`ensure_criu_runtime_target` -- provisions CRIU *inside* ``target_executor`` so a
  ``@pytest.mark.container`` workload has CRIU where it runs; the installer self-clones there
  (``external_build.clone_repo`` cannot reach into a container filesystem).

Both use CRIU as-is when ``criu check`` reports "Looks good" and the plugin is present. CRIU needs
root; passwordless ``sudo -n`` is required or the suite skips cleanly.
"""

from __future__ import annotations

import base64
import logging
import os
import re

import pytest

from framework.common.workspace_layout import REMOTE_WORKSPACE_DIR
from framework.executors.cpu_executor import CpuExecutor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CRIU invocation prefix
# ---------------------------------------------------------------------------

# The installer places criu under /usr/local/sbin. sudo resets the environment, so PATH is set
# explicitly for the elevated process. LD_LIBRARY_PATH is forwarded too: in a ROCm container criu
# links the vendored libnl (librocm_sysdeps_nl_3.so under /opt/rocm/lib), which the loader only
# finds via LD_LIBRARY_PATH. $LD_LIBRARY_PATH expands in the target shell (empty on bare metal,
# which is harmless), and passing it as an env arg survives sudo's environment stripping.
_CRIU_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin"
CRIU = f'sudo -n env "PATH={_CRIU_PATH}" "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" criu'

# Path where the amdgpu CRIU plugin is installed.
_AMDGPU_PLUGIN = "/usr/lib/criu/amdgpu_plugin.so"

# CRIU upstream + git tag to build when auto-installing.
_CRIU_REPO_URL = os.environ.get("CRIU_REPO", "https://github.com/checkpoint-restore/criu.git")
_DEFAULT_CRIU_VERSION = "v4.1"
# Shell-safe git ref (tag/branch) pattern -- validated before interpolation.
_CRIU_VERSION_RE = re.compile(r"^[A-Za-z0-9._/\-]+$")

# The installer is transferred to a persistent path under the managed ``run-rocm-tests`` workspace
# so it works across baremetal (local/SSH) and container executors alike.
_INSTALLER_NAME = "rocm_test_criu_installer.py"

_INSTALL_HINT = (
    "CRIU + amdgpu_plugin is required by this suite. When missing it is auto-installed on the "
    "test node via tests/common/criu (installer.py) -- set ROCM_TEST_CRIU_AUTO_INSTALL=0 to "
    "disable, ROCM_TEST_CRIU_VERSION=<tag> to pin the version. Auto-install needs passwordless "
    "sudo, a C toolchain, and network access; you can also run tests/common/criu/installer.py "
    "manually on the node beforehand."
)


def _criu_ready(probe_exec) -> tuple[bool, str]:
    """Return ``(ready, diagnostic)``: ready when the amdgpu plugin exists and ``criu check`` says "Looks good"."""
    if not probe_exec.run(f"test -f {_AMDGPU_PLUGIN}").ok:
        return False, f"amdgpu plugin not found at {_AMDGPU_PLUGIN}"
    check = probe_exec.run(f"{CRIU} check")
    combined = f"{check.stdout}\n{check.stderr}"
    if "Looks good" not in combined:
        return False, f"`criu check` did not report 'Looks good':\n{combined[-1500:]}"
    return True, ""


def _installer_dest(probe_exec) -> str:
    """Return a persistent workspace path for the transferred installer (baremetal + container).

    Uses the executor's managed workspace when available (SSH resolves the remote home); otherwise a
    ``$HOME/run-rocm-tests/...`` path expanded in the target shell (local baremetal / container).
    """
    if hasattr(probe_exec, "workspace_path_for"):
        return str(probe_exec.workspace_path_for(_INSTALLER_NAME, category="generated"))
    return f"$HOME/{REMOTE_WORKSPACE_DIR}/output/generated/{_INSTALLER_NAME}"


def _ship_installer(probe_exec):
    """Base64-transfer installer.py to a persistent workspace path in the remote/target environment.

    Returns ``(remote_path, None)`` on success or ``(None, transfer_result)`` on failure, so the
    caller can surface the failed ExecutionResult.
    """
    installer_src = os.path.join(os.path.dirname(__file__), "installer.py")
    with open(installer_src, "rb") as handle:
        payload = base64.b64encode(handle.read()).decode()
    dest = _installer_dest(probe_exec)
    # Create the workspace dir if absent; $HOME expands in the target shell.
    transfer = probe_exec.run(f'mkdir -p "$(dirname "{dest}")" && echo {payload} | base64 -d > "{dest}"')
    if not transfer.ok:
        return None, transfer
    return dest, None


def _install_host(external_build, probe_exec, is_remote: bool, version: str, timeout: float):
    """Host/SSH path: clone CRIU via ``external_build.clone_repo`` and build from that checkout.

    The clone needs no privileges; the installer performs the elevated build/install steps. Remote
    runs transfer the installer module into the SSH node first (the repo may not be checked out
    there); local runs execute it in place.
    """
    clone_path = external_build.clone_repo(_CRIU_REPO_URL, f"criu/criu-{version}", ref=version, timeout=timeout)
    src_dir = str(clone_path) if is_remote else os.path.realpath(str(clone_path))
    if is_remote:
        script_path, failure = _ship_installer(probe_exec)
        if failure is not None:
            return failure
    else:
        script_path = os.path.join(os.path.dirname(__file__), "installer.py")
    return probe_exec.run(f'python3 "{script_path}" --src-dir {src_dir} {version}', timeout=timeout)


def _install_target(probe_exec, version: str, timeout: float):
    """Container/target path: transfer the installer into the target; it self-clones and builds there.

    ``external_build.clone_repo`` runs on the host/SSH node, so it cannot place source inside a
    container. The installer's own clone (run inside the target) puts CRIU where the workload lives.
    """
    script_path, failure = _ship_installer(probe_exec)
    if failure is not None:
        return failure
    return probe_exec.run(f'python3 "{script_path}" {version}', timeout=timeout)


def _ensure(probe_exec, framework_config, install_fn) -> str:
    """Make CRIU ready via *probe_exec*, auto-installing with *install_fn* when missing.

    *install_fn* is ``(version, timeout) -> ExecutionResult`` and encapsulates whether the source is
    cloned on the host (host path) or inside the target (container path).
    """
    # Passwordless sudo is required to both install and run CRIU.
    if not probe_exec.run("sudo -n true").ok:
        pytest.skip("Passwordless sudo is not available for the test user. " + _INSTALL_HINT)

    ready, diagnostic = _criu_ready(probe_exec)
    if ready:
        logger.info("CRIU runtime available: amdgpu_plugin present and 'criu check' passed.")
        return CRIU

    if os.environ.get("ROCM_TEST_CRIU_AUTO_INSTALL", "1") == "0":
        pytest.skip(f"CRIU not ready ({diagnostic}) and auto-install disabled. " + _INSTALL_HINT)

    version = os.environ.get("ROCM_TEST_CRIU_VERSION", _DEFAULT_CRIU_VERSION)
    if not _CRIU_VERSION_RE.match(version):
        pytest.fail(f"Invalid ROCM_TEST_CRIU_VERSION {version!r}; expected a git tag/branch name.")

    logger.warning("CRIU not ready (%s). Auto-installing CRIU %s ...", diagnostic, version)
    install = install_fn(version, float(framework_config.therock.build_timeout_secs))
    if not install.ok:
        pytest.fail(
            f"Automatic CRIU installation failed (exit={install.exit_code}).\n"
            f"stdout: {install.stdout[-2000:]}\nstderr: {install.stderr[-2000:]}\n" + _INSTALL_HINT
        )

    ready, diagnostic = _criu_ready(probe_exec)
    if not ready:
        pytest.fail(f"CRIU still not ready after auto-installation ({diagnostic}). " + _INSTALL_HINT)

    logger.info("CRIU installed; runtime available.")
    return CRIU


def ensure_criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Ensure CRIU is ready on the host / SSH node; auto-install if missing. Returns the criu prefix.

    For bare-metal and ``--remote-node`` suites where the workload runs on the host. Disable install
    with ``ROCM_TEST_CRIU_AUTO_INSTALL=0``; pick the tag with ``ROCM_TEST_CRIU_VERSION`` (``v4.1``).
    """
    is_remote = cmake_executor is not None
    probe_exec = cmake_executor if is_remote else CpuExecutor(suppress_output_log=True)
    return _ensure(
        probe_exec,
        framework_config,
        lambda version, timeout: _install_host(external_build, probe_exec, is_remote, version, timeout),
    )


def ensure_criu_runtime_target(target_executor, framework_config) -> str:
    """Ensure CRIU is ready *inside* ``target_executor``; auto-install if missing. Returns the criu prefix.

    For ``@pytest.mark.container`` tests so CRIU lives where the workload runs: the installer is
    transferred into the target and self-clones/builds there.
    """
    return _ensure(
        target_executor,
        framework_config,
        lambda version, timeout: _install_target(target_executor, version, timeout),
    )
