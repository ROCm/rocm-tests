# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build and environment fixtures for the CRIU checkpoint/restore suite.

Provides the CRIU runtime prefix (host and in-target) and the
PyTorch MNIST checkout, grouped into the COMMON / MNIST sections below.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

import pytest
from tests.common.criu import ensure_criu_runtime, ensure_criu_runtime_target

logger = logging.getLogger(__name__)


# ###########################################################################
# #### COMMON #### -- CRIU runtime prefix (host and in-target)
# ###########################################################################


@pytest.fixture(scope="session")
def criu_runtime(external_build, cmake_executor, framework_config) -> str:
    """Make CRIU + amdgpu_plugin ready on the test node (host/SSH); auto-install if not.

    Session-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return ensure_criu_runtime(external_build, cmake_executor, framework_config)


@pytest.fixture
def criu_runtime_target(target_executor, framework_config) -> str:
    """Make CRIU ready *inside* ``target_executor`` so it lives where the workload runs.

    Function-scoped. Returns the ``sudo -n ... criu`` command prefix.
    """
    return ensure_criu_runtime_target(target_executor, framework_config)


# ###########################################################################
# #### MNIST #### -- pytorch/examples MNIST setup
# Upstream pytorch/examples (BSD-3-Clause); cloned at runtime, not vendored. See NOTICES.md.
# ###########################################################################

_PYT_EXAMPLES_URL = os.environ.get(
    "ROCM_TEST_PYT_EXAMPLES_URL",
    "https://github.com/pytorch/examples.git",
)
# Writable checkout dir inside the target; the workload runs in target_executor, not on the host.
_PYT_WORKDIR = os.environ.get("ROCM_TEST_PYT_WORKDIR", "/tmp/rocm-tests/pyt_examples")

# Interpreters probed for ROCm (HIP) PyTorch, in order; the ambient torch is used as-is.
_PYTHON_CANDIDATES = ("python3", "python", "/opt/venv/bin/python3", "/opt/conda/bin/python3")


@dataclass(frozen=True)
class PytMnistSetup:
    """MNIST checkout: ``workdir`` (examples/mnist in the target) and the ROCm ``python``."""

    workdir: str
    python: str


def _detect_rocm_python(probe_exec) -> str | None:
    """Return the first interpreter whose ``torch.version.hip`` is truthy, else None.

    Probes ``ROCM_TEST_MNIST_PYTHON`` then common interpreters, running from ``/tmp`` so a pytorch
    source checkout on the default WORKDIR cannot shadow the installed torch on ``sys.path``.
    """
    override = os.environ.get("ROCM_TEST_MNIST_PYTHON")
    candidates = [override] if override else list(_PYTHON_CANDIDATES)
    probe = "import torch,sys; sys.exit(0 if getattr(torch.version,'hip',None) else 1)"
    for interp in candidates:
        if not interp or not probe_exec.run(f"command -v {interp} >/dev/null 2>&1").ok:
            continue
        if probe_exec.run(f'cd /tmp && {interp} -c "{probe}"').ok:
            return interp
    return None


@pytest.fixture
def pyt_mnist_setup(target_executor, framework_config) -> PytMnistSetup:
    """Clone pytorch/examples inside ``target_executor`` and resolve the ambient ROCm python.

    Git-clone only (ambient ROCm PyTorch is used as-is). Skips when git or a ROCm-torch python is absent.
    """
    ex = target_executor
    if "GIT_OK" not in (ex.run("command -v git >/dev/null 2>&1 && echo GIT_OK").stdout or ""):
        pytest.skip("git is not available in the target environment -- cannot clone pytorch/examples.")

    python = _detect_rocm_python(ex)
    if not python:
        pytest.skip(
            "No python interpreter with ROCm (HIP) PyTorch found -- run this test inside a ROCm "
            "PyTorch container (set ROCM_TEST_MNIST_PYTHON to point at the interpreter)."
        )

    # POSIX paths inside the target (never os.path.join -- the pytest host may be Windows).
    clone_dir = f"{_PYT_WORKDIR}/examples"
    workdir = f"{clone_dir}/mnist"

    script = "\n".join(
        (
            "set -e",
            f"mkdir -p {_PYT_WORKDIR}",
            f"rm -rf {clone_dir}",
            f"git clone --depth 1 {_PYT_EXAMPLES_URL} {clone_dir}",
            f"test -f {workdir}/main.py && echo CLONE_OK",
        )
    )
    logger.info("Cloning pytorch/examples into %s", clone_dir)
    result = ex.run(script, timeout=float(framework_config.therock.build_timeout_secs))
    if "CLONE_OK" not in (result.stdout or ""):
        pytest.fail(
            f"pytorch/examples clone failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return PytMnistSetup(workdir=workdir, python=python)


# #### END MNIST ####
