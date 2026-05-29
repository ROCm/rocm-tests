# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
executor_plugin.py -- Pytest fixtures for CPU and container executors.

Fixtures: cpu_executor, container_executor.
For new tests use target_executor from remote_node_plugin.

CLI options added: --container-mode, --container-image, --container-runtime.
Loaded automatically via pytest_plugins in conftest.py.
"""

from __future__ import annotations

import logging

import pytest

from framework.common.helpers import executor_log_path
from framework.executors.container_executor import ContainerExecutor
from framework.executors.cpu_executor import CpuExecutor

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register executor-related CLI options.

    Options added here complement the GPU options in ``gpu_plugin`` and are
    intentionally kept separate so each plugin owns a focused set of flags.
    """
    group = parser.getgroup("rocm-executor", "ROCm executor options")
    group.addoption(
        "--container-mode",
        action="store_true",
        default=False,
        help="Route GPU test commands through a Docker/Podman container.",
    )
    group.addoption(
        "--container-image",
        action="store",
        default="rocm/pytorch:latest",
        metavar="IMAGE",
        help="Container image used when --container-mode is active.",
    )
    group.addoption(
        "--container-runtime",
        action="store",
        default="docker",
        choices=["docker", "podman"],
        metavar="RUNTIME",
        help="Container runtime binary to use (default: docker).",
    )


@pytest.fixture
def cpu_executor(framework_config, request) -> CpuExecutor:
    """Provide a ``CpuExecutor`` for ``hw.cpu_only`` tests.

    Runs real subprocess commands on the local host *without* setting
    ``HIP_VISIBLE_DEVICES`` or requiring a GPU allocation.  Use this for
    ROCm tool probes, compiler invocations, and config validation steps that
    do not touch GPU hardware.

    Streaming behaviour:
        STDERR is always written to ``sys.stderr`` in real time.  STDOUT is
        also streamed when ``ROCM_TEST_FRAMEWORK_LOG_LEVEL=debug`` (or
        ``[framework] log_level = "debug"`` in ``rocm-test.toml``).  Both
        channels are appended to ``output/artifacts/<test_dir>/<test>.log``.

    Returns:
        ``CpuExecutor`` ready for ``.run()`` calls.

    Example::

        @pytest.mark.hw.cpu_only
        @pytest.mark.ci.pr
        @pytest.mark.layer.runtime
        @pytest.mark.runtime.fast
        def test_rocm_smi_version(cpu_executor):
            result = cpu_executor.run("rocm-smi --version")
            assert result.ok
            assert "ROCm" in result.stdout
    """
    log_path = executor_log_path(framework_config.framework.artifact_dir, request.node.name, request.node.nodeid)
    session_log = getattr(request.config, "_session_log_path", None)
    if session_log is None:
        import pathlib as _pathlib

        session_log = str(_pathlib.Path(framework_config.framework.artifact_dir) / "session.log")
        _pathlib.Path(session_log).parent.mkdir(parents=True, exist_ok=True)
    return CpuExecutor(log_path=log_path, session_log_path=session_log)


@pytest.fixture
def container_executor(request) -> ContainerExecutor:
    """Provide a ``ContainerExecutor`` with AMD GPU device passthrough.

    The target image is resolved from (in priority order):
        1. The ``@pytest.mark.container_image("<image>")`` marker on the test.
        2. The ``--container-image`` CLI option (default: ``rocm/pytorch:latest``).

    The container runtime is resolved from ``--container-runtime``
    (default: ``"docker"``).

    Exposes three methods for use at any step of the test:
        - ``probe()``              — inspect daemon/device health → ``ContainerStatus``
        - ``run(command)``         — one-shot container execution
        - ``exec_in(name, cmd)``   — command in a named running container

    Returns:
        ``ContainerExecutor`` configured with the resolved image and runtime.

    Example::

        @pytest.mark.hw.gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.ml_framework
        @pytest.mark.container_image("rocm/pytorch:6.3")
        def test_pytorch_hip_version(container_executor):
            status = container_executor.probe()
            assert status.ready, f"Container env not ready: {status.errors}"

            result = container_executor.run(
                "python3 -c 'import torch; print(torch.version.hip)'"
            )
            assert result.ok
            assert result.stdout.strip() != ""
    """
    marker = request.node.get_closest_marker("container_image")
    image = marker.args[0] if marker else request.config.getoption("--container-image", default="rocm/pytorch:latest")
    runtime = request.config.getoption("--container-runtime", default="docker")
    return ContainerExecutor(image=image, runtime=runtime)
