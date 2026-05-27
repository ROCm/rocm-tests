# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
executor_plugin.py -- Pytest fixtures for CPU, container, SSH, and session executors.

Every executor in ``framework.executors`` is accessible through a named fixture
defined here.  Tests never import executor classes directly — they declare the
fixture as a parameter and receive a ready-to-use executor.

FIXTURES
--------
cpu_executor
    ``CpuExecutor`` — real subprocess on the local host with no GPU environment
    modifications.  For ``@pytest.mark.hw.cpu_only`` tests.

session_executor
    ``ExecutorFactory``-selected executor.  Reads ``--no-gpu``,
    ``--container-mode``, and related CLI flags to return the correct backend
    (``LocalExecutor``, ``ContainerExecutor``, or ``DryRunExecutor``) without
    any changes to test code.  For new tests, prefer ``target_executor`` from
    ``remote_node_plugin`` which supports both local and remote GPU execution.

remote_pool
    ``RemoteNodePool`` — manages a registry of ``SshExecutor`` sessions keyed
    by ``user@host:port``.  Requesting the same host twice returns the *same*
    ``SshExecutor`` instance (connection reuse).  All sessions are closed when
    the test function exits.  Replaces the concept of a simple ``ssh_node``
    factory fixture: the pool is a first-class object that a test can pass
    between helper functions at any step.

container_executor
    ``ContainerExecutor`` with AMD GPU passthrough.  Reads the target image
    from the ``@pytest.mark.container_image`` marker or ``--container-image``
    CLI option.  Exposes ``probe()``, ``run()``, and ``exec_in()`` so tests
    can check runtime health and execute commands in the same fixture.

NOTE: ``local_executor`` has been removed.  Use ``target_executor`` (from
``remote_node_plugin``) for GPU tests — it transparently selects
``LocalExecutor`` or ``SshGpuExecutor`` depending on ``--remote-node``.

ADDITIONAL CLI OPTIONS (registered here)
-----------------------------------------
    --container-mode      Activate ContainerExecutor via ``session_executor``.
    --container-image     Image to use (default: rocm/pytorch:latest).
    --container-runtime   "docker" or "podman" (default: docker).

Loaded automatically via ``pytest_plugins`` in ``conftest.py``.
"""

from __future__ import annotations

from collections.abc import Generator
import logging

import pytest

from framework.common.helpers import executor_log_path
from framework.executors.container_executor import ContainerExecutor
from framework.executors.cpu_executor import CpuExecutor
from framework.executors.dry_run_executor import DryRunExecutor
from framework.executors.executor_factory import ExecutorFactory
from framework.executors.ssh_executor import SshExecutor
from framework.gpu.allocator import GpuAllocator
from framework.gpu.detector import GpuDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI option registration
# ---------------------------------------------------------------------------


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
        help=(
            "Route GPU test commands through a Docker/Podman container.  "
            "The session_executor fixture will return a ContainerExecutor "
            "instead of the default LocalExecutor."
        ),
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


# ---------------------------------------------------------------------------
# RemoteNodePool — backing class for the remote_pool fixture
# ---------------------------------------------------------------------------


class RemoteNodePool:
    """Registry of ``SshExecutor`` sessions keyed by ``user@host:port``.

    Deduplicates paramiko connections when the same host is requested more
    than once within a single test function.  All registered sessions are
    closed when the ``remote_pool`` fixture tears down.

    This replaces the pattern of passing a simple factory function (like
    ``ssh_node``) into tests.  A pool is a first-class object that a test
    can pass to helper functions at any step, inspect its active sessions,
    and hand off to parametrized sub-fixtures — none of which is possible
    with a bare closure.

    Usage::

        @pytest.mark.hw.multi_gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.runtime
        def test_rccl_two_nodes(remote_pool):
            node_a = remote_pool.acquire("gpu-node-01", key_path="~/.ssh/ci_rsa")
            node_b = remote_pool.acquire("gpu-node-02", key_path="~/.ssh/ci_rsa")

            # Acquiring the same host again reuses the open connection:
            node_a2 = remote_pool.acquire("gpu-node-01", key_path="~/.ssh/ci_rsa")
            assert node_a is node_a2

            r_a = node_a.run("rocm-smi --showid")
            r_b = node_b.run("rocm-smi --showid")
            assert r_a.ok and r_b.ok

        # All SSH sessions are closed automatically after the test exits.
    """

    def __init__(self) -> None:
        self._registry: dict[str, SshExecutor] = {}

    def acquire(
        self,
        host: str,
        user: str | None = None,
        key_path: str | None = None,
        password: str | None = None,
        port: int = 22,
    ) -> SshExecutor:
        """Return the ``SshExecutor`` for *host*, registering it if first seen.

        The first call for a given ``user@host:port`` combination constructs
        an ``SshExecutor`` and stores it in the pool.  Subsequent calls with
        the same arguments return the same instance — no second TCP connection
        is opened.

        The SSH connection itself is still lazy: it opens only when ``run()``
        is first called on the executor.

        Args:
            host:     Remote hostname or IP address.
            user:     SSH login name (default: ``$USER``).
            key_path: Path to SSH private key file (``~`` is expanded).
            password: SSH password — prefer *key_path* for automated pipelines.
            port:     SSH server port (default 22).

        Returns:
            An ``SshExecutor`` whose ``session_key`` matches
            ``"user@host:port"``.
        """
        # Build a temporary executor just to obtain the canonical session_key
        # (avoids duplicating the user/port defaulting logic here).
        probe = SshExecutor(
            host=host,
            user=user,
            key_path=key_path,
            password=password,
            port=port,
        )
        key = probe.session_key

        if key not in self._registry:
            logger.debug("RemoteNodePool: registering new session %s", key)
            self._registry[key] = probe
        else:
            logger.debug("RemoteNodePool: reusing existing session %s", key)

        return self._registry[key]

    def release_all(self) -> None:
        """Close every SSH session tracked by this pool.

        Called automatically by the ``remote_pool`` fixture on teardown.
        Safe to call when no sessions have been opened.
        """
        for key, executor in list(self._registry.items()):
            logger.debug("RemoteNodePool: closing session %s", key)
            executor.close()
        self._registry.clear()

    @property
    def active_sessions(self) -> list:
        """Session keys of all currently registered nodes.

        A session key has the form ``"user@host:port"``.  Useful for
        diagnostic logging inside test helper functions.

        Returns:
            List of session key strings (order reflects registration order).
        """
        return list(self._registry.keys())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def session_executor(request, framework_config):
    """Provide the executor selected by the active pytest configuration.

    Delegates to ``ExecutorFactory.resolve()`` which reads ``--no-gpu``,
    ``--container-mode``, and related CLI flags to decide which backend
    to return.  When GPU hardware is needed, a GPU is acquired from
    ``GpuAllocator`` (NUMA-aware, arch-filtered) and released on teardown.
    Tests run transparently against a local GPU, a container, or a dry-run
    stub — no test code changes needed.

    +--------------------------------+-------------------------------+
    | pytest invocation              | executor returned             |
    +================================+===============================+
    | (default, GPU present)         | LocalExecutor(gpu_index=N)   |
    +--------------------------------+-------------------------------+
    | ``--no-gpu``                   | DryRunExecutor()             |
    +--------------------------------+-------------------------------+
    | ``--container-mode``           | ContainerExecutor(image=...) |
    +--------------------------------+-------------------------------+

    Yields:
        An ``AbstractExecutor`` instance backed by the correctly-allocated GPU.

    Example::

        @pytest.mark.hw.gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.runtime
        def test_hip_device_available(session_executor):
            result = session_executor.run(
                "python3 -c 'import torch; print(torch.cuda.is_available())'"
            )
            assert result.ok
            assert "True" in result.stdout
    """
    config = request.config

    # --no-gpu: return a synthetic executor without touching the GPU pool.
    if config.getoption("--no-gpu", default=False):
        yield DryRunExecutor()
        return

    # All other modes: allocate a real GPU from the pool.
    detector = getattr(config, "_gpu_detector", None) or GpuDetector()
    allocator = GpuAllocator(detector=detector)
    gpu_info = allocator.allocate()

    log_path = executor_log_path(framework_config.framework.artifact_dir, request.node.name, request.node.nodeid)

    try:
        yield ExecutorFactory.resolve(
            request,
            gpu_index=gpu_info.index,
            log_path=log_path,
        )
    finally:
        allocator.release(gpu_info)


@pytest.fixture
def remote_pool() -> Generator[RemoteNodePool, None, None]:
    """Provide a ``RemoteNodePool`` for multi-node SSH tests.

    The pool manages ``SshExecutor`` sessions keyed by ``user@host:port``.
    Requesting the same host twice returns the same executor instance,
    avoiding duplicate TCP connections.  All sessions are closed
    automatically when the test function exits.

    This fixture is the intended entry point for any test that needs SSH
    access to remote nodes.  It replaces the simpler ``ssh_node`` factory
    pattern with a stateful pool object that tests can pass to helper
    functions at any step.

    Yields:
        ``RemoteNodePool`` with ``acquire(host, ...)``, ``release_all()``,
        and ``active_sessions``.

    Example::

        @pytest.mark.hw.multi_gpu
        @pytest.mark.ci.nightly
        @pytest.mark.layer.runtime
        @pytest.mark.e2e.multinode
        def test_rccl_allreduce(remote_pool):
            node_a = remote_pool.acquire("gpu-node-01", key_path="~/.ssh/ci_rsa")
            node_b = remote_pool.acquire("gpu-node-02", key_path="~/.ssh/ci_rsa")

            r_a = node_a.run("torchrun --nproc_per_node=2 allreduce.py")
            r_b = node_b.run("torchrun --nproc_per_node=2 allreduce.py")
            assert r_a.ok and r_b.ok
    """
    pool = RemoteNodePool()
    yield pool
    pool.release_all()


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
