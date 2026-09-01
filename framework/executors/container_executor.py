# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
container_executor.py -- Docker/Podman executor with AMD GPU device passthrough.

Provides three distinct operations:

    probe()    Inspect the container runtime environment on the host:
               daemon reachability, compose availability, AMD device presence,
               and user permissions.  Returns a ``ContainerStatus`` report so
               tests can make assertions or skip before spending time on pulls.

    run()      One-shot container execution (``docker run --rm``).
               AMD KFD and DRI devices are passed through automatically so
               ROCm workloads see the GPU inside the container.

    exec_in()  Execute a command inside an *already-running* named container.
               Use for long-lived containers started by a session-scoped fixture
               where creating a new container per command is too expensive.

Usage (via ``container_executor`` fixture — not instantiated directly in tests):
    status = container_executor.probe()
    assert status.ready, status.errors

    result = container_executor.run("python3 -c 'import torch; print(torch.version.hip)'")
    assert result.ok
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import logging
import os
import shlex

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.cpu_executor import CpuExecutor
from framework.os_adapter import os_adapter_factory

logger = logging.getLogger(__name__)

# Monotonic suffix for unique persistent-container names (Date/random are unavailable
# and would break reproducibility anyway; PID + counter is unique per session).
_NAME_COUNTER = itertools.count(1)


@dataclass
class ContainerStatus:
    """Snapshot of the container runtime environment on a host.

    Returned by ``ContainerExecutor.probe()``.  Tests should assert
    ``status.ready`` (or inspect individual fields) before running
    GPU workloads inside containers.

    Attributes:
        runtime_available:   Container runtime binary is installed.
        runtime_version:     Version string reported by the runtime binary.
        daemon_active:       Runtime daemon is reachable (``docker info`` succeeds).
        compose_available:   Docker Compose plugin or standalone binary is present.
        compose_version:     Version string reported by Compose.
        amd_devices_present: ``/dev/kfd`` and ``/dev/dri`` exist on the host,
                             which is required for AMD GPU passthrough.
        user_in_group:       Current user can run container commands without sudo.
        errors:              Accumulated human-readable error descriptions from
                             the probe — useful for skip messages.
    """

    runtime_available: bool = False
    runtime_version: str = ""
    daemon_active: bool = False
    compose_available: bool = False
    compose_version: str = ""
    amd_devices_present: bool = False
    user_in_group: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        """True when the runtime is fully operational for AMD GPU container workloads.

        All four conditions must hold: runtime installed, daemon reachable,
        AMD devices present, and the user has permission to run containers.
        """
        return self.runtime_available and self.daemon_active and self.amd_devices_present and self.user_in_group


class ContainerExecutor(AbstractExecutor):
    """Run commands via Docker or Podman with AMD GPU device passthrough.

    AMD GPU access inside containers requires two host devices:
        - ``/dev/kfd``  — KFD (Kernel Fusion Driver) character device.
        - ``/dev/dri``  — DRI render nodes (``renderD128``, etc.).

    Both are passed with ``--device`` flags and the container is added to the
    ``video`` group when ``use_amd_devices=True`` (the default).

    ``ROCR_VISIBLE_DEVICES`` is set inside the container to the value of
    *gpu_index* so ROCm sees the correct GPU even when multiple cards are
    present in the host.

    Two usage modes:
        * One-shot (default): each ``run()`` is an independent ``docker run --rm``.
        * Persistent: call ``start()`` once (``docker run -d ... sleep infinity``);
          every subsequent ``run()`` is a ``docker exec`` into that same container,
          so state (background processes, checkpoints) survives across calls. Call
          ``stop()`` to remove it. Used by ``target_executor`` for ``@pytest.mark.container``.

    Args:
        image:           Full container image reference (e.g. ``"rocm/pytorch:6.3"``).
        gpu_index:       AMD GPU ordinal injected as ``ROCR_VISIBLE_DEVICES`` inside
                         the container (default 0). Ignored when *gpu_indices* is set.
        gpu_indices:     Explicit list of GPU ordinals to expose as
                         ``ROCR_VISIBLE_DEVICES`` (multi-GPU); overrides *gpu_index*.
        runtime:         ``"docker"`` or ``"podman"`` (default ``"docker"``).
        use_amd_devices: Pass AMD KFD/DRI devices through to the container
                         (default ``True``).
        ipc:             Value for ``--ipc`` (e.g. ``"host"``); omitted when empty.
        privileged:      Add ``--privileged`` (required by CRIU checkpoint/restore).
        extra_run_flags: Additional flags forwarded verbatim to every
                         ``docker run`` invocation (e.g. ``"--network host"``).
        host_executor:   Executor that runs the docker CLI itself. Defaults to a
                         local ``CpuExecutor``; pass an ``SshExecutor`` to run the
                         container on a remote node.
    """

    _KFD_DEVICE = "/dev/kfd"
    _DRI_PATH = "/dev/dri"

    def __init__(
        self,
        image: str,
        gpu_index: int = 0,
        runtime: str = "docker",
        use_amd_devices: bool = True,
        extra_run_flags: str = "",
        *,
        gpu_indices: list[int] | None = None,
        ipc: str = "",
        privileged: bool = False,
        host_executor: AbstractExecutor | None = None,
    ) -> None:
        self.image = image
        self.gpu_index = gpu_index
        self.gpu_indices = gpu_indices
        self.runtime = runtime
        self.use_amd_devices = use_amd_devices
        self.ipc = ipc
        self.privileged = privileged
        self.extra_run_flags = extra_run_flags
        # Delegate docker CLI invocations to the host executor (local CpuExecutor by
        # default, or an SshExecutor for a remote node) so the container runs where
        # the GPUs were acquired. No GPU env is injected into the docker client.
        self._host = host_executor or CpuExecutor()
        # Name of the running persistent container, or None in one-shot mode.
        self._container: str | None = None

    # ------------------------------------------------------------------
    # Runtime environment probe
    # ------------------------------------------------------------------

    def probe(self) -> ContainerStatus:
        """Inspect the container runtime environment on the local host.

        Issues lightweight shell commands to determine whether the runtime is
        installed, whether the daemon is reachable, whether AMD GPU devices are
        present, and whether the current user has permission to run containers.

        Returns:
            ContainerStatus populated from the probe results.  Check
            ``status.ready`` for an all-in-one gate, or inspect individual
            fields for targeted skip conditions.
        """
        status = ContainerStatus()

        # -- Runtime binary present ----------------------------------------
        ver_result = self._host.run(f"{self.runtime} --version 2>/dev/null")
        if not ver_result.ok or not ver_result.stdout.strip():
            status.errors.append(f"Container runtime {self.runtime!r} is not installed")
            return status  # nothing more to probe without a working runtime

        status.runtime_available = True
        status.runtime_version = ver_result.stdout.strip()

        # -- Daemon reachable ----------------------------------------------
        info_result = self._host.run(f"{self.runtime} info 2>/dev/null")
        status.daemon_active = info_result.ok
        if not info_result.ok:
            status.errors.append(f"{self.runtime!r} daemon is not reachable — " "is the service running?")

        # -- Compose: plugin style first, fall back to legacy standalone ---
        compose_result = self._host.run(f"{self.runtime} compose version 2>/dev/null")
        if compose_result.ok and compose_result.stdout.strip():
            status.compose_available = True
            status.compose_version = compose_result.stdout.strip()
        else:
            legacy_result = self._host.run("docker-compose --version 2>/dev/null")
            if legacy_result.ok and legacy_result.stdout.strip():
                status.compose_available = True
                status.compose_version = legacy_result.stdout.strip()
            else:
                status.errors.append("Docker Compose is not available (neither plugin nor standalone)")

        # -- AMD device nodes present on the host --------------------------
        kfd_result = self._host.run(f"test -c {self._KFD_DEVICE} && echo present")
        dri_result = self._host.run(f"test -d {self._DRI_PATH} && echo present")
        status.amd_devices_present = (
            kfd_result.ok and "present" in kfd_result.stdout and dri_result.ok and "present" in dri_result.stdout
        )
        if not status.amd_devices_present:
            status.errors.append(
                f"AMD GPU devices not found on host: " f"{self._KFD_DEVICE} and/or {self._DRI_PATH} are missing"
            )

        # -- User permissions (can run containers without sudo) -----------
        ps_result = self._host.run(f"{self.runtime} ps 2>/dev/null")
        status.user_in_group = ps_result.ok
        if not ps_result.ok:
            status.errors.append(
                f"Current user cannot run {self.runtime!r} commands — " "check docker group membership"
            )

        return status

    # ------------------------------------------------------------------
    # AbstractExecutor contract — one-shot container execution
    # ------------------------------------------------------------------

    def run(self, command: str, timeout: float | None = None) -> ExecutionResult:
        """Run *command* in a one-shot container with AMD GPU device passthrough.

        Equivalent to::

            docker run --rm \
                --device=/dev/kfd --device=/dev/dri \
                --group-add=video \
                --env=ROCR_VISIBLE_DEVICES=<gpu_index> \
                <image> sh -c <command>

        The container is removed automatically after the command exits
        (``--rm``).  For repeated commands against the same container, use
        a session-scoped fixture that starts the container once and calls
        ``exec_in()`` instead.

        Args:
            command: Shell command string to execute inside the container.
            timeout: Maximum seconds before the container is force-killed
                     (default 300 s).

        Returns:
            ExecutionResult with exit_code, stdout, stderr, and wall-clock
            duration, forwarded from the host subprocess.
        """
        # Persistent mode: exec into the container started by start() so state
        # (background PIDs, checkpoints) survives across calls.
        if self._container is not None:
            return self.exec_in(self._container, command, timeout=timeout)
        cli = self._assemble_run_command(command)
        logger.debug("ContainerExecutor.run: %s", cli)
        return self._host.run(cli, timeout=timeout or 300.0)

    # ------------------------------------------------------------------
    # Persistent container lifecycle (start once, exec many, stop)
    # ------------------------------------------------------------------

    def start(self, name: str | None = None, timeout: float = 120.0, pull_timeout: float = 1800.0) -> str:
        """Start a detached long-lived container and return its name.

        Pulls the image if absent (large ROCm images can exceed the short detached-run timeout
        when pulled inline), then runs ``docker run -d ... <image> sleep infinity`` (idempotent).
        After this, ``run()`` execs into this container instead of spawning a one-shot one.
        """
        if self._container is not None:
            return self._container
        container_name = name or f"rocmtest_{os.getpid()}_{next(_NAME_COUNTER)}"

        # Pre-pull separately so the (potentially long) image download gets its own generous
        # timeout, leaving the detached run quick. Skipped when the image is already local.
        pull = self._host.run(
            f"{self.runtime} image inspect {self.image} >/dev/null 2>&1 || {self.runtime} pull {self.image}",
            timeout=pull_timeout,
        )
        if not pull.ok:
            raise RuntimeError(
                f"Failed to pull container image {self.image!r} "
                f"(exit={pull.exit_code}): {(pull.stderr or pull.stdout)[-1000:]}"
            )

        # --init runs a real init (tini) as PID 1 that reaps zombies. Without it, PID 1 is
        # `sleep infinity` (never wait()s), so a process killed inside the container -- e.g. the
        # workload SIGKILLed by `criu dump` -- lingers forever as a zombie, holding its PID and
        # breaking `criu restore`, which recreates the process at that same PID.
        parts = [self.runtime, "run", "-d", "--init", "--name", container_name, *self._common_run_flags()]
        parts += [self.image, "sleep", "infinity"]
        cli = " ".join(parts)
        logger.debug("ContainerExecutor.start: %s", cli)
        result = self._host.run(cli, timeout=timeout)
        if not result.ok:
            raise RuntimeError(
                f"Failed to start container from image {self.image!r} "
                f"(exit={result.exit_code}): {(result.stderr or result.stdout)[-1000:]}"
            )
        self._container = container_name
        logger.info("Started persistent container %s from %s", container_name, self.image)
        return container_name

    def stop(self, timeout: float = 60.0) -> None:
        """Force-remove the persistent container started by ``start()`` (no-op otherwise)."""
        if self._container is None:
            return
        self._host.run(f"{self.runtime} rm -f {self._container}", timeout=timeout)
        logger.info("Removed persistent container %s", self._container)
        self._container = None

    # ------------------------------------------------------------------
    # Exec into a running container
    # ------------------------------------------------------------------

    def exec_in(
        self,
        container_name: str,
        command: str,
        timeout: float | None = None,
    ) -> ExecutionResult:
        """Execute *command* inside an already-running named container.

        Unlike ``run()``, no new container is created.  Use this for
        long-lived test containers started by a session-scoped fixture where
        creating a container per command is too expensive.

        Args:
            container_name: Name or ID of the running container.
            command:        Shell command to execute inside it.
            timeout:        Maximum seconds (default 300 s).

        Returns:
            ExecutionResult forwarded from the ``docker exec`` invocation.
        """
        cli = f"{self.runtime} exec {container_name} " f"sh -c {shlex.quote(command)}"
        logger.debug("ContainerExecutor.exec_in[%s]: %s", container_name, command)
        return self._host.run(cli, timeout=timeout or 300.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assemble_run_command(self, user_command: str) -> str:
        """Build the full ``docker run`` CLI invocation string.

        On Linux, AMD GPU device nodes (``/dev/kfd``, ``/dev/dri``) and the
        ``video`` group are added when ``use_amd_devices=True``.  On Windows
        these device paths do not exist; ROCm for Windows exposes GPU access
        through a different mechanism so they are omitted automatically.

        Args:
            user_command: The shell command to run inside the container.

        Returns:
            A shell-safe string suitable for passing to ``CpuExecutor.run()``.
        """
        parts = [self.runtime, "run", "--rm", *self._common_run_flags()]
        parts.append(self.image)
        parts += ["sh", "-c", shlex.quote(user_command)]
        return " ".join(parts)

    def _rocr_value(self) -> str:
        """Return the ``ROCR_VISIBLE_DEVICES`` value (multi-index list or single index)."""
        if self.gpu_indices:
            return ",".join(str(i) for i in self.gpu_indices)
        return str(self.gpu_index)

    def _common_run_flags(self) -> list[str]:
        """Return the ``docker run`` flags shared by one-shot and persistent modes.

        Covers ``--ipc``, ``--privileged``, AMD device passthrough + ``ROCR_VISIBLE_DEVICES``
        (Linux), and any caller-supplied ``extra_run_flags``.
        """
        parts: list[str] = []
        if self.ipc:
            parts.append(f"--ipc={self.ipc}")
        if self.privileged:
            parts.append("--privileged")

        if self.use_amd_devices:
            platform = os_adapter_factory().get_platform_name()
            if platform == "linux":
                # Linux: pass AMD KFD and DRI device nodes into the container.
                parts += [
                    f"--device={self._KFD_DEVICE}",
                    f"--device={self._DRI_PATH}",
                    "--group-add=video",
                ]
            # Windows: GPU access is managed by the ROCm Windows driver stack;
            # device passthrough flags are not applicable.
            parts.append(f"--env=ROCR_VISIBLE_DEVICES={self._rocr_value()}")

        if self.extra_run_flags:
            parts.append(self.extra_run_flags)
        return parts
