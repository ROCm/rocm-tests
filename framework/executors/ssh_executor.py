# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ssh_executor.py -- Persistent SSH executor for remote AMD lab nodes.

Executes commands on a remote host via a reusable Paramiko SSH session.
Designed for multi-node tests (RCCL allreduce, multi-host E2E) where
``LocalExecutor`` cannot reach the target machine.

The TCP connection is established lazily on the first ``run()`` call and kept
alive across subsequent calls.  This avoids per-command handshake overhead in
tests that issue many commands to the same node.

Call ``close()`` — or use the executor as a context manager — to release the
connection when finished.  The ``remote_pool`` fixture in ``executor_plugin``
manages this lifecycle automatically.

Usage (via ``remote_pool`` fixture — not instantiated directly in tests):
    with SshExecutor(host="gpu-node-01", user="ci", key_path="~/.ssh/id_rsa") as node:
        result = node.run("rocm-smi --showid")
        assert result.ok
"""

from __future__ import annotations

import logging
import os
import time

from framework.common.helpers import ExecutionResult
from framework.executors.abstract_executor import AbstractExecutor

try:
    import paramiko

    _PARAMIKO_OK = True
except ImportError:
    _PARAMIKO_OK = False

logger = logging.getLogger(__name__)

# Paramiko's transport layer emits connection and auth INFO messages that clutter
# test output.  Raise the threshold so only warnings and errors are shown.
_PARAMIKO_LOG_THRESHOLD = logging.WARNING


class SshExecutor(AbstractExecutor):
    """Execute commands on a remote host via a persistent Paramiko SSH session.

    The SSH connection is created on the first ``run()`` call (lazy) and kept
    alive for all subsequent calls, avoiding per-command TCP overhead in
    multi-step tests.

    Two authentication paths are supported:
        - Key-based (preferred for CI): pass *key_path*.
        - Password-based (lab/development): pass *password*.

    The executor also exposes a ``session_key`` property that ``RemoteNodePool``
    uses to deduplicate connections when the same host is requested multiple
    times within a single test.

    Args:
        host:            Remote hostname or IP address.
        user:            SSH login name (default: ``$USER`` environment variable).
        key_path:        Path to an SSH private key file; ``~`` is expanded.
        password:        SSH password — prefer *key_path* for automated environments.
        port:            SSH server port (default 22).
        connect_timeout: Seconds allowed for the initial TCP handshake (default 30 s).
    """

    def __init__(
        self,
        host: str,
        user: str | None = None,
        key_path: str | None = None,
        password: str | None = None,
        port: int = 22,
        connect_timeout: float = 30.0,
    ) -> None:
        self.host = host
        self.user = user or os.getenv("USER", "root")
        self.key_path = os.path.expanduser(key_path) if key_path else None
        self.password = password
        self.port = port
        self.connect_timeout = connect_timeout
        self._client: paramiko.SSHClient | None = None

        # Silence INFO-level messages from paramiko.transport / paramiko.auth
        # (e.g. "Authentication (publickey) failed.") that pollute test logs.
        logging.getLogger("paramiko.transport").setLevel(_PARAMIKO_LOG_THRESHOLD)
        logging.getLogger("paramiko.auth").setLevel(_PARAMIKO_LOG_THRESHOLD)

    # ------------------------------------------------------------------
    # Session identity — used by RemoteNodePool for deduplication
    # ------------------------------------------------------------------

    @property
    def session_key(self) -> str:
        """Unique identifier for this host/user/port combination.

        ``RemoteNodePool`` uses this string as the dictionary key so that
        ``acquire("gpu-node-01")`` called twice in the same test returns the
        same ``SshExecutor`` instance instead of opening a second TCP connection.

        Returns:
            String in the form ``"user@host:port"``.
        """
        return f"{self.user}@{self.host}:{self.port}"

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def _connect(self) -> paramiko.SSHClient:
        """Return an active Paramiko client, opening a new connection if needed.

        Raises:
            RuntimeError: If paramiko is not installed.
        """
        if not _PARAMIKO_OK:
            raise RuntimeError("paramiko is not installed — run: pip install paramiko")

        if (
            self._client is not None
            and self._client.get_transport() is not None
            and self._client.get_transport().is_active()
        ):
            return self._client

        logger.info("SshExecutor: opening connection to %s", self.session_key)
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs: dict = {
            "hostname": self.host,
            "port": self.port,
            "username": self.user,
            "timeout": self.connect_timeout,
            "allow_agent": True,
            "look_for_keys": True,
        }
        if self.key_path:
            connect_kwargs["key_filename"] = self.key_path
        if self.password:
            connect_kwargs["password"] = self.password

        client.connect(**connect_kwargs)
        self._client = client
        logger.debug("SshExecutor: connected to %s", self.session_key)
        return client

    def close(self) -> None:
        """Close the underlying SSH connection if one is open.

        Safe to call when no connection has been established.
        """
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.debug("SshExecutor: closed connection to %s", self.session_key)

    def __enter__(self) -> SshExecutor:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # AbstractExecutor contract
    # ------------------------------------------------------------------

    def run(
        self,
        command: str,
        timeout: float | None = None,
        env_overrides: dict | None = None,
    ) -> ExecutionResult:
        """Execute *command* on the remote host and return the result.

        The SSH connection is established on the first call and reused for all
        subsequent calls.  Any variables in *env_overrides* are prepended to
        *command* as ``export K=V`` shell assignments so the remote shell sees
        them without relying on SSH ``SendEnv`` configuration.

        Args:
            command:       Shell command string to execute on the remote host.
            timeout:       Maximum seconds to wait for the command to finish
                           (default 300 s).
            env_overrides: Environment variables to export on the remote shell
                           before running *command*.

        Returns:
            ExecutionResult with exit_code, stdout, stderr, and wall-clock duration.

        Raises:
            RuntimeError: If SSH connection or authentication fails.
            TimeoutError: If the remote command exceeds *timeout*.
        """
        effective_timeout = timeout if timeout is not None else 300.0

        if env_overrides:
            assignments = " ".join(f"{k}={v}" for k, v in env_overrides.items())
            command = f"export {assignments}; {command}"

        client = self._connect()
        logger.debug("SshExecutor[%s] running: %s", self.session_key, command)

        t0 = time.monotonic()
        _, stdout_ch, stderr_ch = client.exec_command(command, timeout=effective_timeout)
        stdout_ch.channel.settimeout(effective_timeout)

        raw_stdout = stdout_ch.read().decode("utf-8", errors="replace")
        raw_stderr = stderr_ch.read().decode("utf-8", errors="replace")
        exit_code = stdout_ch.channel.recv_exit_status()
        duration = time.monotonic() - t0

        logger.debug(
            "SshExecutor[%s] rc=%d duration=%.3fs",
            self.session_key,
            exit_code,
            duration,
        )
        return ExecutionResult(
            exit_code=exit_code,
            stdout=raw_stdout.rstrip(),
            stderr=raw_stderr.rstrip(),
            duration=duration,
        )
