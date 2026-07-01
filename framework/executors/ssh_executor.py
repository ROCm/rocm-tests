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
connection when finished.

Usage (via ``NodePool`` / ``target_executor`` — not instantiated directly in tests):
    with SshExecutor(host="gpu-node-01", user="ci", key_path="~/.ssh/id_rsa") as node:
        result = node.run("rocm-smi --showid")
        assert result.ok
"""

from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import posixpath
import select
import shlex
import sys
import time

from framework.common.helpers import ExecutionResult
from framework.common.workspace_layout import (
    REMOTE_WORKSPACE_DIR,
    is_managed_remote_path,
    remote_workspace_path,
    sftp_stage_path,
)
from framework.executors.abstract_executor import AbstractExecutor
from framework.logging.test_logger import TestLogger

try:
    import paramiko

    _PARAMIKO_OK = True
except ImportError:
    _PARAMIKO_OK = False

logger = logging.getLogger(__name__)

# Paramiko's transport layer emits connection and auth INFO messages that clutter
# test output.  Raise the threshold so only warnings and errors are shown.
_PARAMIKO_LOG_THRESHOLD = logging.WARNING

# When streaming (long remote builds), emit a heartbeat at this cadence so the
# console shows the command is alive and how long it has been running, even when
# the build produces no output for minutes at a time.
_STREAM_HEARTBEAT_SECS = 60.0


def _stream_remote_channel(
    chan,
    timeout: float,
    started_at: float,
    session_key: str,
    pre_stderr: list[str] | None = None,
) -> tuple[str, str, int]:
    """Drain a Paramiko channel incrementally, forwarding output as it arrives.

    Reads stdout/stderr in a non-blocking loop, writing each chunk to
    ``sys.stdout`` in real time (visible with pytest ``-s`` / capture disabled)
    and logging a periodic elapsed-time heartbeat (always visible via ``log_cli``).
    This gives progress visibility for long-running remote builds, which would
    otherwise be silent until completion under the default blocking read.

    Args:
        chan:        Paramiko ``Channel`` (``stdout`` channel of ``exec_command``).
        timeout:     Maximum seconds from *started_at* before raising.
        started_at:  ``time.monotonic()`` value captured when the command started.
        session_key: Executor session key, used in heartbeat log lines.
        pre_stderr:  Any stderr lines already read (e.g. while capturing the PID).

    Returns:
        Tuple ``(stdout, stderr, exit_code)``.

    Raises:
        TimeoutError: If the command exceeds *timeout*.
    """
    stdout_parts: list[str] = []
    stderr_parts: list[str] = list(pre_stderr or [])
    last_beat = time.monotonic()
    chan.setblocking(False)

    while True:
        select.select([chan], [], [], 1.0)
        got = False
        if chan.recv_ready():
            data = chan.recv(65536).decode("utf-8", errors="replace")
            if data:
                stdout_parts.append(data)
                sys.stdout.write(data)
                sys.stdout.flush()
                got = True
        if chan.recv_stderr_ready():
            data = chan.recv_stderr(65536).decode("utf-8", errors="replace")
            if data:
                stderr_parts.append(data)
                sys.stdout.write(data)
                sys.stdout.flush()
                got = True

        now = time.monotonic()
        if now - started_at > timeout:
            raise TimeoutError(f"remote command exceeded {timeout:.0f}s")
        if now - last_beat >= _STREAM_HEARTBEAT_SECS:
            logger.info("SshExecutor[%s] still running (elapsed %.0fs)…", session_key, now - started_at)
            last_beat = now

        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if not got:
            time.sleep(0.1)

    exit_code = chan.recv_exit_status()
    return "".join(stdout_parts), "".join(stderr_parts), exit_code


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

    When *gpu_indices* is non-empty, ``ROCR_VISIBLE_DEVICES`` is injected
    automatically on every ``run()`` call to restrict GPU visibility on the
    remote host.
    """

    def __init__(
        self,
        host: str,
        user: str | None = None,
        key_path: str | None = None,
        password: str | None = None,
        port: int = 22,
        connect_timeout: float = 30.0,
        gpu_indices: list[int] | None = None,
        test_logger: TestLogger | None = None,
    ) -> None:
        self.host = host
        self.user = user or os.getenv("USER", "root")
        self.key_path = os.path.expanduser(key_path) if key_path else None
        self.password = password
        self.port = port
        self.connect_timeout = connect_timeout
        self.gpu_indices: list[int] = list(gpu_indices) if gpu_indices else []
        self.test_logger = test_logger
        self._client: paramiko.SSHClient | None = None
        self._remote_home: str | None = None
        self._remote_workspace_root: str | None = None

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

        logger.info("SshExecutor: opening connection host=%s", self.session_key)
        client = paramiko.SSHClient()
        client.load_system_host_keys()  # /etc/ssh/ssh_known_hosts
        with contextlib.suppress(FileNotFoundError):
            client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
        client.set_missing_host_key_policy(paramiko.RejectPolicy())

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

    def _get_remote_home(self) -> str:
        """Return the remote user's home directory, cached per SSH executor."""
        if self._remote_home:
            return self._remote_home
        result = self.run("printf '%s' \"$HOME\"", timeout=10.0)
        if not result.ok or not result.stdout.strip():
            raise RuntimeError(
                f"SshExecutor: could not resolve remote HOME on {self.host} "
                f"(exit={result.exit_code}):\n{result.stderr}"
            )
        self._remote_home = result.stdout.strip()
        return self._remote_home

    def remote_path_for(self, local_path: str) -> str:
        """Return the stable remote staging path for a local path.

        Local absolute paths are staged under
        ``$HOME/run-rocm-tests/sftp`` rather than mirrored literally. This keeps
        remote builds portable across hosts where the local checkout owner/path
        """
        raw = pathlib.PurePath(local_path).as_posix()
        if self._is_managed_remote_path(raw):
            return posixpath.normpath(raw)
        return sftp_stage_path(self.remote_workspace_root(), local_path)

    def remote_workspace_root(self) -> str:
        """Return the root directory for all framework-managed remote files."""
        if self._remote_workspace_root:
            return self._remote_workspace_root
        self._remote_workspace_root = posixpath.join(self._get_remote_home(), REMOTE_WORKSPACE_DIR)
        return self._remote_workspace_root

    def _is_managed_remote_path(self, path: str) -> bool:
        """Return True when *path* is already under this executor's workspace."""
        return is_managed_remote_path(path, self.remote_workspace_root())

    def workspace_path_for(self, path: str | os.PathLike, category: str = "work") -> str:
        """Map a local/relative path into the managed remote workspace.

        Args:
            path: Local absolute path or repo-relative path.
            category: Workspace category. ``"work"`` is retained as a
                compatibility alias for ``output/``.

        Returns:
            Absolute remote path under the managed workspace.
        """
        norm = path if isinstance(path, str) else pathlib.PurePath(path)
        return remote_workspace_path(self.remote_workspace_root(), norm, category=category)

    def _mkdir_remote_dirs(self, dirs: list[str]) -> None:
        """Create remote directories in bounded batches and fail loudly."""
        if not dirs:
            return

        batch: list[str] = []
        batch_len = len("mkdir -p")
        for directory in dirs:
            quoted = shlex.quote(directory)
            # Keep each SSH command comfortably below common ARG_MAX limits.
            if batch and batch_len + len(quoted) + 1 > 24000:
                result = self.run("mkdir -p " + " ".join(batch), timeout=60.0)
                if not result.ok:
                    raise RuntimeError(
                        f"SshExecutor.upload_tree: remote mkdir failed on {self.host} "
                        f"(exit={result.exit_code}):\n{result.stderr}"
                    )
                batch = []
                batch_len = len("mkdir -p")
            batch.append(quoted)
            batch_len += len(quoted) + 1

        if batch:
            result = self.run("mkdir -p " + " ".join(batch), timeout=60.0)
            if not result.ok:
                raise RuntimeError(
                    f"SshExecutor.upload_tree: remote mkdir failed on {self.host} "
                    f"(exit={result.exit_code}):\n{result.stderr}"
                )

    def upload_tree(self, local_dir: str) -> str:
        """Upload a local directory tree to the remote host at the same absolute path.

        Creates parent directories on the remote via SSH ``mkdir -p``, then
        transfers files via SFTP.

        Args:
            local_dir: Absolute local path to upload (mirrored at same path on remote).

        Raises:
            RuntimeError: If the SFTP transfer fails.
        """
        local_root = pathlib.Path(local_dir).resolve()
        if not local_root.exists():
            raise RuntimeError(f"SshExecutor.upload_tree: local path does not exist: {local_root}")
        if not local_root.is_dir():
            raise RuntimeError(f"SshExecutor.upload_tree: local path is not a directory: {local_root}")

        remote_root = self.remote_path_for(str(local_root))
        files = [f for f in local_root.rglob("*") if f.is_file()]
        if not files:
            self._mkdir_remote_dirs([remote_root])
            return remote_root

        dirs = sorted({posixpath.join(remote_root, f.relative_to(local_root).parent.as_posix()) for f in files})
        self._mkdir_remote_dirs(dirs)

        sftp = self._connect().open_sftp()
        try:
            for f in files:
                rel = f.relative_to(local_root).as_posix()
                remote_file = posixpath.join(remote_root, rel)
                sftp.put(str(f), remote_file)
                logger.debug("SshExecutor.upload_tree: %s → %s:%s", f, self.host, remote_file)
        except Exception as exc:
            raise RuntimeError(
                f"SshExecutor.upload_tree: SFTP upload to {self.host} failed while staging "
                f"{local_root} under {remote_root}: {exc}"
            ) from exc
        finally:
            sftp.close()
        return remote_root

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
        stream: bool = False,
    ) -> ExecutionResult:
        """Execute *command* on the remote host and return the result.

        The SSH connection is established on the first call and reused for all
        subsequent calls.  Any variables in *env_overrides* are prepended to
        *command* as ``export K=V`` shell assignments so the remote shell sees
        them without relying on SSH ``SendEnv`` configuration.

        When the executor was constructed with *gpu_indices*, ``ROCR_VISIBLE_DEVICES``
        is injected automatically before any caller-supplied *env_overrides*.

        Args:
            command:       Shell command string to execute on the remote host.
            timeout:       Maximum seconds to wait for the command to finish
                           (default 300 s).
            env_overrides: Additional environment variables to export on the
                           remote shell before running *command*.

        Returns:
            ExecutionResult with exit_code, stdout, stderr, and wall-clock duration.

        Raises:
            RuntimeError: If SSH connection or authentication fails.
            TimeoutError: If the remote command exceeds *timeout*.
        """
        effective_timeout = timeout if timeout is not None else 300.0

        def _kill_remote(client: paramiko.SSHClient, pid: int) -> None:
            """Send SIGTERM then SIGKILL to *pid* on the remote host."""
            kill_cmd = f"kill -TERM {pid} 2>/dev/null; sleep 1; kill -KILL {pid} 2>/dev/null || true"
            with contextlib.suppress(Exception):
                client.exec_command(kill_cmd, timeout=5.0)  # nosec B601 — intentional SSH kill command
            logger.warning(
                "SshExecutor[%s] killed orphaned remote PID %d after timeout",
                self.session_key,
                pid,
            )

        def _inner_run(cmd: str, t: float | None) -> ExecutionResult:
            t_eff = t if t is not None else effective_timeout
            effective_env: dict = {}
            if self.gpu_indices:
                effective_env["ROCR_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_indices)
            if env_overrides:
                effective_env.update(env_overrides)

            exports = ""
            if effective_env:
                exports = " ".join(f"export {k}={shlex.quote(str(v))};" for k, v in effective_env.items()) + " "

            # Wrap in a subshell that emits its PID on stderr before exec-ing the
            # real command.  The PID is captured immediately so we can kill the
            # remote process on timeout instead of leaving it as an orphan.
            pid_marker = "__SSH_PID__"
            script = f"echo {pid_marker}$$ >&2; {exports}exec bash -lc {shlex.quote(cmd)}"
            wrapped_cmd = f"bash -c {shlex.quote(script)}"

            client = self._connect()
            logger.debug("SshExecutor[%s] running: %s%s", self.session_key, exports, cmd)

            remote_pid: int | None = None
            t0 = time.monotonic()
            _, stdout_ch, stderr_ch = client.exec_command(  # nosec B601 — SSH executor by design
                wrapped_cmd, timeout=t_eff
            )
            stdout_ch.channel.settimeout(t_eff)

            # Read the PID line from stderr first (short timeout; fast in practice).
            stderr_ch.channel.settimeout(5.0)
            pid_line_buf: list[str] = []
            with contextlib.suppress(Exception):
                for raw_line in stderr_ch:
                    line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
                    if line.startswith(pid_marker):
                        with contextlib.suppress(ValueError):
                            remote_pid = int(line[len(pid_marker) :].strip())
                        break
                    pid_line_buf.append(line)
            stderr_ch.channel.settimeout(t_eff)

            try:
                if stream:
                    raw_stdout, raw_stderr, exit_code = _stream_remote_channel(
                        stdout_ch.channel, t_eff, t0, self.session_key, pre_stderr=pid_line_buf
                    )
                else:
                    raw_stdout = stdout_ch.read().decode("utf-8", errors="replace")
                    raw_stderr = "".join(pid_line_buf) + stderr_ch.read().decode("utf-8", errors="replace")
                    exit_code = stdout_ch.channel.recv_exit_status()
            except Exception:
                # Kill the remote process so it cannot hold the GPU after the
                # NodeSlot is released back to the pool (cascading timeouts).
                if remote_pid is not None:
                    _kill_remote(client, remote_pid)
                else:
                    logger.debug("SshExecutor[%s] timeout with unknown remote PID", self.session_key)
                stdout_ch.channel.close()
                raise

            duration = time.monotonic() - t0

            logger.debug("SshExecutor[%s] rc=%d duration=%.3fs", self.session_key, exit_code, duration)
            return ExecutionResult(
                exit_code=exit_code,
                stdout=raw_stdout.rstrip(),
                stderr=raw_stderr.rstrip(),
                duration=duration,
            )

        if self.test_logger is not None:
            start = self.test_logger.cmd_start(command)
            raw = _inner_run(command, timeout)
            self.test_logger.cmd_end(raw.stdout, raw.stderr, raw.exit_code, start)
            return raw
        return _inner_run(command, timeout)
