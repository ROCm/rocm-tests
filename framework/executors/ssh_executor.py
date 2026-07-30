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
import itertools
import logging
import os
import pathlib
import posixpath
import select
import shlex
import sys
import threading
import time
import uuid

from framework.common.helpers import ExecutionResult
from framework.common.workspace_layout import (
    REMOTE_WORKSPACE_DIR,
    is_managed_remote_path,
    remote_workspace_path,
    sftp_stage_path,
)
from framework.executors.abstract_executor import AbstractExecutor
from framework.executors.background_process import (
    _CAPTURE_TRUNCATED_NOTICE,
    _MAX_CAPTURE_BYTES,
    AbstractBackgroundProcess,
)
from framework.logging.test_logger import TestLogger

# Poll cadence for the shared background-output poller. One coalesced control
# channel per tick reads every streamed handle, so a relaxed interval keeps
# control-channel traffic negligible even with many concurrent background roles.
_SSH_BG_POLL_INTERVAL = 5.0

# Live background output is routed here (console + per-test log) so it is
# timestamped and aggregated exactly like foreground run() output.
_ROCM_TEST_LOGGER = logging.getLogger("rocm.test")

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


class _BackgroundStreamPoller:
    """Per-executor coalesced poller that streams many background handles cheaply.

    Instead of one polling thread + channel per background process (which does
    not scale — N handles => 2N channels per second), a single daemon thread
    reads **every** registered handle's newly-appended stdout/stderr in ONE
    control channel per tick (default 5 s) using a length-prefixed protocol.  It
    then line-buffers and dispatches the new bytes to each handle.

    The thread starts on the first ``register()`` and stops once the last handle
    is ``unregister()``-ed, so idle executors carry no background threads.
    """

    def __init__(self, executor: SshExecutor, interval: float = _SSH_BG_POLL_INTERVAL) -> None:
        self._executor = executor
        self._interval = interval
        self._handles: dict[str, SshBackgroundProcess] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def register(self, handle: SshBackgroundProcess) -> None:
        """Add *handle* to the poll set, starting the thread if needed."""
        with self._lock:
            self._handles[handle.stream_id] = handle
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()

    def unregister(self, handle: SshBackgroundProcess) -> None:
        """Remove *handle*; stop the thread when no handles remain."""
        with self._lock:
            self._handles.pop(handle.stream_id, None)
            empty = not self._handles
        if empty:
            self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(Exception):
                self._poll_once()
            self._stop.wait(self._interval)
        with contextlib.suppress(Exception):
            self._poll_once()  # final drain for any still-registered handles

    @staticmethod
    def _segment_cmd(marker: str, path: str, off: int) -> str:
        """Shell fragment emitting ``@@<marker> <size> <n>`` then *n* new bytes."""
        return (
            f"sz=$(wc -c < {path} 2>/dev/null || echo 0); "
            f'n=0; [ "$sz" -gt {off} ] && n=$((sz-{off})); '
            f'printf "\\n@@{marker} %s %s\\n" "$sz" "$n"; '
            f'[ "$n" -gt 0 ] && tail -c "$n" {path} 2>/dev/null; '
            "true"
        )

    @staticmethod
    def _parse(blob: str):
        """Yield ``(marker, size, data)`` from a length-prefixed poll response."""
        pos = 0
        while True:
            i = blob.find("@@", pos)
            if i < 0:
                break
            nl = blob.find("\n", i)
            if nl < 0:
                break
            parts = blob[i + 2 : nl].split()
            if len(parts) != 3:
                pos = nl + 1
                continue
            marker, sz_s, n_s = parts
            try:
                size, n = int(sz_s), int(n_s)
            except ValueError:
                pos = nl + 1
                continue
            data = blob[nl + 1 : nl + 1 + n]
            yield marker, size, data
            pos = nl + 1 + n

    def _poll_once(self) -> None:
        with self._lock:
            handles = list(self._handles.values())
        if not handles:
            return
        segments = []
        for h in handles:
            segments.append(self._segment_cmd(f"O:{h.stream_id}", h._out_file, h._o_off))
            segments.append(self._segment_cmd(f"E:{h.stream_id}", h._err_file, h._e_off))
        _, blob = self._executor._control("; ".join(segments), timeout=30.0)
        by_id = {h.stream_id: h for h in handles}
        for marker, size, data in self._parse(blob):
            kind, _, hid = marker.partition(":")
            handle = by_id.get(hid)
            if handle is None:
                continue
            if kind == "O":
                handle._o_off = size
                handle._feed("o", data)
            else:
                handle._e_off = size
                handle._feed("e", data)


class SshBackgroundProcess(AbstractBackgroundProcess):
    """Detached remote background handle mirroring ``BackgroundProcess``'s API.

    ``BackgroundProcess`` wraps a local ``subprocess.Popen``; that model does not
    fit Paramiko's channel-based transport.  An earlier design held one dedicated
    ``exec_command`` channel open for each background role's entire lifetime, but
    that saturates OpenSSH's ``MaxSessions`` limit (default 10) as soon as a test
    launches many concurrent roles on one connection — the server rejects the
    excess with ``open failed: Connect failed`` and the run wedges.

    This handle instead launches each role **fully detached** on the node (via
    ``setsid`` with its stdin closed and stdout/stderr redirected to node-side
    capture files), so the launching channel can close immediately.  Liveness,
    streaming, and teardown use brief, transient control channels serialised by
    an executor-level lock — never more than one open at a time.

    Output handling is deliberately lightweight:

    - **Default (no live view):** nothing is polled during the run.  ``stop()``
      fetches the capture files once into a clean, stream-separated
      ``ExecutionResult``.  The only run-time control traffic is the caller's own
      ``is_alive``/``poll`` calls.
    - **Opt-in live view** (``stream=True`` and/or a ``log_path``): the handle
      registers with the executor's shared coalesced poller, which reads every
      streamed handle in ONE control channel per tick (default 5 s).  New output
      is line-buffered and forwarded — labelled ``[bg <console_label>]`` per
      process — to the ``rocm.test`` logger (console + per-test log) when ``stream`` is set,
      and appended to *log_path* when given.

    Attributes:
        stop_result:   ``ExecutionResult`` populated by ``stop()`` once the remote
                       process terminates.  ``None`` until ``stop()`` is called.
        console_label: Human-readable label used to attribute live output lines.
        stream_id:     Stable per-handle id used by the shared poller.
    """

    _counter = itertools.count(1)

    def __init__(
        self,
        *,
        executor: SshExecutor,
        remote_pid: int | None,
        rc_file: str,
        pid_file: str,
        out_file: str,
        err_file: str,
        t0: float,
        session_key: str,
        log_path: str | None = None,
        console_label: str | None = None,
        stream: bool = False,
    ) -> None:
        """Build a detached handle; register for live streaming only if requested.

        Called by ``SshExecutor.start_background()`` only.
        """
        self._executor = executor
        self._remote_pid = remote_pid
        self._rc_file = rc_file
        self._pid_file = pid_file
        self._out_file = out_file
        self._err_file = err_file
        self._log_path = log_path
        self._t0 = t0
        self._session_key = session_key
        self.stop_result: ExecutionResult | None = None

        self.stream_id = f"bg{next(self._counter)}"
        self.console_label = console_label or (f"pid{remote_pid}" if remote_pid is not None else self.stream_id)
        self._console = stream
        # Byte offsets already forwarded per stream, and partial-line remainders
        # so a labelled line is never split across two poll ticks.
        self._o_off = 0
        self._e_off = 0
        self._o_buf = ""
        self._e_buf = ""
        # Long-lived handle intentionally kept open for the process lifetime and
        # closed in stop().
        self._log_fh = (
            open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — long-lived handle, closed in stop()
            if log_path
            else None
        )  # pylint: disable=consider-using-with
        self._streaming = bool(stream or log_path)
        if self._streaming:
            self._executor._register_stream(self)

    def _emit_line(self, stream: str, line: str) -> None:
        """Forward one complete *line* (labelled) to the logger and/or log file."""
        if self._console:
            if stream == "o":
                _ROCM_TEST_LOGGER.info("[bg %s] %s", self.console_label, line)
            else:
                _ROCM_TEST_LOGGER.warning("[bg %s] %s", self.console_label, line)
        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.write(line + "\n")
                self._log_fh.flush()

    def _feed(self, stream: str, data: str) -> None:
        """Buffer *data* for *stream* (``'o'``/``'e'``), emitting each complete line."""
        if not data:
            return
        if stream == "o":
            self._o_buf += data
            lines = self._o_buf.split("\n")
            self._o_buf = lines.pop()
        else:
            self._e_buf += data
            lines = self._e_buf.split("\n")
            self._e_buf = lines.pop()
        for line in lines:
            self._emit_line(stream, line)

    def _flush_partial(self) -> None:
        """Emit any trailing partial (newline-less) lines at teardown."""
        if self._o_buf:
            self._emit_line("o", self._o_buf)
            self._o_buf = ""
        if self._e_buf:
            self._emit_line("e", self._e_buf)
            self._e_buf = ""

    @property
    def pid(self) -> int:
        """Remote OS process ID (``-1`` when PID capture failed)."""
        return self._remote_pid if self._remote_pid is not None else -1

    @property
    def is_alive(self) -> bool:
        """True while the detached remote process (session leader) still exists."""
        if self.stop_result is not None or self._remote_pid is None:
            return False
        _, out = self._executor._control(f"kill -0 {self._remote_pid} 2>/dev/null && echo A || echo D")
        return out.strip().endswith("A")

    def _read_rc(self) -> int:
        """Return the remote exit code recorded on completion (``-1`` if absent)."""
        _, out = self._executor._control(f"cat {self._rc_file} 2>/dev/null || true")
        line = out.strip().splitlines()[-1] if out.strip() else ""
        try:
            return int(line)
        except ValueError:
            return -1

    def poll(self) -> int | None:
        """Non-blocking exit-code check; ``None`` while the remote process runs."""
        if self.is_alive:
            return None
        return self._read_rc()

    def _fetch_capture(self, path: str) -> str:
        """Return the tail-capped contents of a node-side capture file.

        Only the most recent ``_MAX_CAPTURE_BYTES`` are retrieved (via ``tail -c``),
        matching the local ``BackgroundProcess`` cap; a truncation notice is
        prepended when the file was larger.
        """
        _, size_out = self._executor._control(f"wc -c < {path} 2>/dev/null || echo 0")
        try:
            size = int(size_out.strip().splitlines()[-1]) if size_out.strip() else 0
        except ValueError:
            size = 0
        _, data = self._executor._control(f"tail -c {_MAX_CAPTURE_BYTES} {path} 2>/dev/null || true")
        text = data.rstrip()
        if size > _MAX_CAPTURE_BYTES:
            text = _CAPTURE_TRUNCATED_NOTICE + text
        return text

    def stop(self, timeout: float = 30.0) -> ExecutionResult:
        """Terminate the detached remote process group (if running) and collect its result.

        Idempotent: repeated calls return the same cached ``ExecutionResult``.

        Because the role is launched under its own session (``setsid``), signalling
        the negative PID reaches the whole process group; the plain PID is also
        signalled as a fallback.  If live streaming was enabled the handle is
        unregistered from the shared poller (and any trailing partial line
        flushed) first, then the node-side stdout/stderr capture files are
        fetched (tail-capped) into the result.

        Args:
            timeout: Maximum seconds to wait for the kill control command.

        Returns:
            ``ExecutionResult`` with the remote exit code (``-1`` if the process
            was killed before recording one), captured stdout/stderr, and
            wall-clock duration.
        """
        if self.stop_result is not None:
            return self.stop_result

        # Detach from the shared poller so it stops reading our capture files,
        # then flush any buffered partial line to the console/log.
        if self._streaming:
            with contextlib.suppress(Exception):
                self._executor._unregister_stream(self)
            self._flush_partial()

        if self._remote_pid is not None:
            kill_cmd = (
                f"kill -TERM -{self._remote_pid} 2>/dev/null; kill -TERM {self._remote_pid} 2>/dev/null; "
                f"sleep 1; kill -KILL -{self._remote_pid} 2>/dev/null; kill -KILL {self._remote_pid} 2>/dev/null; "
                "true"
            )
            with contextlib.suppress(Exception):
                self._executor._control(kill_cmd, timeout=timeout)

        exit_code = self._read_rc()
        stdout = self._fetch_capture(self._out_file)
        stderr = self._fetch_capture(self._err_file)

        if self._log_fh is not None:
            with contextlib.suppress(Exception):
                self._log_fh.close()

        with contextlib.suppress(Exception):
            self._executor._control(
                f"rm -f {self._rc_file} {self._pid_file} {self._out_file} {self._err_file} 2>/dev/null || true"
            )

        self.stop_result = ExecutionResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=time.monotonic() - self._t0,
        )
        return self.stop_result

    def __enter__(self) -> SshBackgroundProcess:
        return self


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
        # Serialises transient control channels (liveness/streaming/teardown of
        # background processes) so that many concurrent SshBackgroundProcess
        # handles sharing this connection never open more than one control
        # channel at a time — keeping well under the remote MaxSessions limit.
        self._ctrl_lock = threading.Lock()
        # Shared coalesced poller for live background output; created lazily when
        # the first streamed background handle registers.
        self._stream_poller: _BackgroundStreamPoller | None = None

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
        *,
        stream: bool = False,
        env_overrides: dict | None = None,
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
            stream:        When True, stream stdout/stderr and emit periodic
                           heartbeat lines while the command is still running.
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

    def _control(self, cmd: str, timeout: float = 15.0) -> tuple[int, str]:
        """Run a short control command on a transient channel and return ``(rc, stdout)``.

        Used by :class:`SshBackgroundProcess` for liveness checks, output
        streaming, and teardown.  Calls are serialised by ``_ctrl_lock`` and each
        opens and closes exactly one channel, so control traffic never
        accumulates concurrent sessions even when many background handles share
        this connection.  It deliberately bypasses ``TestLogger`` and the
        PID-marker wrapping used by ``run()`` to keep the poll loop quiet and
        cheap.
        """
        with self._ctrl_lock:
            client = self._connect()
            try:
                _, stdout_ch, _ = client.exec_command(cmd, timeout=timeout)  # nosec B601 — SSH executor by design
                stdout_ch.channel.settimeout(timeout)
                data = stdout_ch.read().decode("utf-8", errors="replace")
                rc = stdout_ch.channel.recv_exit_status()
                return rc, data
            except Exception:  # pylint: disable=broad-except
                return -1, ""

    def _register_stream(self, handle: SshBackgroundProcess) -> None:
        """Register a background handle for live output streaming (lazy poller)."""
        if self._stream_poller is None:
            self._stream_poller = _BackgroundStreamPoller(self)
        self._stream_poller.register(handle)

    def _unregister_stream(self, handle: SshBackgroundProcess) -> None:
        """Deregister a background handle from the live output poller."""
        if self._stream_poller is not None:
            self._stream_poller.unregister(handle)

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
        console_label: str | None = None,
        stream: bool = False,
    ) -> AbstractBackgroundProcess:
        """Start *command* on the remote host as a fully detached background process.

        The command is launched under ``setsid`` with its stdin closed and its
        stdout/stderr redirected to node-side capture files, so the launch channel
        closes immediately instead of being held open for the process lifetime.
        This avoids exhausting OpenSSH's ``MaxSessions`` limit when many roles run
        concurrently over a single connection.  ``ROCR_VISIBLE_DEVICES`` is
        injected (via ``run()``) when the executor was constructed with
        *gpu_indices*, so detached roles observe the same GPU isolation.

        Live output streaming is **opt-in** and cheap: by default nothing is
        polled during the run and ``stop()`` fetches the capture files once into a
        clean, stream-separated ``ExecutionResult``.  When *stream* is set and/or a
        *log_path* is given, the handle registers with the executor's shared
        coalesced poller (one control channel per ~5 s tick for *all* streamed
        handles); new output is line-buffered and forwarded — labelled
        ``[bg <console_label>]`` — to the ``rocm.test`` logger (console + per-test
        log) when *stream* is set, and appended to *log_path* when given.

        Args:
            command:  Shell command to launch on the remote host.  A command that
                      adds its own redirection (e.g. ``> file 2> file``) takes
                      precedence and leaves the capture files empty (so nothing
                      is echoed for it).
            timeout:  Launch timeout (also the default stop grace); the per-call
                      stop grace is passed to ``SshBackgroundProcess.stop()``.
            log_path: If given, streamed stdout/stderr are appended to this local
                      file as they arrive (one poll interval behind the node).
                      Use a distinct path per process to keep output attributable.
            console_label: Human-readable label for live console lines
                      (``[bg <console_label>]``). Defaults to ``pid<N>`` / an
                      internal id.
            stream:   Emit live output to the ``rocm.test`` logger (console +
                      per-test log).  Combine with *log_path* to also record it.

        Returns:
            ``SshBackgroundProcess`` handle (duck-typed to ``BackgroundProcess``).
        """
        token = uuid.uuid4().hex[:12]
        base = f"/tmp/.rock_bg_{token}"  # nosec B108 — node-local scratch control files
        rc_file = f"{base}.rc"
        pid_file = f"{base}.pid"
        out_file = f"{base}.out"
        err_file = f"{base}.err"

        # Inner shell records the session-leader PID, runs the workload in a subshell
        # whose stdout/stderr are captured to node-side files, then records the exit
        # code. ROCR_VISIBLE_DEVICES is inherited from run()'s injected env. A command
        # with its own redirection overrides the subshell capture (files stay empty).
        inner = f"echo $$ > {pid_file}; ( {command} ) > {out_file} 2> {err_file}; echo $? > {rc_file}"
        # Detach with setsid so the role survives the launch channel closing and its
        # negative-PID kill signals the whole process group. Wait briefly for the PID
        # file, then emit the PID (falling back to $! of the detached job).
        launch = (
            f"setsid bash -lc {shlex.quote(inner)} </dev/null >/dev/null 2>&1 & "
            f"bgpid=$!; "
            f"for _i in $(seq 1 50); do [ -s {pid_file} ] && break; sleep 0.1; done; "
            f"cat {pid_file} 2>/dev/null || echo $bgpid"
        )

        t0 = time.monotonic()
        result = self.run(launch, timeout=timeout if timeout is not None else 60.0)
        remote_pid: int | None = None
        text = result.stdout.strip()
        if text:
            with contextlib.suppress(ValueError):
                remote_pid = int(text.splitlines()[-1].strip())
        if remote_pid is None:
            logger.warning(
                "SshExecutor[%s] start_background: could not capture remote PID (stdout=%r stderr=%r)",
                self.session_key,
                result.stdout[:200],
                result.stderr[:200],
            )

        return SshBackgroundProcess(
            executor=self,
            remote_pid=remote_pid,
            rc_file=rc_file,
            pid_file=pid_file,
            out_file=out_file,
            err_file=err_file,
            t0=t0,
            session_key=self.session_key,
            log_path=log_path,
            console_label=console_label,
            stream=stream,
        )
