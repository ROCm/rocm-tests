# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bridge gpu_monitored workloads to rocm-tests executors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
import pathlib
import posixpath
import shlex
import subprocess
from typing import TYPE_CHECKING

from framework.executors.abstract_executor import AbstractExecutor

if TYPE_CHECKING:
    from framework.executors.executor_group import NodeExecutorGroup


@dataclass(frozen=True)
class CommandResult:
    """Captured subprocess / executor result."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""


def make_monitor_executor(
    workload_executor: AbstractExecutor,
    *,
    rock_dir: str | None,
) -> AbstractExecutor:
    """Return an executor for ``amd-smi monitor`` (no GPU visibility mask).

    Mirrors ``remote_node_plugin._monitoring_executor``: local runs use
    ``CpuExecutor``; remote runs reuse ``SshExecutor`` with monitor commands
    prefixed to clear visibility env vars.
    """
    from framework.executors.cpu_executor import CpuExecutor
    from framework.executors.ssh_executor import SshExecutor

    if isinstance(workload_executor, SshExecutor):
        return _UnmaskedSshMonitorExecutor(workload_executor)

    env: dict[str, str] = {}
    if rock_dir:
        bin_dir = os.path.join(rock_dir, "bin")
        if os.path.isdir(bin_dir):
            env["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"
    return CpuExecutor(env_overrides=env, suppress_output_log=True)


class _UnmaskedSshMonitorExecutor(AbstractExecutor):
    """Wrap ``SshExecutor`` so monitor commands run without GPU masks.

    Inherits the executor base class so ``make_monitor_executor``'s declared
    return type holds for this branch too, and callers that check against
    ``AbstractExecutor`` accept it.
    """

    def __init__(self, ssh: AbstractExecutor) -> None:
        self._ssh = ssh

    def run(self, command: str, timeout: float | None = None, *, stream: bool = False):
        cleared = "env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES " f"-u CUDA_VISIBLE_DEVICES {command}"
        return self._ssh.run(cleared, timeout=timeout, stream=stream)

    def start_background(
        self,
        command: str,
        timeout: float | None = None,
        log_path: str | None = None,
        console_label: str | None = None,
        stream: bool = False,
    ):
        cleared = "env -u ROCR_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES " f"-u CUDA_VISIBLE_DEVICES {command}"
        return self._ssh.start_background(
            cleared,
            timeout=timeout,
            log_path=log_path,
            console_label=console_label,
            stream=stream,
        )


def _connection_executor(executor: AbstractExecutor | None) -> object | None:
    """Return the executor that owns the connection behind *executor*.

    Workloads are handed a ``NodeExecutorGroup`` and the monitor an unmasked
    wrapper; both delegate to the executor that knows whether the commands leave
    this machine and where the node keeps its files.
    """
    inner = getattr(executor, "_ssh", executor)
    group = getattr(inner, "_executors", None)
    if group:
        inner = group[0]
        inner = getattr(inner, "_ssh", inner)
    return inner


def is_remote_executor(executor: AbstractExecutor | None) -> bool:
    """Return True when *executor* runs its commands on another machine."""
    from framework.executors.ssh_executor import SshExecutor

    return isinstance(_connection_executor(executor), SshExecutor)


def executable_exists(executor: AbstractExecutor | None, path: os.PathLike[str] | str) -> bool:
    """Return True when *path* is an executable file on the node that will run it.

    Workloads are discovered where they will execute, which is not where pytest
    is running once a remote node is in play: a binary present under the node's
    ROCm tree would otherwise be reported missing and the test written off as
    unbuildable.
    """
    if is_remote_executor(executor):
        return executor.run(f"test -x {shlex.quote(str(path))}").ok
    local = pathlib.Path(path)
    return local.is_file() and os.access(local, os.X_OK)


def directory_exists(executor: AbstractExecutor | None, path: os.PathLike[str] | str) -> bool:
    """Return True when *path* is a directory on the node the workload runs on."""
    if is_remote_executor(executor):
        return executor.run(f"test -d {shlex.quote(str(path))}").ok
    return pathlib.Path(path).is_dir()


def ensure_remote_dir(executor: AbstractExecutor | None, path: os.PathLike[str] | str) -> None:
    """Create *path* on the executor's node so redirects into it can succeed."""
    if is_remote_executor(executor):
        executor.run(f"mkdir -p {shlex.quote(str(path))}")


def node_path_for(executor: AbstractExecutor | None, path: os.PathLike[str] | str) -> str:
    """Return the path *path* should take on the executor's node.

    A local absolute path is meaningless on a remote node -- it names a home
    directory belonging to a different user -- so it is mapped into the
    framework's managed remote workspace. Returns *path* unchanged when the
    workload runs here.
    """
    mapper = getattr(_connection_executor(executor), "workspace_path_for", None)
    if mapper is None or not is_remote_executor(executor):
        return str(path)
    return mapper(path)


def fetch_remote_file(
    executor: AbstractExecutor | None,
    local_path: os.PathLike[str] | str,
    remote_path: os.PathLike[str] | str | None = None,
) -> bool:
    """Copy the node's copy of an artifact to *local_path*.

    Workloads produce their artifacts -- a redirected stdout, the monitor's CSV
    -- on whichever node ran them, and every log- and sample-based validator
    reads them here. The framework ships ``upload_tree`` but no download, so the
    file comes back over the command channel.

    *remote_path* defaults to the workspace mapping of *local_path*. Returns
    False and leaves the local file alone for local executors and for files the
    node does not have, so callers can fetch unconditionally.
    """
    if not is_remote_executor(executor):
        return False
    source = remote_path if remote_path is not None else node_path_for(executor, local_path)
    result = executor.run(f"cat {shlex.quote(str(source))}")
    if not result.ok:
        return False
    local = pathlib.Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(result.stdout)
    return True


def workload_executor_from(
    target_executor: NodeExecutorGroup | AbstractExecutor,
) -> AbstractExecutor:
    """First executor behind a ``NodeExecutorGroup`` (``target_executor``)."""
    if hasattr(target_executor, "run") and hasattr(target_executor, "_executors"):
        return target_executor._executors[0]
    return target_executor


def format_shell_command(
    cmd: Sequence[str] | str,
    *,
    env: dict[str, str] | None = None,
    cwd: os.PathLike[str] | str | None = None,
    redirect_stdout: os.PathLike[str] | str | None = None,
) -> str:
    """Build a single shell command string for ``executor.run()``."""
    parts: list[str] = []
    if cwd is not None:
        parts.append(f"cd {shlex.quote(str(cwd))} &&")
    if env:
        parts.append("env " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()))
    if isinstance(cmd, str):
        parts.append(cmd)
    else:
        parts.append(" ".join(shlex.quote(str(c)) for c in cmd))
    cmd_str = " ".join(parts)
    if redirect_stdout is not None:
        cmd_str = f"{cmd_str} > {shlex.quote(str(redirect_stdout))} 2>&1"
    return cmd_str


def run_command(
    executor: AbstractExecutor | None,
    cmd: Sequence[str] | str,
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: os.PathLike[str] | str | None = None,
    check: bool = False,
    stream: bool = True,
) -> int:
    """Run a shell command via a framework executor when available."""
    cmd_str = format_shell_command(cmd, env=env, cwd=cwd)

    if executor is not None:
        try:
            result = executor.run(cmd_str, timeout=timeout, stream=stream)
            if check and not result.ok:
                raise subprocess.CalledProcessError(result.exit_code, cmd_str)
            return result.exit_code
        except TimeoutError:
            return 124

    merged = dict(os.environ)
    if env:
        merged.update(env)
    argv = cmd if isinstance(cmd, list) else shlex.split(cmd_str)
    try:
        proc = subprocess.run(
            argv,
            env=merged,
            timeout=timeout,
            check=check,
            capture_output=not check,
            cwd=str(cwd) if cwd is not None else None,
        )
        return proc.returncode
    except subprocess.TimeoutExpired:
        return 124
    except subprocess.CalledProcessError as exc:
        return exc.returncode


def run_command_captured(
    executor: AbstractExecutor | None,
    cmd: Sequence[str] | str,
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: os.PathLike[str] | str | None = None,
    check: bool = False,
) -> CommandResult:
    """Run a command and return stdout/stderr (pytest path uses executor)."""
    cmd_str = format_shell_command(cmd, env=env, cwd=cwd)

    if executor is not None:
        try:
            result = executor.run(cmd_str, timeout=timeout, stream=False)
            if check and not result.ok:
                raise subprocess.CalledProcessError(result.exit_code, cmd_str)
            return CommandResult(
                result.exit_code,
                result.stdout or "",
                result.stderr or "",
            )
        except TimeoutError:
            return CommandResult(124)

    merged = dict(os.environ)
    if env:
        merged.update(env)
    argv = cmd if isinstance(cmd, list) else shlex.split(cmd_str)
    try:
        proc = subprocess.run(
            argv,
            env=merged,
            timeout=timeout,
            check=check,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
        return CommandResult(proc.returncode, proc.stdout or "", proc.stderr or "")
    except subprocess.TimeoutExpired:
        return CommandResult(124)
    except subprocess.CalledProcessError as exc:
        return CommandResult(exc.returncode, exc.stdout or "", exc.stderr or "")


def run_command_redirect(
    executor: AbstractExecutor | None,
    cmd: Sequence[str] | str,
    stdout_file: os.PathLike[str] | str,
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
    cwd: os.PathLike[str] | str | None = None,
) -> int:
    """Run ``cmd`` with stdout/stderr redirected to ``stdout_file``.

    Every workload that needs a real file descriptor rather than a pipe comes
    through here, so the remote-node handling lives here too: the redirect names
    the run directory by its local path, which a remote shell has neither to
    write into nor to leave the result in. The directory is created on the node
    before the run and the file pulled back after, so callers can keep reading
    *stdout_file* locally whichever node produced it.
    """
    target = node_path_for(executor, stdout_file)
    ensure_remote_dir(executor, posixpath.dirname(target))
    if cwd is not None:
        # ``cwd`` is a run directory too, so it needs the same mapping or the
        # command's leading ``cd`` fails before the workload starts.
        cwd = node_path_for(executor, cwd)
        ensure_remote_dir(executor, cwd)
    redirect = format_shell_command(
        cmd,
        env=env,
        cwd=cwd,
        redirect_stdout=target,
    )
    wrapped = redirect
    if timeout:
        wrapped = f"timeout {int(timeout)} {redirect}"
    wall_timeout = float(timeout) + 5.0 if timeout else None
    rc = run_command(executor, wrapped, timeout=wall_timeout, stream=True)
    fetch_remote_file(executor, stdout_file, target)
    return rc


class BackgroundSessionAdapter:
    """Expose ``poll()`` like ``subprocess.Popen`` for executor sessions."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def poll(self) -> int | None:
        return self._handle.poll()
