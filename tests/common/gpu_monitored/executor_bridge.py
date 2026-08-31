# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bridge gpu_monitored workloads to rocm-tests executors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import os
import shlex
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor
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


class _UnmaskedSshMonitorExecutor:
    """Wrap ``SshExecutor`` so monitor commands run without GPU masks."""

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
    """Run ``cmd`` with stdout/stderr redirected to ``stdout_file``."""
    redirect = format_shell_command(
        cmd,
        env=env,
        cwd=cwd,
        redirect_stdout=stdout_file,
    )
    wrapped = redirect
    if timeout:
        wrapped = f"timeout {int(timeout)} {redirect}"
    wall_timeout = float(timeout) + 5.0 if timeout else None
    return run_command(executor, wrapped, timeout=wall_timeout, stream=True)


class BackgroundSessionAdapter:
    """Expose ``poll()`` like ``subprocess.Popen`` for executor sessions."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def poll(self) -> int | None:
        return self._handle.poll()
