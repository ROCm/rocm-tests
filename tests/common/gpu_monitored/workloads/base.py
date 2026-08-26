# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Base Test class + context dataclasses."""

from __future__ import annotations

import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.executor_bridge import (
    run_command,
    run_command_captured,
    run_command_redirect,
)

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor
    from framework.executors.executor_group import NodeExecutorGroup


# ---------------------------------------------------------------------------
# Status enums
# ---------------------------------------------------------------------------
class BuildStatus(Enum):
    OK = "ok"                    # binary available (installed or built)
    BUILD_FAILED = "build_failed"
    SOURCE_MISSING = "source_missing"
    SKIPPED = "skipped"          # test not selected, skipped


# ---------------------------------------------------------------------------
# Test specification
# ---------------------------------------------------------------------------
@dataclass
class TestSpec:
    name: str                              # e.g. "cudamemtest"
    goal: str                              # human-readable goal for banner
    # Monitoring workload profile (used by analyze_monitoring.py)
    # Example: {"min_util": 70, "min_vram_pct": 30, "serial": False}
    workload_profile: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Run/Build contexts
# ---------------------------------------------------------------------------
@dataclass
class BuildContext:
    config: Config
    monitor_executor: Optional["AbstractExecutor"] = None

    @property
    def rocm_root(self) -> Path:
        return self.config.rocm_root

    @property
    def build_dir(self) -> Path:
        return self.config.build_dir

    @property
    def script_dir(self) -> Path:
        return self.config.script_dir


@dataclass
class RunResult:
    exit_code: int = 0
    reproduce_cmd: str = ""
    # UNSUPPORTED is a classification, not a process exit code. Prefer this
    # flag over squatting on a magic exit value.
    unsupported: bool = False


@dataclass
class RunContext:
    config: Config
    run_dir: Path
    log_root: Path
    target_executor: Optional["NodeExecutorGroup"] = None
    monitor_executor: Optional["AbstractExecutor"] = None
    console_log: Optional[Path] = None

    @property
    def rocm_root(self) -> Path:
        return self.config.rocm_root

    @property
    def build_dir(self) -> Path:
        return self.config.build_dir

    @property
    def num_gpus(self) -> int:
        return self.config.num_gpus

    def append_console(self, text: str) -> None:
        """Append workload output to ``console.log`` for validation."""
        if not text or self.console_log is None:
            return
        self.console_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.console_log, "a", encoding="utf-8", errors="replace") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")

    def exec(self, cmd: Sequence, env: Optional[Dict[str, str]] = None,
             timeout: Optional[int] = None, check: bool = False,
             stdout_file: Optional[Path] = None) -> int:
        """Execute a workload command via ``target_executor`` when wired.

        Falls back to subprocess only for standalone CLI use (no executor).
        When ``stdout_file`` is set, output is redirected to that file via
        the executor shell (RVS requires a real file fd, not a pipe).
        """
        executor = self.target_executor or self.monitor_executor

        if stdout_file is not None and executor is not None:
            stdout_file = Path(stdout_file)
            stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stdout_file.write_bytes(b"")
            done = threading.Event()

            def _follow() -> None:
                try:
                    reader = open(stdout_file, "rb")
                except OSError:
                    return
                try:
                    while True:
                        chunk = reader.read(65536)
                        if chunk:
                            try:
                                os.write(1, chunk)
                            except OSError:
                                return
                            continue
                        if done.is_set():
                            return
                        time.sleep(0.2)
                finally:
                    reader.close()

            try:
                sys.stdout.flush()
            except (OSError, ValueError):
                pass
            follower = threading.Thread(target=_follow, daemon=True)
            follower.start()
            try:
                return run_command_redirect(
                    self.target_executor or self.monitor_executor,
                    list(cmd),
                    stdout_file,
                    timeout=float(timeout) if timeout is not None else None,
                    env=env,
                )
            finally:
                done.set()
                follower.join(timeout=30)
                if self.console_log is not None and stdout_file.is_file():
                    try:
                        self.append_console(stdout_file.read_text(errors="replace"))
                    except OSError:
                        pass

        executor = self.target_executor or self.monitor_executor
        if stdout_file is None and self.console_log is not None and executor is not None:
            res = run_command_captured(
                executor,
                list(cmd),
                timeout=float(timeout) if timeout is not None else None,
                env=env,
                check=check,
            )
            if res.stdout or res.stderr:
                self.append_console(res.stdout + res.stderr)
            return res.exit_code

        if stdout_file is None and self.target_executor is not None:
            return run_command(
                self.target_executor,
                list(cmd),
                timeout=float(timeout) if timeout is not None else None,
                env=env,
                check=check,
            )

        return run_command(
            self.monitor_executor,
            list(cmd),
            timeout=float(timeout) if timeout is not None else None,
            env=env,
            check=check,
        )


# ---------------------------------------------------------------------------
# Base Test class
# ---------------------------------------------------------------------------
class Test(ABC):
    """Abstract base for all monitored tests."""

    spec: TestSpec

    def build(self, ctx: BuildContext) -> BuildStatus:
        """Default: nothing to build. Override for tests with build steps."""
        return BuildStatus.OK

    @abstractmethod
    def run(self, ctx: RunContext) -> RunResult:
        """Execute the workload. Returns RunResult with exit code + reproduce cmd."""
        raise NotImplementedError

    def available(self, config: Config) -> bool:
        """Return True if the test's binaries/prerequisites are available.
        Default: True. Override to check for built binaries."""
        return True
