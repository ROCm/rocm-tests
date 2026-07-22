# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Executor-transparent helpers to run a GoogleTest binary, drive its native
sharding (``GTEST_TOTAL_SHARDS``/``GTEST_SHARD_INDEX``), and parse its summary."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.abstract_executor import AbstractExecutor

logger = logging.getLogger(__name__)

# GoogleTest summary lines, e.g.:
#   [==========] 42 tests from 8 test suites ran. (1234 ms total)
#   [  PASSED  ] 41 tests.
#   [  FAILED  ] 1 test, listed below:
_RAN_RE = re.compile(r"\[=+\]\s+(\d+)\s+tests?\s+from\s+\d+\s+test\s+suites?\s+ran", re.IGNORECASE)
_PASSED_RE = re.compile(r"\[\s*PASSED\s*\]\s+(\d+)\s+tests?", re.IGNORECASE)
_FAILED_RE = re.compile(r"\[\s*FAILED\s*\]\s+(\d+)\s+tests?", re.IGNORECASE)

# stdout substrings that indicate the binary crashed or produced no runnable tests,
# independent of the numeric summary.
_CRASH_MARKERS: tuple[str, ...] = (
    "segmentation fault",
    "core dumped",
    "terminate called",
    "aborted",
)


@dataclass
class GtestResult:
    """Parsed outcome of a GoogleTest run.

    Attributes:
        ok:           Exit 0 and no FAILED case.
        exit_code:    Process exit code.
        total:        Tests that ran.
        passed:       Tests reported PASSED.
        failed:       Tests reported FAILED.
        shard_index:  GTEST_SHARD_INDEX used (None if unsharded).
        total_shards: GTEST_TOTAL_SHARDS used (None if unsharded).
        raw_output:   Combined stdout/stderr.
    """

    ok: bool
    exit_code: int
    total: int = 0
    passed: int = 0
    failed: int = 0
    shard_index: int | None = None
    total_shards: int | None = None
    raw_output: str = ""


def gtest_shard_env(shard_index: int, total_shards: int) -> dict[str, str]:
    """Return the GTEST shard env vars for one slice.

    Args:
        shard_index:  Zero-based slice index.
        total_shards: Slice count; <= 1 disables sharding.

    Returns:
        ``{"GTEST_TOTAL_SHARDS", "GTEST_SHARD_INDEX"}`` (empty when total_shards <= 1).

    Raises:
        ValueError: If shard_index is outside ``[0, total_shards)``.
    """
    if total_shards <= 1:
        return {}
    if not 0 <= shard_index < total_shards:
        raise ValueError(f"shard_index {shard_index} out of range for total_shards {total_shards}")
    return {"GTEST_TOTAL_SHARDS": str(total_shards), "GTEST_SHARD_INDEX": str(shard_index)}


def build_gtest_command(
    binary: str,
    *,
    gtest_filter: str | None = None,
    shard_index: int | None = None,
    total_shards: int | None = None,
    env: dict[str, str] | None = None,
    extra_args: str = "",
    cwd: str | None = None,
    omp_num_threads: int | str | None = None,
) -> str:
    """Build the shell command for a gtest run.

    Merges sharding/OMP into an ``env VAR=...`` prefix; never sets GPU
    device-selection env vars (the executor injects those).

    Args:
        binary:          Path to the gtest binary.
        gtest_filter:    ``--gtest_filter`` value.
        shard_index:     Zero-based shard index (with *total_shards*).
        total_shards:    Shard count; applied only when > 1.
        env:             Extra environment variables.
        extra_args:      Verbatim extra gtest flags.
        cwd:             Working directory (wrapped as ``cd <cwd> && ...``).
        omp_num_threads: Optional ``OMP_NUM_THREADS`` export.

    Returns:
        A ready-to-run shell command string.
    """
    merged: dict[str, str] = dict(env or {})
    if shard_index is not None and total_shards is not None:
        merged.update(gtest_shard_env(shard_index, total_shards))
    if omp_num_threads is not None:
        merged["OMP_NUM_THREADS"] = str(omp_num_threads)

    core = binary
    if gtest_filter:
        core += f" --gtest_filter={shlex.quote(gtest_filter)}"
    if extra_args:
        core += f" {extra_args}"
    if merged:
        prefix = "env " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in sorted(merged.items()))
        core = f"{prefix} {core}"
    if cwd:
        core = f"cd {shlex.quote(cwd)} && {core}"
    return core


def run_gtest(
    executor: AbstractExecutor,
    binary: str,
    *,
    gtest_filter: str | None = None,
    shard_index: int | None = None,
    total_shards: int | None = None,
    env: dict[str, str] | None = None,
    extra_args: str = "",
    cwd: str | None = None,
    omp_num_threads: int | str | None = None,
    timeout: float = 1800.0,
) -> GtestResult:
    """Run a gtest binary on *executor* and return a parsed :class:`GtestResult`.

    Transparent across local, SSH (remote GPU), and container executors.

    Args:
        executor:        Executor / ``NodeExecutorGroup`` with ``.run()``.
        binary:          Path to the gtest binary.
        gtest_filter:    ``--gtest_filter`` value.
        shard_index:     Zero-based shard index (with *total_shards*).
        total_shards:    Shard count; applied only when > 1.
        env:             Extra environment variables.
        extra_args:      Verbatim extra gtest flags.
        cwd:             Working directory for the binary.
        omp_num_threads: Optional ``OMP_NUM_THREADS`` export.
        timeout:         Maximum seconds to wait.

    Returns:
        A :class:`GtestResult` with the parsed summary and ``raw_output``.
    """
    cmd = build_gtest_command(
        binary,
        gtest_filter=gtest_filter,
        shard_index=shard_index,
        total_shards=total_shards,
        env=env,
        extra_args=extra_args,
        cwd=cwd,
        omp_num_threads=omp_num_threads,
    )
    result = executor.run(cmd, timeout=timeout)
    combined = "\n".join(part for part in (result.stdout, result.stderr) if part)
    parsed = parse_gtest_summary(combined)
    applied_shards = total_shards if (total_shards or 0) > 1 else None
    applied_index = shard_index if applied_shards is not None else None
    return GtestResult(
        ok=result.ok and parsed["failed"] == 0 and not _has_crash_marker(combined),
        exit_code=result.exit_code,
        total=parsed["total"],
        passed=parsed["passed"],
        failed=parsed["failed"],
        shard_index=applied_index,
        total_shards=applied_shards,
        raw_output=combined,
    )


def parse_gtest_summary(output: str) -> dict[str, int]:
    """Parse the gtest summary from combined output.

    Args:
        output: Combined process output.

    Returns:
        ``{"total", "passed", "failed"}``; missing fields are 0.
    """
    ran = _RAN_RE.search(output)
    passed = _PASSED_RE.search(output)
    failed = _FAILED_RE.search(output)
    return {
        "total": int(ran.group(1)) if ran else 0,
        "passed": int(passed.group(1)) if passed else 0,
        "failed": int(failed.group(1)) if failed else 0,
    }


def _has_crash_marker(output: str) -> bool:
    """Return True when *output* contains a hard crash marker (segfault, abort, ...)."""
    low = output.lower()
    return any(marker in low for marker in _CRASH_MARKERS)
