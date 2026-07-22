# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
gtest.py -- GoogleTest binary execution, sharding, and output parsing helpers.

Many third-party ROCm test suites ship a single GoogleTest binary that
self-validates and exits non-zero on any failed case.
This module provides a *generic*, executor-transparent way to:

    * run such a binary with a ``--gtest_filter`` and arbitrary env,
    * drive GoogleTest's built-in sharding
      (``GTEST_TOTAL_SHARDS`` / ``GTEST_SHARD_INDEX``) so a large suite can be
      split into N independent slices distributed across pytest workers / GPUs,
    * parse the ``[==========] N tests ran`` / ``[  PASSED  ]`` / ``[  FAILED  ]``
      summary into a structured result for assertions.

Every helper delegates execution to the caller's executor, so the same code
works under ``LocalExecutor``, ``SshExecutor`` (remote GPU), and
``ContainerExecutor`` without change.

Usage::

    from framework.rocm.libs.gtest import run_gtest

    def test_suite(target_executor):
        res = run_gtest(
            target_executor,
            "build/test/gtest/gtest",
            gtest_filter="*rocm*",
            shard_index=0,
            total_shards=4,
            env={"LD_LIBRARY_PATH": ld},
        )
        assert res.ok, res.raw_output[-4000:]
        assert res.failed == 0
        assert res.total > 0
"""

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
    """Structured outcome of a GoogleTest binary run.

    Attributes:
        ok:           True when the process exited 0 *and* no test reported FAILED.
        exit_code:    Process exit code from the executor.
        total:        Number of tests that ran (from the ``... ran`` banner).
        passed:       Number of tests reported ``[  PASSED  ]``.
        failed:       Number of tests reported ``[  FAILED  ]`` (0 when absent).
        shard_index:  ``GTEST_SHARD_INDEX`` used for this run (``None`` if unsharded).
        total_shards: ``GTEST_TOTAL_SHARDS`` used for this run (``None`` if unsharded).
        raw_output:   Full combined stdout/stderr from the run.
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
    """Return the GoogleTest sharding environment variables for one slice.

    GoogleTest natively shards a single binary across ``total_shards`` invocations:
    each process runs only the test cases whose stable index modulo the shard count
    equals ``shard_index`` (see the GoogleTest ``GTEST_TOTAL_SHARDS`` /
    ``GTEST_SHARD_INDEX`` protocol). Distribute the slices across pytest-xdist
    workers (one GPU each) to parallelise a long suite.

    Args:
        shard_index:  Zero-based slice index, ``0 <= shard_index < total_shards``.
        total_shards: Total number of slices. Values ``<= 1`` disable sharding and
                      return an empty dict (the binary then runs every test).

    Returns:
        ``{"GTEST_TOTAL_SHARDS": N, "GTEST_SHARD_INDEX": i}`` as strings, or an
        empty dict when ``total_shards <= 1``.

    Raises:
        ValueError: If ``shard_index`` is outside ``[0, total_shards)``.
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
    """Assemble the shell command string for a GoogleTest binary run.

    Never sets any GPU device-selection env var (``ROCR_VISIBLE_DEVICES`` etc.) —
    the executor injects those. Sharding and ``OMP_NUM_THREADS`` are merged into the
    ``env VAR=... `` prefix so callers do not format the command by hand.

    Args:
        binary:          Path to the gtest binary (absolute, or relative to *cwd*).
        gtest_filter:    Value for ``--gtest_filter`` (e.g. ``"*rocm*"``); omitted when None.
        shard_index:     Zero-based shard index; combined with *total_shards*.
        total_shards:    Total shard count; sharding is applied only when ``> 1``.
        env:             Extra environment variables (e.g. ``LD_LIBRARY_PATH``).
        extra_args:      Verbatim extra gtest flags appended after the filter.
        cwd:             Working directory; wraps the command in ``cd <cwd> && ...``.
        omp_num_threads: When set, exports ``OMP_NUM_THREADS`` for the run.

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
    """Run a GoogleTest binary on *executor* and return a parsed :class:`GtestResult`.

    Delegates execution to ``executor.run()`` so it is transparent across local,
    SSH (remote GPU), and container contexts.

    Args:
        executor:        Any executor / ``NodeExecutorGroup`` with a ``.run()`` method.
        binary:          Path to the gtest binary.
        gtest_filter:    ``--gtest_filter`` value (e.g. ``"*rocm*"``).
        shard_index:     Zero-based shard index (with *total_shards*).
        total_shards:    Total shard count; sharding applies only when ``> 1``.
        env:             Extra environment variables for the run.
        extra_args:      Verbatim extra gtest flags.
        cwd:             Working directory for the binary.
        omp_num_threads: Optional ``OMP_NUM_THREADS`` export.
        timeout:         Maximum seconds to wait.

    Returns:
        A :class:`GtestResult` with the parsed summary and full ``raw_output``.
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
    """Parse the GoogleTest run summary from combined stdout/stderr.

    Args:
        output: Combined process output.

    Returns:
        ``{"total": int, "passed": int, "failed": int}``; missing fields are 0.
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
