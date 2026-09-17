# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PTL runtime toggle under RCCL load.

Validates that per-GPU ptl_enable sysfs writes are race-free and non-disruptive
while RCCL all_reduce_perf collectives run concurrently across all GPUs via MPI.
"""

from __future__ import annotations

import os
import re

import pytest

_SCRIPT = "tests/e2e/hip_runtime/src/ptl_toggle/run.sh"
_SKIP_EXIT_CODE = 77

# Override via ROCM_TEST_PTL_GPU_COUNT=4 or =all (default: all GPUs on the node).
_GPU_COUNT: int | str = os.environ.get("ROCM_TEST_PTL_GPU_COUNT", "all")
if isinstance(_GPU_COUNT, str) and _GPU_COUNT.lower() != "all":
    _GPU_COUNT = int(_GPU_COUNT)


def _parse_verdict_field(stdout: str, field: str) -> str | None:
    """Return the value of a VERDICT field from run.sh stdout, or None."""
    m = re.search(rf"^\s*{re.escape(field)}\s*:\s*(.+)$", stdout, re.MULTILINE)
    return m.group(1).strip() if m else None


def _parse_int_field(stdout: str, field: str) -> int:
    """Return a VERDICT integer field; raise AssertionError if absent or non-numeric."""
    raw = _parse_verdict_field(stdout, field)
    assert raw is not None, f"VERDICT field '{field}' not found in script output"
    numeric = re.search(r"\d+", raw)
    assert numeric, f"VERDICT field '{field}' has no numeric value: {raw!r}"
    return int(numeric.group())


@pytest.mark.hw.multi_gpu
@pytest.mark.runtime.medium
@pytest.mark.gpu_count(_GPU_COUNT)
def test_ptl_runtime_toggle_rccl_workload(
    target_executor,
    rock_dir: str,
) -> None:
    """Stress-test PTL runtime toggle while RCCL all_reduce_perf runs on all GPUs.

    Runs run.sh, which toggles per-GPU ptl_enable sysfs at 0.5 s intervals for the
    full span of RCCL_ITER_COUNT (5) all_reduce sweeps, then verifies the verdict block.
    """
    result = target_executor.run(
        f"env PATH={rock_dir}/bin:$PATH"
        f" ROCM_PATH={rock_dir}"
        f" RCCL_BIN={rock_dir}/bin/all_reduce_perf"
        f" bash {_SCRIPT}"
    )

    if result.exit_code == _SKIP_EXIT_CODE:
        skip_line = _parse_verdict_field(result.stdout, "SKIP") or (
            result.stderr.splitlines()[0] if result.stderr else "see stdout"
        )
        pytest.skip(f"run.sh requested skip: {skip_line}")

    assert result.ok, (
        f"run.sh failed (exit={result.exit_code}):\n" f"stdout: {result.stdout[:3000]}\nstderr: {result.stderr[:500]}"
    )

    stdout = result.stdout

    result_field = _parse_verdict_field(stdout, "result")
    assert result_field is not None, f"VERDICT 'result' field not found:\n{stdout[:2000]}"
    assert result_field.upper() == "PASS", (
        f"run.sh reported result={result_field!r}. "
        f"fail_reasons: {_parse_verdict_field(stdout, 'fail_reasons') or 'not listed'}\n"
        f"stdout: {stdout[:3000]}"
    )

    ptl_writes_bad = _parse_int_field(stdout, "ptl_writes_bad")
    assert ptl_writes_bad == 0, (
        f"ptl_writes_bad={ptl_writes_bad}: sysfs readback did not confirm writes.\n" f"stdout: {stdout[:2000]}"
    )

    rccl_runs_failed = _parse_int_field(stdout, "rccl_runs_failed")
    assert rccl_runs_failed == 0, (
        f"rccl_runs_failed={rccl_runs_failed}: one or more all_reduce_perf iterations exited non-zero.\n"
        f"stdout: {stdout[:2000]}"
    )

    dmesg_critical_delta = _parse_int_field(stdout, "dmesg_critical_delta")
    assert dmesg_critical_delta == 0, (
        f"dmesg_critical_delta={dmesg_critical_delta}: critical kernel/driver events during PTL toggle.\n"
        f"stdout: {stdout[:2000]}"
    )

    n_gpus_raw = _parse_verdict_field(stdout, "n_gpus")
    assert n_gpus_raw is not None, f"VERDICT field 'n_gpus' not found:\n{stdout[:2000]}"
    counts = re.findall(r"\d+", n_gpus_raw)
    assert len(counts) >= 2, f"Could not parse before/after GPU count from n_gpus={n_gpus_raw!r}"
    before_count, after_count = int(counts[0]), int(counts[1])
    assert before_count == after_count, (
        f"GPU count changed during run: before={before_count} after={after_count}.\n" f"stdout: {stdout[:2000]}"
    )

    ptl_state_restored = _parse_verdict_field(stdout, "ptl_state_restored")
    assert ptl_state_restored == "yes", (
        f"ptl_state_restored={ptl_state_restored!r}: original PTL state was not fully restored.\n"
        f"stdout: {stdout[:2000]}"
    )
