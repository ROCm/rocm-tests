# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""vLLM + Mooncake DMABUF distributed KV-cache system test.

Test ID: vllm_mooncake_dmabuf_distributed_kv_cache

Splits every GPU on the node into disaggregated vLLM prefill and decode instances whose KV blocks move
through MooncakeConnector over DMABUF-registered memory, drives long-context batches while side tensors hold
each GPU at 90% VRAM, and checks the amdgpu/HIP VRAM accounting stays consistent.

Pass criteria:
    - every request completes with the requested number of tokens
    - no fake OOM: no allocation refused while VRAM is free, no OOM or DMABUF registration error in vLLM logs
    - |HIP used - amdgpu used| and the post-run leak both stay below 1% of total VRAM

Scale: ``ROCM_TEST_WORKLOAD_SCALE=smoke`` (8K x 16) or ``full`` (8K x 128, 32K x 64, 128K x 16).
Workload: src/dmabuf_kv_cache_workload.py
"""

from __future__ import annotations

import json
import logging
import pathlib
import shlex
import time

import pytest

from tests.e2e.ml_frameworks.vllm_mooncake._constants import (
    FAKE_OOM_FREE_FLOOR_BYTES,
    GPU_COUNT_ARG,
    GPU_MEMORY_UTILIZATION,
    MAX_DRIFT_FRAC,
    MIN_GPU_VRAM_GB,
    MODEL,
    OUTPUT_TOKENS,
    PHASES,
    PRESSURE_TARGET,
    PRESSURE_TOLERANCE,
    REQUEST_TIMEOUT,
    RUN_TIMEOUT,
    STARTUP_TIMEOUT,
    SUMMARY_END,
    SUMMARY_START,
)
from tests.e2e.ml_frameworks.vllm_mooncake._plan import DeploymentPlan

logger = logging.getLogger(__name__)

_SRC_DIR = pathlib.Path(__file__).parent / "src"
_WORKLOAD = "dmabuf_kv_cache_workload.py"
_GIB = 1 << 30


def _workload_path(target_executor) -> str:
    executor = next(iter(target_executor))
    if hasattr(executor, "upload_tree"):
        return f"{executor.upload_tree(str(_SRC_DIR))}/{_WORKLOAD}"
    return str(_SRC_DIR / _WORKLOAD)


def _workload_command(
    python: str, script: str, ld_path: dict, scale: str, plan: DeploymentPlan, log_dir: pathlib.Path
) -> str:
    phases = ",".join(f"{seq}x{batch}" for seq, batch in PHASES[scale])
    args = (
        f"--model {shlex.quote(MODEL)} --tp {plan.tp} --phases {phases} --output-tokens {OUTPUT_TOKENS} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION} --pressure-target {PRESSURE_TARGET} "
        f"--drift-limit {MAX_DRIFT_FRAC} --startup-timeout {STARTUP_TIMEOUT} --request-timeout {REQUEST_TIMEOUT} "
        f"--log-dir {shlex.quote(str(log_dir))}"
    )
    ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
    return f"env LD_LIBRARY_PATH={ld} {shlex.quote(python)} {shlex.quote(script)} {args}"


def _log_dir(framework_config) -> pathlib.Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return (pathlib.Path(framework_config.framework.artifact_dir) / "vllm_mooncake" / stamp).resolve()


def _extract_summary(stdout: str) -> dict | None:
    start = stdout.find(SUMMARY_START)
    end = stdout.find(SUMMARY_END)
    if start == -1 or end < start:
        return None
    return json.loads(stdout[start + len(SUMMARY_START) : end])


def _failures(summary: dict) -> list[str]:
    failures = []
    for phase in summary["phases"]:
        if phase["completed"] != phase["batch"]:
            failures.append(
                f"{phase['seq_len']} tokens x {phase['batch']}: only {phase['completed']} completed; "
                f"first error: {phase['errors'][:1]}"
            )

    for gpu in summary["pressure"]:
        if gpu["fake_oom"]:
            failures.append(f"GPU {gpu['gpu']}: allocation refused while VRAM was still free (fake OOM)")
        if gpu["used_frac"] < PRESSURE_TARGET - PRESSURE_TOLERANCE:
            failures.append(f"GPU {gpu['gpu']}: pressure reached only {gpu['used_frac']:.1%}")
        if not gpu["intact"]:
            failures.append(f"GPU {gpu['gpu']}: side tensors were corrupted during inference")

    accounting = summary["accounting"]
    if accounting["max_drift_frac"] >= MAX_DRIFT_FRAC:
        failures.append(f"HIP vs amdgpu VRAM drift {accounting['max_drift_frac']:.2%} >= {MAX_DRIFT_FRAC:.0%}")
    if accounting["leak_frac"] >= MAX_DRIFT_FRAC:
        failures.append(f"VRAM not returned after teardown: {accounting['leak_frac']:.2%} >= {MAX_DRIFT_FRAC:.0%}")

    errors = summary["log_errors"]
    if errors["oom"]:
        kind = "fake OOM" if accounting["min_free_bytes"] > FAKE_OOM_FREE_FLOOR_BYTES else "OOM"
        failures.append(
            f"{kind} in vLLM logs with {accounting['min_free_bytes'] / _GIB:.1f} GiB free at the lowest sample: "
            f"{errors['oom']}"
        )
    if errors["dmabuf"]:
        failures.append(f"DMABUF registration errors: {errors['dmabuf']}")
    return failures


@pytest.mark.runtime.soak
@pytest.mark.gpu_count(GPU_COUNT_ARG)
@pytest.mark.gpu_vram(MIN_GPU_VRAM_GB)
def test_vllm_mooncake_dmabuf_distributed_kv_cache(
    target_executor,
    vllm_python: str,
    require_rdma,
    ld_path: dict,
    workload_scale: str,
    deployment_plan: DeploymentPlan,
    framework_config,
):
    """Long-context disaggregated inference under 90% VRAM pressure keeps VRAM accounting consistent."""
    script = _workload_path(target_executor)
    log_dir = _log_dir(framework_config)
    command = _workload_command(vllm_python, script, ld_path, workload_scale, deployment_plan, log_dir)
    logger.info("Starting %s workload on %s", workload_scale, deployment_plan)
    result = target_executor.run(command, timeout=RUN_TIMEOUT[workload_scale])

    summary = _extract_summary(result.stdout)
    assert result.ok, (
        f"workload failed (exit={result.exit_code}), logs kept in {log_dir}:\n"
        f"stdout tail: {result.stdout[-3000:]}\nstderr tail: {result.stderr[-12000:]}"
    )
    assert summary is not None, f"workload printed no summary:\n{result.stdout[-3000:]}"
    logger.info("Summary: %s", json.dumps(summary))

    failures = _failures(summary)
    assert not failures, "DMABUF KV-cache test failed:\n  " + "\n  ".join(failures) + f"\nlogs: {summary['log_dir']}"
