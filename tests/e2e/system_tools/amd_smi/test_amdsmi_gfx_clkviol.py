# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validate amd-smi GFX clock-violation telemetry under an RVS IET load.

The RVS binary, runtime environment, and GPU-specific ``iet_stress.conf`` are
provided by the RVS fixtures re-exported in this directory's ``conftest.py``.
The test then samples
the equivalent fields exposed by ``amd-smi monitor --violation`` and
``amd-smi metric --throttle``.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import shlex
import sys
import time
from typing import Any

import pytest

from framework.reporting.allure_reporter import attach_json, attach_text, report_metric, step
from framework.rocm.libs.stack import get_rocm_version

logger = logging.getLogger(__name__)

_IET_CONF = "iet_stress.conf"
_IET_STARTUP_SECS = 5
_IET_STOP_WAIT_SECS = 60.0
_WATCH_INTERVAL_SECS = int(os.environ.get("ROCM_TEST_GFX_CLKVIOL_INTERVAL_SECS", "1"))
_WATCH_ITERATIONS = int(os.environ.get("ROCM_TEST_GFX_CLKVIOL_ITERATIONS", "100"))
# On an eight-GPU host one nominal one-second iteration can take nearly two
# seconds because amd-smi serializes per-device queries and JSON emission.
_AMDSMI_TIMEOUT_SECS = max(180.0, _WATCH_INTERVAL_SECS * _WATCH_ITERATIONS * 3.0 + 60.0)

_MONITOR_FIELDS = ("gfxclk_pviol", "gfxclk_totalviol")
_METRIC_FIELDS = (
    "gfx_clk_below_host_limit_power_violation_activity",
    "total_gfx_clk_below_host_limit_violation_activity",
)


def _normalize_samples(data: Any) -> list[dict]:
    """Return per-sample dictionaries across supported amd-smi JSON wrappers."""
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []

    for wrapper in ("gpu_data", "data"):
        wrapped = data.get(wrapper)
        if isinstance(wrapped, list):
            return [item for item in wrapped if isinstance(item, dict)]
        if isinstance(wrapped, dict):
            return [item for item in wrapped.values() if isinstance(item, dict)]

    return [data]


def _percentage_values(node: Any) -> list[float] | None:
    """Flatten one amd-smi percentage field, rejecting ``N/A`` or bad shapes."""
    if isinstance(node, bool) or node is None or node == "N/A":
        return None
    if isinstance(node, (int, float)):
        return [float(node)]
    if isinstance(node, dict):
        if "value" in node:
            return _percentage_values(node["value"])
        children = [_percentage_values(value) for value in node.values()]
    elif isinstance(node, list):
        children = [_percentage_values(value) for value in node]
    else:
        return None

    if not children or any(child is None for child in children):
        return None
    return [value for child in children for value in child]


def _valid_percentage_field(node: Any) -> bool:
    values = _percentage_values(node)
    return bool(values) and all(0.0 <= value <= 100.0 for value in values)


def _valid_monitor_samples(data: Any) -> list[dict]:
    """Return monitor samples containing both required, usable violation fields."""
    return [
        sample
        for sample in _normalize_samples(data)
        if all(field in sample and _valid_percentage_field(sample[field]) for field in _MONITOR_FIELDS)
    ]


def _valid_metric_samples(data: Any) -> list[dict]:
    """Return metric samples containing both required, usable throttle fields."""
    valid = []
    for sample in _normalize_samples(data):
        throttle = sample.get("throttle")
        if not isinstance(throttle, dict):
            continue
        if all(field in throttle and _valid_percentage_field(throttle[field]) for field in _METRIC_FIELDS):
            valid.append(sample)
    return valid


def _run_json_capture(executor, command: str, output_path: str, label: str) -> Any:
    """Run one amd-smi watch command and parse its target-node JSON file."""
    quoted_path = shlex.quote(output_path)
    result = executor.run(
        f"rm -f {quoted_path} && {command} --json --file {quoted_path} --overwrite",
        timeout=_AMDSMI_TIMEOUT_SECS,
    )
    assert result.ok, (
        f"{label} failed (exit={result.exit_code}):\n"
        f"stdout:\n{(result.stdout or '')[-2000:]}\nstderr:\n{(result.stderr or '')[-2000:]}"
    )

    captured = executor.run(f"cat {quoted_path}")
    assert captured.ok, f"{label} produced no JSON in {output_path}"
    assert (captured.stdout or "").strip(), f"{label} wrote an empty JSON file at {output_path}"
    try:
        return json.loads(captured.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{label} produced invalid JSON: {exc}\n{captured.stdout[-2000:]}") from exc


def _safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _skip_if_unsupported(executor, rock_dir: str) -> None:
    """Match the original Windows / ROCm < 7.0 skips."""
    if sys.platform.startswith("win"):
        pytest.skip("amd-smi monitor is not supported on windows")

    version = ""
    if rock_dir:
        result = executor.run(f"cat {shlex.quote(rock_dir.rstrip('/'))}/.info/version 2>/dev/null")
        if result.ok:
            version = (result.stdout or "").strip().split()[0]
    if not version:
        version = get_rocm_version(executor) or ""
    match = re.match(r"(\d+)\.(\d+)", version)
    if match and (int(match.group(1)), int(match.group(2))) < (7, 0):
        pytest.skip("amd-smi is not supported for rocm version below 7.0.0")


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_amdsmi_gfx_clock_violation(
    target_executor,
    rvs_binary,
    rvs_find_conf,
    gpu_conf_dir,
    rvs_env,
    rock_dir,
    run_ctx,
    request,
):
    """GFX clock-violation percentages must be available from both amd-smi views."""
    _skip_if_unsupported(target_executor, rock_dir)

    conf = rvs_find_conf(_IET_CONF, gpu_conf_dir=gpu_conf_dir)
    rvs_cmd = (
        f"exec env {rvs_env} {shlex.quote(str(pathlib.Path(rvs_binary).resolve()))} "
        f"-c {shlex.quote(str(pathlib.Path(conf).resolve()))} -d 3"
    )

    tag = _safe_tag(f"{run_ctx.run_id}_{request.node.name}")
    scratch = f"/tmp/rocm_test_gfx_clkviol_{tag}"  # nosec B108 - target-node scratch
    monitor_path = f"{scratch}/monitor.json"
    metric_path = f"{scratch}/metric.json"
    target_executor.run(f"rm -rf {shlex.quote(scratch)} && mkdir -p {shlex.quote(scratch)}")

    logger.info("starting RVS IET stress with %s", conf)
    workload = target_executor.start_background(
        rvs_cmd,
        log_path=os.path.join("output", "artifacts", "amd_smi", "gfx_clkviol_rvs_iet.log"),
        console_label="rvs/iet-gfx-clkviol",
    )
    try:
        time.sleep(_IET_STARTUP_SECS)
        assert workload.is_alive, "RVS IET workload exited during startup"

        watch = f"-w {_WATCH_INTERVAL_SECS} -i {_WATCH_ITERATIONS}"
        with step("Sample amd-smi monitor violation telemetry"):
            monitor_data = _run_json_capture(
                target_executor,
                f"amd-smi monitor --violation {watch}",
                monitor_path,
                "amd-smi monitor --violation",
            )
            monitor_samples = _valid_monitor_samples(monitor_data)
            attach_json(json.dumps(monitor_data, indent=2), name="amdsmi_monitor_violation")
            assert monitor_samples, (
                f"no monitor sample contained usable {_MONITOR_FIELDS}; "
                "fields must be present, non-N/A, and within 0-100%"
            )
            assert workload.is_alive, "RVS IET workload exited while monitor telemetry was sampled"

        with step("Sample amd-smi metric throttle telemetry"):
            metric_data = _run_json_capture(
                target_executor,
                f"amd-smi metric --throttle {watch}",
                metric_path,
                "amd-smi metric --throttle",
            )
            metric_samples = _valid_metric_samples(metric_data)
            attach_json(json.dumps(metric_data, indent=2), name="amdsmi_metric_throttle")
            assert metric_samples, (
                f"no metric sample contained usable {_METRIC_FIELDS}; "
                "fields must be present, non-N/A, and within 0-100%"
            )
            assert workload.is_alive, "RVS IET workload exited while metric telemetry was sampled"

        report_metric("AMDSMI_MONITOR_VALID_SAMPLES", float(len(monitor_samples)))
        report_metric("AMDSMI_METRIC_VALID_SAMPLES", float(len(metric_samples)))
    finally:
        workload_result = workload.stop(timeout=30.0)
        attach_text(
            (workload_result.stdout or "") + (workload_result.stderr or ""),
            name="rvs_iet_gfx_clkviol",
        )
        logger.info("Wait %ss to kill RVS Workload", int(_IET_STOP_WAIT_SECS))
        time.sleep(_IET_STOP_WAIT_SECS)
        target_executor.run(f"rm -rf {shlex.quote(scratch)}")
