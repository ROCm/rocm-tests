# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Fixtures for GPU-monitored e2e tests.

Uses rocm-tests ``target_executor`` for GPU workloads and a separate
monitor executor (``CpuExecutor`` / unmasked ``SshExecutor``) for
``amd-smi monitor``, matching ``remote_node_plugin._monitoring_executor``.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

import pytest

from tests.common.gpu_monitored.executor_bridge import (
    make_monitor_executor,
    workload_executor_from,
)
from tests.common.gpu_monitored.framework_bridge import (
    ensure_gpu_environment,
    make_monitored_config,
    resolve_gpu_identity,
)
from tests.common.gpu_monitored.orchestrator import MonitoredTestOrchestrator, TestOutcome
from tests.common.gpu_monitored.validation import pretest_health_probe
from tests.common.gpu_monitored.workloads import get_test
from tests.common.gpu_monitored.workloads.base import BuildContext, BuildStatus
from tests.e2e.rvs.conftest import (  # noqa: F401
    export_rvs_env_paths,
    gpu_conf_dir,
    rvs_binary as _rvs_binary,
    rvs_find_conf,
    rvs_source as _rvs_source,
    transferbench_binary as _transferbench_binary,
)

logger = logging.getLogger(__name__)

# pytest only auto-loads conftest.py for its own directory tree, so the RVS
# build fixtures from the sibling ``tests/e2e/rvs`` suite are re-exported here
# to make them requestable by the monitored workloads below.
rvs_binary = _rvs_binary
rvs_source = _rvs_source
transferbench_binary = _transferbench_binary

_RVS_WORKLOADS = frozenset({"rvs_tst", "rvs_iet_stress"})
_TRANSFERBENCH_WORKLOADS = frozenset({"transferbench"})
_CUDAMEMTEST_WORKLOADS = frozenset({"cudamemtest"})

_CUDA_MEMTEST_REPO_URL = "https://github.com/ComputationalRadiationPhysics/cuda_memtest.git"


@pytest.fixture(scope="session")
def _gpu_monitored_rvs_env(rvs_binary, rvs_source, rock_dir, compiler_build_dir):
    """Export RVS paths — only requested by workloads that drive ``rvs``."""
    export_rvs_env_paths(
        rvs_binary,
        rvs_source,
        rock_dir,
        compiler_build_dir=compiler_build_dir,
    )


@pytest.fixture(scope="session")
def cuda_memtest_source(external_build, compiler_build_dir: str) -> str:
    """Clone cuda_memtest once per session at the pinned commit (framework git helper)."""
    from tests.common.gpu_monitored.workloads.cudamemtest import CudaMemtest

    dest = pathlib.Path(compiler_build_dir) / "gpu_monitored" / "cuda_memtest"
    src_dir = external_build.clone_repo(
        _CUDA_MEMTEST_REPO_URL,
        dest,
        ref=CudaMemtest.COMMIT,
        timeout=600.0,
    )
    external_build.assert_license_present(src_dir)
    os.environ["ROCM_TEST_CUDA_MEMTEST_SRC"] = str(src_dir)
    return str(src_dir)


@pytest.fixture(scope="session")
def _gpu_monitored_transferbench_env(transferbench_binary, rock_dir, compiler_build_dir):
    """Export TransferBench path without pulling in a full RVS build."""
    export_rvs_env_paths(
        None,
        None,
        rock_dir,
        transferbench_binary=transferbench_binary,
        compiler_build_dir=compiler_build_dir,
    )


def _ensure_workload_prerequisites(request, test_name: str) -> None:
    """Lazily resolve RVS / TransferBench fixtures only when a test needs them."""
    if test_name in _RVS_WORKLOADS:
        request.getfixturevalue("_gpu_monitored_rvs_env")
    elif test_name in _TRANSFERBENCH_WORKLOADS:
        request.getfixturevalue("_gpu_monitored_transferbench_env")
    elif test_name in _CUDAMEMTEST_WORKLOADS:
        request.getfixturevalue("cuda_memtest_source")


@pytest.fixture(scope="session")
def gpu_monitor_interval() -> int:
    """Monitoring sample interval in seconds, overridable via env.

    Defaults to 1 Hz to match the standalone suite (``--sample-interval``
    default 1): the analysis thresholds — steady-state windows, ramp-up
    detection, active-sample counts — are calibrated for per-second
    telemetry. ``framework_config.gpu.monitor_interval_secs`` is
    deliberately not consulted; that 15 s knob configures the framework's
    separate ``--monitor-gpu`` background poller, and borrowing it here
    made a 5-minute thermal test land only ~4 loaded samples per GPU.
    """
    env_val = os.environ.get("GPU_MONITOR_INTERVAL", "").strip()
    return int(env_val) if env_val else 1


@pytest.fixture
def gpu_monitored_monitor_executor(target_executor, rock_dir):
    """Executor for ``amd-smi monitor`` — no ``ROCR_VISIBLE_DEVICES`` mask."""
    return make_monitor_executor(
        workload_executor_from(target_executor),
        rock_dir=rock_dir,
    )


@pytest.fixture
def monitored_config(
    request,
    rock_dir,
    ld_path,
    compiler_build_dir,
    framework_config,
    gpu_monitor_interval,
    target_executor,
    gpu_monitored_monitor_executor,
    gpu_arch,
):
    """Per-test :class:`Config` from framework GPU detection + ROCm paths."""
    num_gpus = target_executor.visible_gpu_count
    arch, model, _bdf = resolve_gpu_identity(
        gpu_monitored_monitor_executor,
        gpu_arch,
        rock_dir,
    )

    from tests.common.gpu_monitored.environment import detect_gpu_device_id

    cfg = make_monitored_config(
        rock_dir=rock_dir,
        ld_path=ld_path,
        compiler_build_dir=compiler_build_dir,
        artifact_dir=framework_config.framework.artifact_dir,
        sample_interval=gpu_monitor_interval,
        rocmtest_path=os.environ.get("ROCM_TEST_ROCMTEST_PATH"),
        num_gpus=num_gpus,
        gpu_arch=arch,
        gpu_model=model,
        gpu_device_id=detect_gpu_device_id(),
    )
    ensure_gpu_environment(cfg)
    return cfg


def _test_name_from_request(request) -> str:
    name = request.node.name
    if name.startswith("test_gpu_") and name.endswith("_monitored"):
        return name[len("test_gpu_") : -len("_monitored")]
    return name.replace("test_", "")


@pytest.fixture
def run_monitored_test(
    request,
    monitored_config,
    target_executor,
    gpu_monitored_monitor_executor,
):
    """Run a registered gpu_monitored workload with full monitoring pipeline."""

    def _run(test_name: str | None = None) -> TestOutcome:
        name = test_name or _test_name_from_request(request)
        test = get_test(name)
        if test is None:
            pytest.fail(f"Unknown gpu_monitored test: {name}")

        _ensure_workload_prerequisites(request, name)

        config = monitored_config
        lookback = int(os.environ.get("GPU_MONITOR_PRETEST_LOOKBACK_MIN", "30"))
        clean, health_summary = pretest_health_probe(
            lookback_min=lookback,
            cpu_executor=gpu_monitored_monitor_executor,
        )
        config.pretest_kernel_dirty = not clean
        if health_summary.get("critical_total", 0) > 0:
            config.inherited_critical_categories = [
                cat for cat, count in health_summary.get("by_category", {}).items() if count > 0
            ]

        run_dir = config.log_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "pretest_health.json", "w") as fh:
            json.dump(health_summary, fh, indent=2)

        strict_pretest = os.environ.get("GPU_MONITOR_STRICT_PRETEST", "").lower() in ("1", "true", "yes")
        if strict_pretest and not clean:
            pytest.fail(
                f"Pre-test kernel health check failed: {health_summary.get('critical_total', 0)} "
                f"critical event(s) in last {lookback} minutes"
            )

        build_ctx = BuildContext(
            config=config,
            monitor_executor=gpu_monitored_monitor_executor,
        )
        build_status = test.build(build_ctx)
        if build_status == BuildStatus.SOURCE_MISSING:
            pytest.skip(f"{name}: source not available (set ROCM_TEST_ROCMTEST_PATH for NDA workloads)")
        if build_status == BuildStatus.BUILD_FAILED:
            pytest.fail(f"{name}: build failed — prerequisites missing or build error")
        if not test.available(config):
            pytest.skip(f"{name}: workload prerequisites not available on this host")

        orchestrator = MonitoredTestOrchestrator(
            config,
            target_executor=target_executor,
            monitor_executor=gpu_monitored_monitor_executor,
        )
        outcome = orchestrator.run_one(test)

        if outcome.status == "UNSUPPORTED":
            reason = outcome.validation or f"{name}: unsupported on this device"
            if not reason.startswith(f"{name}:"):
                reason = f"{name}: {reason}"
            pytest.skip(reason)
        if outcome.status == "BUILD_FAILED":
            pytest.fail(outcome.validation or f"{name}: build failed at runtime")
        return outcome

    return _run
