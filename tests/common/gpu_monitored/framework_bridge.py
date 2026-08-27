# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Bridge rocm-tests pytest fixtures to gpu_monitored Config."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from framework.gpu.detector import GpuDetector
from framework.rocm.libs.amd_smi import list_devices
from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.environment import apply_framework_environment
from tests.common.gpu_pci_map import short_name_for_device

if TYPE_CHECKING:
    from framework.executors.cpu_executor import CpuExecutor
    from framework.nodes.node_pool import NodePool

logger = logging.getLogger(__name__)


def rocm_tests_root() -> Path:
    """Return the rocm-tests repository root (directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "tests" / "e2e").is_dir():
            return parent
    return here.parents[3]


def resolve_pytorch_validator(config: Config) -> Path | None:
    """Locate ``pytorch_training_validator/rocm_diag/run_suite.py`` if present."""
    if config.pytorch_validator_path:
        path = Path(config.pytorch_validator_path)
        if path.is_file():
            return path.resolve()
    env_path = os.environ.get("PYTORCH_VALIDATOR_PATH", "").strip()
    if env_path:
        path = Path(env_path)
        if path.is_file():
            return path.resolve()
    root = rocm_tests_root()
    candidates = (
        root.parent / "test_contents" / "pytorch_training_validator" / "rocm_diag" / "run_suite.py",
        root.parent / "pytorch_training_validator" / "rocm_diag" / "run_suite.py",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def resolve_host_gpu_count(
    node_pool: NodePool | None,
    cpu_executor: CpuExecutor,
    rock_dir: str | None = None,
) -> int:
    """GPU count from framework ``NodePool`` or ``amd-smi`` / ``GpuDetector``."""
    if node_pool is not None:
        count = node_pool.total_gpu_slots()
        if count > 0:
            return count

    devices = list_devices(cpu_executor)
    if devices:
        return len(devices)

    detected = GpuDetector(rock_dir=rock_dir).detect()
    return len(detected)


def resolve_gpu_identity(
    cpu_executor: CpuExecutor,
    gpu_arch: str | None,
    rock_dir: str | None = None,
) -> tuple[str, str, str]:
    """Return ``(arch, model_name, primary_bdf)`` from framework detection."""
    devices = list_devices(cpu_executor)
    arch = gpu_arch or ""
    model = ""
    bdf = ""

    if devices:
        arch = arch or devices[0].arch or "unknown"
        bdf = devices[0].bdf or ""
        if devices[0].asic_serial and devices[0].asic_serial != "unknown":
            model = devices[0].asic_serial

    if not arch or arch == "unknown":
        for gpu in GpuDetector(rock_dir=rock_dir).detect():
            if gpu.arch and gpu.arch != "unknown":
                arch = gpu.arch
                break

    if not model:
        amd_smi = "amd-smi"
        if rock_dir:
            candidate = Path(rock_dir) / "bin" / "amd-smi"
            if candidate.is_file():
                amd_smi = str(candidate)
        result = cpu_executor.run(f"{amd_smi} static -a -g 0")
        if result.ok:
            for line in result.stdout.splitlines():
                if "MARKET_NAME" in line and ":" in line:
                    model = line.split(":", 1)[1].strip()
                    break

    return arch or "unknown", model, bdf


def make_monitored_config(
    *,
    rock_dir: str,
    ld_path: dict[str, str],
    compiler_build_dir: str,
    artifact_dir: str,
    sample_interval: int | None,
    rocmtest_path: str | None,
    num_gpus: int,
    gpu_arch: str,
    gpu_model: str,
    gpu_device_id: str,
) -> Config:
    """Build a :class:`Config` from framework fixture values (no re-probing)."""
    build_dir = Path(compiler_build_dir) / "gpu_monitored"
    build_dir.mkdir(parents=True, exist_ok=True)

    log_root = Path(artifact_dir) / "gpu_monitored"
    log_root.mkdir(parents=True, exist_ok=True)

    interval = sample_interval
    if interval is None:
        interval = int(os.environ.get("GPU_MONITOR_INTERVAL", "1"))

    rocmtest = Path(rocmtest_path) if rocmtest_path else Path(os.environ.get("ROCM_TEST_ROCMTEST_PATH", ""))

    cfg = Config(
        script_dir=rocm_tests_root(),
        build_dir=build_dir,
        rocmtest_path=rocmtest,
        sample_interval=interval,
        enable_cu_occupancy=os.environ.get("GPU_MONITOR_CU_OCCUPANCY", "").lower() in ("1", "true", "yes"),
        per_iter_watchdog=int(os.environ.get("GPU_MONITOR_PER_ITER_WATCHDOG", "0") or "0"),
        memtest_duration=int(os.environ.get("CUDAMEMTEST_DURATION", "1800")),
        memtest_blocks=os.environ.get("CUDAMEMTEST_MAX_BLOCKS", ""),
        hmm_blocks=os.environ.get("HMM_MEMTEST_MAX_BLOCKS", ""),
        include_bit_fade=os.environ.get("CUDAMEMTEST_INCLUDE_BIT_FADE", "").lower() in ("1", "true", "yes"),
        stress_iters=int(os.environ.get("SLN_STRESS_ITERS", "20")),
        power_bands=os.environ.get("POWER_BANDS", "100,75,50,75,100"),
        power_hold_sec=int(os.environ.get("POWER_HOLD_SEC", "20")),
        power_cycles=int(os.environ.get("POWER_CYCLES", "10")),
        pytorch_validator_path=os.environ.get("PYTORCH_VALIDATOR_PATH", ""),
        inference_model=os.environ.get("INFERENCE_MODEL", ""),
        inference_load_sec=int(os.environ.get("INFERENCE_LOAD_SEC", "0") or "0"),
        inference_warmup_sec=int(os.environ.get("INFERENCE_WARMUP_SEC", "0") or "0"),
        inference_concurrency=int(os.environ.get("INFERENCE_CONCURRENCY", "0") or "0"),
        inference_max_num_seqs=int(os.environ.get("INFERENCE_MAX_NUM_SEQS", "0") or "0"),
        log_root=log_root,
        single_test_run=True,
        num_gpus=num_gpus,
        gpu_arch=gpu_arch,
        gpu_model=gpu_model,
        gpu_device_id=gpu_device_id,
        gpu_short_name=short_name_for_device(gpu_device_id),
    )
    apply_framework_environment(cfg, rock_dir=rock_dir, ld_path=ld_path)
    return cfg


def ensure_gpu_environment(config: Config) -> None:
    """Fail fast when whole-node prerequisites are missing."""
    if config.num_gpus < 1:
        pytest.fail("Framework reports zero workload-visible GPUs. " "Check NodePool allocation and device access.")
