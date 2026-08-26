# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration dataclasses for gpu_monitored workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Config:
    """Runtime configuration assembled from CLI args + defaults."""

    # Paths
    script_dir: Path
    build_dir: Path = Path("/tmp/gpu_test_builds")
    rocmtest_path: Path = Path()  # defaults to script_dir/ROCmTest

    # Test tuning
    sample_interval: int = 1
    stress_iters: int = 20
    # Gates starting another required sub-test; a sub-test already in progress
    # still runs to natural completion at the automatic memory target. Kept
    # above a conservative 250-second allowance for each required sub-test so
    # the full six-test contract can start on slower or larger-VRAM hosts. A
    # wedged GPU is caught by ``--per-iter-watchdog``, not by this budget.
    memtest_duration: int = 1800
    # ``memtest_blocks`` caps per-GPU allocation (in MB) passed as
    # ``cuda_memtest --max_num_blocks N``. Empty computes a high-intensity
    # automatic target from 90% of the least-free visible GPU.
    memtest_blocks: str = ""
    # ``hmm_blocks`` caps per-GPU host-side allocation (in MB) passed
    # as ``cuda_memtest_malloc --max_num_blocks N``. Each per-GPU
    # thread does ``new char[max_num_blocks * 1 MB]`` of HOST pageable
    # RAM, so on N-GPU hosts total host RAM consumption is N x this.
    # Empty computes the same high-intensity GPU target while reserving host
    # RAM for the OS/runtime before dividing it among GPU worker threads.
    hmm_blocks: str = ""
    # If True, include Test9 (Bit fade) in the cudamemtest sub-test
    # loop. Off by default because upstream cuda_memtest hardcodes
    # ``sleep(60*90)`` in Test9 (tests.cu: "sleeping for 90 minutes")
    # — runtime is independent of VRAM size and adds 90 min to every
    # cudamemtest invocation. Only useful for genuine bit-decay
    # validation runs; CI / functional pipelines should leave it off.
    include_bit_fade: bool = False
    # Per-sub-test / per-iteration watchdog shared across cudamemtest /
    # hmm_cuda_memtest / transferbench / sln_stress / hipblaslt_bench /
    # RVS tests. 0 = no watchdog (default; workloads run to
    # natural completion). >0 = explicit kill-after-N-seconds for CI /
    # wedged-GPU detection.
    per_iter_watchdog: int = 0

    # Feature flags
    enable_cu_occupancy: bool = False
    # True when exactly one test runs (pytest always sets this).
    single_test_run: bool = False

    # Power band stress. Keep ``power_hold_sec`` in sync with the CLI
    # default (``--power-hold-sec``) so direct ``Config(...)``
    # construction (unit tests, notebooks, downstream tooling) sees
    # the same hold window the README and CLI advertise. The
    # earlier 30 here was a stale leftover.
    power_bands: str = "100,75,50,75,100"
    power_hold_sec: int = 20
    # Number of times the whole band schedule repeats. 10 cycles at the
    # 20s hold above is ~17 min of cap cycling, which exercises the
    # ramp-up/ramp-down transitions repeatedly rather than once.
    power_cycles: int = 10
    pytorch_validator_path: str = ""

    # Inference server stress (vLLM). 0 / "" = "use test class default".
    inference_model: str = ""
    inference_load_sec: int = 0
    inference_warmup_sec: int = 0
    inference_concurrency: int = 0
    # vLLM ``--max-num-seqs`` cap on the running batch. 0 = auto-scale
    # to ``max(8, num_gpus * 4)`` so the harness reliably produces
    # ``num_running > 0 AND num_waiting > 0`` (real prefill/decode
    # overlap) without an operator flag. Without a cap, vLLM's default
    # 256-seq batch is so large that any realistic HTTP-client
    # concurrency drains directly into running with zero waiting, and
    # overlap is never exercised.
    inference_max_num_seqs: int = 0

    # Resolved at runtime (populated by apply_framework_environment)
    rocm_root: Path = field(default=Path())
    rocm_version: str = "unknown"
    gpu_model: str = ""
    gpu_device_id: str = ""
    gpu_short_name: str = ""
    gpu_arch: str = ""
    gpu_conf_dir: str = ""
    num_gpus: int = 0
    clangxx: str = ""
    rocm_lib: Path = field(default=Path())

    # Log root (populated after LOG_ROOT is determined)
    log_root: Path = field(default=Path())

    # Pre-test health probe (Design B)
    # ``pretest_kernel_dirty`` is True iff the pre-test probe found
    # critical dmesg events in its lookback window. Per-test
    # ``summary.json`` reports inherit this so failures can be
    # disambiguated as "ours" vs "we inherited damage". The strict
    # mode (--strict-pretest-gate) aborts the run before this field
    # would even matter; otherwise the run continues with the
    # annotation in place.
    pretest_kernel_dirty: bool = False
    inherited_critical_categories: List[str] = field(default_factory=list)
