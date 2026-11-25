# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run-time constants for the vLLM + Mooncake DMABUF KV-cache test area."""

from __future__ import annotations

import math
import os

# Ungated mirror of Llama-3.1-70B-Instruct (128K context). For another model set ROCM_TEST_VLLM_MODEL and
# ROCM_TEST_VLLM_WEIGHTS_GB (bf16/fp8 checkpoint size); the deployment is re-derived from it.
MODEL = os.environ.get("ROCM_TEST_VLLM_MODEL", "NousResearch/Meta-Llama-3.1-70B-Instruct")
MODEL_WEIGHTS_GB = float(os.environ.get("ROCM_TEST_VLLM_WEIGHTS_GB", "141"))
TENSOR_PARALLEL_OVERRIDE = int(os.environ.get("ROCM_TEST_VLLM_TP", "0"))
VLLM_PYTHON_ENV = "ROCM_TEST_VLLM_PYTHON"

# Every GPU on the node is reserved; prefill and decode instances split them evenly.
GPU_COUNT_ARG = "all"

# Weights may take at most this share of one GPU, leaving the rest for KV cache and side tensors.
WEIGHT_BUDGET_FRAC = 0.5
# Largest tensor-parallel size the test sizes for (an 8-GPU node split into prefill and decode halves).
MAX_TENSOR_PARALLEL = 4
MIN_GPU_VRAM_GB = math.ceil(MODEL_WEIGHTS_GB / (MAX_TENSOR_PARALLEL * WEIGHT_BUDGET_FRAC))

# vLLM leaves SIDE_TENSOR_FRAC of each GPU free so side tensors can lift it to PRESSURE_TARGET.
PRESSURE_TARGET = 0.90
SIDE_TENSOR_FRAC = 0.15
GPU_MEMORY_UTILIZATION = round(PRESSURE_TARGET - SIDE_TENSOR_FRAC, 2)
PRESSURE_TOLERANCE = 0.02
FAKE_OOM_FREE_FLOOR_BYTES = 1 << 30

# Pass criterion: |HIP used - amdgpu used| and post-run leak, as a fraction of total VRAM.
MAX_DRIFT_FRAC = 0.01

OUTPUT_TOKENS = 64

# (total sequence length, concurrent requests) per phase.
PHASES = {
    "smoke": ((8192, 16),),
    "full": ((8192, 128), (32768, 64), (131072, 16)),
}

STARTUP_TIMEOUT = 3600.0
REQUEST_TIMEOUT = 7200.0
RUN_TIMEOUT = {"smoke": 7200.0, "full": 21600.0}

SUMMARY_START = "__DMABUF_KV_SUMMARY_START__"
SUMMARY_END = "__DMABUF_KV_SUMMARY_END__"
