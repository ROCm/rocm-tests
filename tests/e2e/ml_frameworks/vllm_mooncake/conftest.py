# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fixtures for tests/e2e/ml_frameworks/vllm_mooncake/.

``vllm_python`` returns the interpreter that has vLLM and the Mooncake transfer engine. Set
``ROCM_TEST_VLLM_PYTHON`` to a ROCm vLLM environment (for example the one in ``vllm/vllm-openai-rocm``);
otherwise the provisioned PyTorch interpreter is used. Missing optional stack pieces skip the test.
"""

from __future__ import annotations

import json
import os
import shlex

import pytest

from tests.e2e.ml_frameworks.vllm_mooncake._constants import VLLM_PYTHON_ENV
from tests.e2e.ml_frameworks.vllm_mooncake._plan import DeploymentPlan, plan_deployment

_PROBE_TIMEOUT = 300.0
_IMPORT_PROBE = "import httpx, huggingface_hub, torch, vllm, mooncake.engine"
_GPU_PROBE = (
    "import json, torch; "
    "props = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]; "
    "print(json.dumps({'count': len(props), 'arch': props[0].gcnArchName.split(':')[0], "
    "'vram_gb': min(p.total_memory for p in props) / 2**30}))"
)


@pytest.fixture
def vllm_python(request, target_executor) -> str:
    """Return a Python executable that can import vLLM and Mooncake, or skip."""
    python = os.environ.get(VLLM_PYTHON_ENV, "").strip()
    if not python:
        request.getfixturevalue("require_torch")
        python = request.getfixturevalue("torch_python")
    probe = target_executor.run(f"{shlex.quote(python)} -c {shlex.quote(_IMPORT_PROBE)}", timeout=_PROBE_TIMEOUT)
    if not probe.ok:
        pytest.skip(f"vLLM/Mooncake not importable by {python}: {probe.stderr[-500:]}")
    return python


@pytest.fixture
def deployment_plan(target_executor, vllm_python: str) -> DeploymentPlan:
    """Probe the acquired GPUs and size the prefill/decode deployment for them, or skip."""
    probe = target_executor.run(f"{shlex.quote(vllm_python)} -c {shlex.quote(_GPU_PROBE)}", timeout=_PROBE_TIMEOUT)
    if not probe.ok:
        pytest.fail(f"GPU probe failed: {probe.stderr[-500:]}")
    info = json.loads(probe.stdout.strip().splitlines()[-1])
    try:
        return plan_deployment(info["arch"], info["count"], info["vram_gb"])
    except ValueError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def require_rdma(target_executor) -> None:
    """Skip when the node has no RDMA device for Mooncake to register DMABUF memory against."""
    if not target_executor.run("ls /sys/class/infiniband | grep -q .", timeout=30).ok:
        pytest.skip("no RDMA device under /sys/class/infiniband")
