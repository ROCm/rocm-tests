# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Fixtures for the vLLM upstream unit-test port (tests/e2e/frameworks/vllm/).

Provides:
    hf_token        -- HuggingFace access token resolved from the environment;
                       skips the test cleanly when no token is available.
    vllm_container  -- A GPU-passthrough ``ContainerExecutor`` for the vLLM/ROCm
                       image, gated on runtime readiness (docker daemon + AMD
                       devices). Skips instead of failing when the container
                       runtime is unusable or ``--no-gpu`` is active.

No C++ compilation is required for this suite: the vLLM unit tests ship inside
the container image, so there is no ``compile_binary`` fixture here.

Secrets rule (see CLAUDE.md): the HF token is only ever read from an env var,
never hardcoded or committed. Supported env vars (first non-empty wins):
    HF_TOKEN, HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_TOKEN, HF_ACCESS_TOKEN.
"""

from __future__ import annotations

import os

import pytest

# Env vars checked, in priority order, for the HuggingFace access token.
_HF_TOKEN_ENV_VARS = (
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HF_ACCESS_TOKEN",
)


def _resolve_hf_token() -> str | None:
    """Return the first non-empty HuggingFace token from the known env vars."""
    for var in _HF_TOKEN_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    return None


@pytest.fixture
def hf_token() -> str:
    """Return the HuggingFace access token, skipping the test when it is absent.

    Several vLLM unit sub-suites pre-download gated HF models (Mistral, Llama,
    Pixtral, ...), so a valid token is mandatory. When none is configured the
    test is skipped rather than failed — this mirrors the source's early-return
    guard but uses ``pytest.skip`` (never ``sys.exit``).
    """
    token = _resolve_hf_token()
    if not token:
        pytest.skip(
            "HuggingFace access token not configured — set one of "
            f"{', '.join(_HF_TOKEN_ENV_VARS)} to run the vLLM unit suite."
        )
    return token


@pytest.fixture
def vllm_container(request, container_executor):
    """Return a readiness-checked ``ContainerExecutor`` for the vLLM/ROCm image.

    The image is resolved by the ``container_executor`` fixture from either the
    per-test ``@pytest.mark.container_image(...)`` marker or the
    ``--container-image`` CLI option.

    Behaviour:
        * ``--no-gpu`` active            → skip (no hardware session).
        * container runtime not ready    → skip (docker daemon down, AMD devices
                                           absent, or user lacks permissions).

    Skipping (not failing) keeps DryRun / CPU-only collection runs green while
    still exercising fixture wiring.
    """
    if request.config.getoption("--no-gpu", default=False):
        pytest.skip("vLLM unit suite requires GPU hardware — skipped under --no-gpu.")

    status = container_executor.probe()
    if not status.ready:
        pytest.skip("vLLM container runtime not ready: " + "; ".join(status.errors))
    return container_executor
