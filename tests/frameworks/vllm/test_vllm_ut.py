# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_vllm_ut.py -- vLLM upstream unit-test suite on AMD GPUs (ported).


Validates:
    Runs curated slices of the vLLM upstream pytest suite inside a ROCm/vLLM
    container image on an AMD GPU. Each vLLM sub-suite is its own pytest case:

        1. core
        2. test_regression.py
        3. engine test_sequence.py test_config.py test_logger.py
        4. tokenization
        5. test_logits_processor.py   (path auto-resolved inside the container)
        6. tool_use
        7. kernels/quantization/test_awq_triton.py

    A separate ``int4`` case runs only the AWQ-Triton quantization kernel test,
    preserving the source's ``--int4`` fast path as a lighter nightly signal.

Semantics preserved from the source:
    * VLLM_WORKER_MULTIPROC_METHOD=spawn is exported before pytest runs.
    * modelscope/tblib pip packages and curl/libsodium23 OS packages are
      installed inside the container before the suite runs.
    * The vLLM tests tree is located at /app/vllm/tests or /vllm-workspace/tests.
    * pytest summary counts (passed/failed/skipped/errors) are parsed and
      reported as Allure metrics.

Marker rationale (tests/e2e/frameworks/vllm has no CATEGORY_PROFILE, so every
dimension is declared explicitly):
    * hw.gpu       -- vLLM unit suite exercises HIP kernels on a single GPU.
    * layer.runtime-- vLLM sits on the ROCm/HIP runtime layer; the taxonomy
                      exposes only {runtime, math_lib} and runtime is the closest
                      fit for an ML-framework suite.
    * ci.weekly    -- the full suite downloads several large gated HF models and
                      runs a broad pytest tree (hours) -> weekly gate.
    * runtime.soak -- multi-hour wall time for the full suite.
    * ci.nightly / runtime.medium for the focused int4 AWQ case.
    * e2e.stack, os.linux -- ROCm software-stack test, Linux-only container.

Secrets: the HuggingFace token is sourced from the environment via the
``hf_token`` fixture (see conftest.py) and injected as HF_TOKEN inside the
container -- never hardcoded or committed. Missing token -> the test skips.

Container image: resolved from ``@pytest.mark.container_image(...)`` (default
below, overridable via the VLLM_UT_DOCKER_IMAGE env var) or the
``--container-image`` CLI option.
"""

from __future__ import annotations

import os
import re
import shlex

import pytest

from framework.reporting.allure_reporter import report_metric

# ---------------------------------------------------------------------------
# Configuration (env-overridable; no secrets)
# ---------------------------------------------------------------------------

# Default vLLM/ROCm image. Override with VLLM_UT_DOCKER_IMAGE or --container-image.
# NOTE: verify the exact published image tag for your ROCm version before use.
_VLLM_IMAGE = os.environ.get("VLLM_UT_DOCKER_IMAGE", "rocm/vllm:latest")

# Per-sub-suite container timeout (seconds). vLLM suites are long-running; the
# full suite is a soak workload, so this defaults to 2h and is env-overridable.
_SUITE_TIMEOUT_SECS = float(os.environ.get("VLLM_UT_TIMEOUT_SECS", "7200"))

# The AWQ-Triton quantization kernel test path (the source's int4 fast path).
_AWQ_TRITON_TEST = "kernels/quantization/test_awq_triton.py"

# Default (non-int4) suite: one entry per pytest case. A single entry may name
# several test files (space-separated) exactly as the source did.
_VLLM_TESTCASES: list[str] = [
    "core",
    "test_regression.py",
    "engine test_sequence.py test_config.py test_logger.py",
    "tokenization",
    "test_logits_processor.py",
    "tool_use",
    _AWQ_TRITON_TEST,
]


def _case_id(testcase: str) -> str:
    """Return a filesystem/nodeid-safe parametrize id for a testcase string."""
    return re.sub(r"[^A-Za-z0-9]+", "_", testcase).strip("_") or "case"


def _resolve_pytest_target(testcase: str) -> str:
    """Return the pytest target expression to run inside the container.

    ``test_logits_processor.py`` lives under ``model_executor/`` on some vLLM
    layouts and at the tests root on others. The source picked the path from the
    ROCm major version; here we resolve it robustly with a shell existence check
    inside the container, which is independent of the ROCm version.
    """
    if testcase == "test_logits_processor.py":
        return (
            "$(test -f model_executor/test_logits_processor.py "
            "&& echo model_executor/test_logits_processor.py "
            "|| echo test_logits_processor.py)"
        )
    return testcase


def _build_suite_command(pytest_target: str, hf_token: str) -> str:
    """Assemble the in-container shell command that runs one vLLM sub-suite.

    Mirrors the source's ``run_unit_tests`` setup: install modelscope/tblib and
    curl/libsodium23, export the multiproc method and HF token, locate the vLLM
    tests tree, then run ``pytest -v -s <target>``. The final exit code is
    pytest's, so ``result.ok`` reflects suite pass/fail.
    """
    tok = shlex.quote(hf_token)
    return "\n".join(
        [
            f"export HF_TOKEN={tok}",
            f"export HUGGING_FACE_HUB_TOKEN={tok}",
            "export VLLM_WORKER_MULTIPROC_METHOD=spawn",
            # Best-effort deps: do not fail the suite if the mirror is flaky.
            "python -m pip install --no-cache-dir modelscope tblib || true",
            "(apt-get update && apt-get install -y --no-install-recommends "
            "curl libsodium23) || true",
            # Locate the vLLM tests tree (same paths the source probed).
            "if [ -d /app/vllm/tests ]; then cd /app/vllm/tests; "
            "elif [ -d /vllm-workspace/tests ]; then cd /vllm-workspace/tests; "
            "else echo VLLM_TESTS_NOT_FOUND; exit 3; fi",
            f"python -m pytest -v -s {pytest_target}",
        ]
    )


# Regexes matching the pytest summary line, mirroring the source's parsers.
_COUNT_PATTERNS = {
    "passed": re.compile(r"(\d+)\s+passed"),
    "failed": re.compile(r"(\d+)\s+failed"),
    "skipped": re.compile(r"(\d+)\s+skipped"),
    "errors": re.compile(r"(\d+)\s+error"),
}


def _parse_and_report_counts(output: str, label: str) -> dict[str, int]:
    """Parse pytest summary counts from *output* and report them as metrics."""
    counts: dict[str, int] = {}
    for name, pattern in _COUNT_PATTERNS.items():
        matches = pattern.findall(output)
        value = int(matches[-1]) if matches else 0
        counts[name] = value
        report_metric(f"vllm_{label}_{name}", float(value))
    return counts


def _run_vllm_suite(vllm_container, hf_token: str, testcase: str, label: str) -> None:
    """Run one vLLM sub-suite in the container and assert it passed."""
    target = _resolve_pytest_target(testcase)
    command = _build_suite_command(target, hf_token)
    result = vllm_container.run(command, timeout=_SUITE_TIMEOUT_SECS)

    if "VLLM_TESTS_NOT_FOUND" in result.stdout or "VLLM_TESTS_NOT_FOUND" in result.stderr:
        pytest.skip(
            "vLLM tests tree not found in the container at /app/vllm/tests or "
            "/vllm-workspace/tests — verify the container image."
        )

    combined = f"{result.stdout}\n{result.stderr}"
    # pytest exit code 5 == no tests collected: treat as skip, not failure.
    if result.exit_code == 5 and "no tests ran" in combined.lower():
        pytest.skip(f"vLLM suite '{testcase}' collected no tests:\n{result.stdout[:1000]}")

    counts = _parse_and_report_counts(combined, label)

    assert result.ok, (
        f"vLLM suite '{testcase}' failed (exit={result.exit_code}; "
        f"passed={counts['passed']}, failed={counts['failed']}, "
        f"errors={counts['errors']}, skipped={counts['skipped']}):\n"
        f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-1000:]}"
    )


@pytest.mark.hw.gpu
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.runtime.soak
@pytest.mark.os.linux
@pytest.mark.container_image(_VLLM_IMAGE)
@pytest.mark.parametrize("testcase", _VLLM_TESTCASES, ids=[_case_id(t) for t in _VLLM_TESTCASES])
def test_vllm_ut(vllm_container, hf_token: str, testcase: str):
    """Run one slice of the vLLM upstream unit-test suite on an AMD GPU."""
    _run_vllm_suite(vllm_container, hf_token, testcase, label=_case_id(testcase))


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.runtime.medium
@pytest.mark.os.linux
@pytest.mark.container_image(_VLLM_IMAGE)
def test_vllm_ut_int4_awq_triton(vllm_container, hf_token: str):
    """Run only the AWQ-Triton quantization kernel test (source's ``--int4`` mode)."""
    _run_vllm_suite(vllm_container, hf_token, _AWQ_TRITON_TEST, label="int4_awq")
