# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_jax_ut.py -- JAX (ML framework) unit-test suites on AMD GPUs.
Validates:
    1. ``jax_ut``     -- the full JAX unit-test suite, dispatched to the runner
                         matching the installed JAX / ROCm versions:
                           * JAX >= 0.9.1 -> ci/run_pytest_rocm.sh (new path)
                           * JAX <  0.9.1 -> build/rocm run scripts with the
                             ROCm-7 plugin-folder relocation (<7.0 / ==7.0 />7.0).
                         Adds the multi-GPU sub-suite when JAX_NUM_GPUS > 1.
    2. ``jax_rnn_ut`` -- tests/experimental_rnn_test.py (single GPU).
    3. ``jax_fp8_ut`` -- tests/lax_test.py -k test_mixed_fp8_dot_general (single GPU).

Environment assumptions:
    The JAX source checkout must already exist on the execution node at
    ``ROCM_TEST_JAX_DIR`` (default ``/workspace/jax``) with JAX installed.
    Each test skips gracefully when the checkout (or the JAX package) is absent
    rather than failing the session.
    All run knobs (checkout path, GPU count, version overrides, timeouts) come
    from ``_workload.py`` env vars.

Markers:
    Required dimensions (hw.*, ci.*, layer.*) are also provided by the
    CATEGORY_PROFILE for ``tests/e2e/frameworks/`` (hw.gpu, layer.runtime,
    ci.nightly, e2e.stack, os.linux); they are declared explicitly here for
    clarity.  ``jax_ut`` overrides hw.* (hw.gpu vs hw.multi_gpu) and ci.* (weekly)
    at the function level based on JAX_NUM_GPUS.  runtime.* is always explicit.
"""

from __future__ import annotations

import logging
import shlex

import pytest

from tests.e2e.frameworks.jax._workload import (
    IS_SINGLE_GPU,
    JAX_DIR,
    JAX_FP8_UT_TIMEOUT,
    JAX_RNN_UT_TIMEOUT,
    JAX_UT_TIMEOUT,
    JAX_VERSION_OVERRIDE,
    NUM_GPUS,
    ROCM_VERSION_OVERRIDE,
    SETUP_TIMEOUT,
    JaxCommand,
    jax_ut_commands,
    parse_jax_version,
    parse_rocm_version,
)

logger = logging.getLogger(__name__)

# jax_ut's hardware dimension follows JAX_NUM_GPUS (single- vs multi-GPU mode).
# A conditional MarkDecorator bound to a name so the runtime marker is real while
# the static required-dimension check is satisfied by the frameworks CATEGORY_PROFILE.
_JAX_UT_HW = pytest.mark.hw.gpu if IS_SINGLE_GPU else pytest.mark.hw.multi_gpu


def _run_in_jax_dir(target_executor, command: str, timeout: float):
    """Run *command* from the JAX checkout directory on the execution node.

    ``ExecutionResult.run`` has no ``cwd`` parameter, so the working directory is
    established with a ``cd`` prefix (the executor injects ``ROCR_VISIBLE_DEVICES``
    automatically -- it is never set here).

    Args:
        target_executor: The GPU executor group for the current test.
        command:         Shell command to run relative to ``JAX_DIR``.
        timeout:         Per-command wall-clock cap in seconds.

    Returns:
        The command's :class:`~framework.common.helpers.ExecutionResult`.
    """
    return target_executor.run(f"cd {shlex.quote(JAX_DIR)} && {command}", timeout=timeout)


def _require_jax_checkout(target_executor) -> None:
    """Skip the test when the JAX checkout is absent on the execution node."""
    probe = target_executor.run(f"test -d {shlex.quote(JAX_DIR)}")
    if not probe.ok:
        pytest.skip(f"JAX checkout not present at {JAX_DIR} on this node (set ROCM_TEST_JAX_DIR)")


def _detect_jax_version(target_executor):
    """Return the installed JAX version, skipping when JAX is unavailable.

    Honours the ``JAX_VERSION`` override; otherwise probes ``pip list`` on the
    execution node
    """
    if JAX_VERSION_OVERRIDE:
        return parse_jax_version(f"jax {JAX_VERSION_OVERRIDE}") or pytest.skip(
            f"JAX_VERSION override {JAX_VERSION_OVERRIDE!r} is not a valid version"
        )
    result = target_executor.run(f"cd {shlex.quote(JAX_DIR)} && python3 -m pip list")
    version = parse_jax_version(result.stdout)
    if version is None:
        pytest.skip(f"JAX not installed on this node (pip list reported no 'jax' package):\n{result.stdout[-500:]}")
    return version


def _detect_rocm_version(target_executor):
    """Return the ROCm MAJOR.MINOR version, or ``None`` when undetectable.

    Honours the ``JAX_ROCM_VERSION`` override; otherwise probes ``hipconfig``
    and the ROCm ``.info/version`` file on the execution node.  A ``None`` result
    causes the command builder to assume the newest legacy plugin layout.
    """
    if ROCM_VERSION_OVERRIDE:
        return parse_rocm_version(ROCM_VERSION_OVERRIDE)
    result = target_executor.run("hipconfig --version 2>/dev/null || cat /opt/rocm/.info/version 2>/dev/null || true")
    version = parse_rocm_version(result.stdout)
    logger.info("jax_ut: detected ROCm version=%s from %r", version, result.stdout.strip()[:120])
    return version


@_JAX_UT_HW
@pytest.mark.gpu_count(NUM_GPUS)
@pytest.mark.ci.weekly
@pytest.mark.layer.runtime
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.soak
def test_jax_ut(target_executor):
    """Run the JAX unit-test suite (``jax_ut``) on AMD GPUs.

    Detects the installed JAX and ROCm versions on the execution node and
    dispatches to the matching runner (new ``ci/run_pytest_rocm.sh`` for
    JAX >= 0.9.1, or the legacy ``build/rocm`` run scripts with ROCm-7
    plugin-folder handling).  Runs the multi-GPU sub-suite as well when
    ``JAX_NUM_GPUS > 1`` (which also selects the ``hw.multi_gpu`` marker).

    The runner scripts exit non-zero on any test failure, so ``result.ok``
    is the source of truth for the suite outcome.
    """
    _require_jax_checkout(target_executor)
    jax_version = _detect_jax_version(target_executor)
    rocm_version = _detect_rocm_version(target_executor)
    logger.info(
        "jax_ut: JAX=%s ROCm=%s single_gpu=%s num_gpus=%s",
        jax_version,
        rocm_version,
        IS_SINGLE_GPU,
        NUM_GPUS,
    )

    commands: list[JaxCommand] = jax_ut_commands(jax_version, rocm_version, single_gpu=IS_SINGLE_GPU)
    assert commands, "jax_ut_commands returned no steps -- unexpected version branch"

    for step in commands:
        timeout = JAX_UT_TIMEOUT if step.validate else SETUP_TIMEOUT
        result = _run_in_jax_dir(target_executor, step.command, timeout)
        assert result.ok, (
            f"jax_ut step '{step.label}' failed (exit={result.exit_code}):\n"
            f"cmd: {step.command}\n"
            f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
        )
        if step.validate:
            assert (
                result.stdout.strip()
            ), f"jax_ut step '{step.label}' produced no output -- runner may not have executed any tests"


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.medium
def test_jax_rnn_ut(target_executor):
    """Run the JAX experimental-RNN unit test (``jax_rnn_ut``) on a single GPU."""
    _require_jax_checkout(target_executor)
    result = _run_in_jax_dir(
        target_executor,
        "pytest -v tests/experimental_rnn_test.py",
        JAX_RNN_UT_TIMEOUT,
    )
    assert result.ok, (
        f"jax_rnn_ut failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), "jax_rnn_ut produced no output -- pytest may not have collected any tests"


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.runtime
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.fast
def test_jax_fp8_ut(target_executor):
    """Run the JAX mixed-FP8 dot-general unit test (``jax_fp8_ut``) on a single GPU."""
    _require_jax_checkout(target_executor)
    result = _run_in_jax_dir(
        target_executor,
        "pytest -v tests/lax_test.py -k test_mixed_fp8_dot_general",
        JAX_FP8_UT_TIMEOUT,
    )
    assert result.ok, (
        f"jax_fp8_ut failed (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
    )
    assert result.stdout.strip(), "jax_fp8_ut produced no output -- pytest may not have collected any tests"
