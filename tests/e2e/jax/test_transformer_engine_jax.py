# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_transformer_engine_jax.py -- Transformer Engine JAX unit tests on ROCm.

Validates:
    Builds the ROCm Transformer Engine (TE) from source and runs its own
    upstream test suites on an AMD GPU node, one framework test per distinct
    suite:

    1. ``te_jax_ut``     -- provisions a Python 3.12 venv, installs the JAX and
       TE wheels, clones TransformerEngine, and runs ``ci/jax.sh`` at
       TEST_LEVEL=3; asserts that tests ran and none failed.
    2. ``te_jax_cpp_ut`` -- installs CMake and the TE wheels, clones
       TransformerEngine, builds ``tests/cpp`` with ``cmake``/``make``, and runs
       ``make test``; asserts that tests ran and none failed.

Configuration:
    Wheel filenames, artifact base URLs, the git branch, and the GPU family are
    supplied via ``ROCM_TEST_*`` environment variables (see ``conftest.py``).
    Each test skips when a value it requires is unset, matching the framework's
    graceful-degradation guidance for un-provisioned nodes.

Supported hardware:
    gfx94X and gfx950. The test skips on any other architecture.

Markers (no CATEGORY_PROFILE exists for tests/e2e/transformer_engine -- all
declared explicitly):
    hw.gpu, ci.nightly, layer.math_lib, e2e.stack, os.linux, runtime.soak.
"""

import pytest

from framework.reporting.allure_reporter import report_metric
from tests.e2e.transformer_engine._transformer_engine_jax import (
    TeJaxConfig,
    TeTestSummary,
    build_rocm_envs,
    build_wheel_urls,
    clone_transformer_engine,
    create_te_venv,
    detect_gfx_family,
    detect_gpu_arch,
    install_cmake,
    install_ninja,
    install_python_toolchain,
    install_te_wheels_system,
    install_ut_wheels,
    is_supported_arch,
    missing_filenames,
    resolve_te_base,
    run_jax_cpp_ut,
    run_jax_ut,
    teardown_venv,
)


def _require_filenames(config: TeJaxConfig, test_name: str) -> None:
    """Skip the test when a required wheel/archive filename is unset."""
    missing = missing_filenames(config, test_name)
    if missing:
        pytest.skip(f"Missing required arguments for {test_name}: {', '.join(missing)}")


def _require_supported_gpu(executor) -> None:
    """Skip the test when the node's GPU is not gfx94X or gfx950."""
    arch = detect_gpu_arch(executor)
    if not is_supported_arch(arch):
        pytest.skip(f"te_jax tests require gfx94X or gfx950; detected: {arch or 'unknown'}")


def _report(prefix: str, summary: TeTestSummary) -> None:
    """Record per-status result counts as Allure metrics."""
    report_metric(f"{prefix}_passed", float(summary.passed))
    report_metric(f"{prefix}_failed", float(summary.failed))
    report_metric(f"{prefix}_skipped", float(summary.skipped))
    report_metric(f"{prefix}_error", float(summary.error))


def _assert_summary(name: str, summary: TeTestSummary) -> None:
    """Assert that the suite ran and reported no failures or errors."""
    assert summary.ran, f"No tests have run for {name}"
    clean = summary.failed == 0 and summary.error == 0
    assert clean, (
        f"{name} reported {summary.failed} failed / {summary.error} error test(s) "
        f"(passed={summary.passed}, skipped={summary.skipped})"
    )


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.math_lib
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.soak
def test_te_jax_ut(target_executor, te_jax_config: TeJaxConfig, rock_dir: str):
    """Build TransformerEngine and run the JAX unit tests via ci/jax.sh."""
    _require_filenames(te_jax_config, "te_jax_ut")
    _require_supported_gpu(target_executor)

    gfx_family = detect_gfx_family(target_executor, te_jax_config.gfx_family)
    try:
        wheel_urls = build_wheel_urls(te_jax_config, gfx_family)
    except ValueError as exc:
        pytest.skip(f"te_jax_ut configuration incomplete: {exc}")

    try:
        libstdcxx_dir = install_python_toolchain(target_executor)
        create_te_venv(target_executor)
        rocm_envs = build_rocm_envs(rock_dir, libstdcxx_dir)
        install_ut_wheels(target_executor, wheel_urls, rocm_envs)
        te_dir = clone_transformer_engine(target_executor, te_jax_config.te_git_branch)
        summary = run_jax_ut(target_executor, te_dir, rocm_envs)
    finally:
        teardown_venv(target_executor)

    _report("te_jax_ut", summary)
    _assert_summary("te_jax_ut", summary)


@pytest.mark.hw.gpu
@pytest.mark.ci.nightly
@pytest.mark.layer.math_lib
@pytest.mark.e2e.stack
@pytest.mark.os.linux
@pytest.mark.runtime.soak
def test_te_jax_cpp_ut(target_executor, te_jax_config: TeJaxConfig):
    """Build TransformerEngine and run the C++ unit tests via make test."""
    _require_filenames(te_jax_config, "te_jax_cpp_ut")
    _require_supported_gpu(target_executor)

    try:
        te_base = resolve_te_base(te_jax_config)
    except ValueError as exc:
        pytest.skip(f"te_jax_cpp_ut configuration incomplete: {exc}")

    install_cmake(target_executor)
    te_dir = clone_transformer_engine(target_executor, te_jax_config.te_git_branch)
    install_ninja(target_executor)
    install_te_wheels_system(target_executor, te_jax_config, te_base)

    summary = run_jax_cpp_ut(target_executor, te_dir)

    _report("te_jax_cpp_ut", summary)
    _assert_summary("te_jax_cpp_ut", summary)
