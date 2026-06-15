# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
gpu_plugin.py -- GPU detection, mock-GPU mode, and GPU-arch filtering.

Adds: --no-gpu, --gpu-arch, --mock-gpu, --rocm-config.
Provides: gpu_arch (session-scoped string | None), dry_run_executor.
"""

from __future__ import annotations

import logging

import pytest

from framework.gpu.detector import GpuDetector, MockGpuDetector

logger = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register rocm-test GPU CLI options."""
    group = parser.getgroup("rocm-gpu", "ROCm GPU options")
    group.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip tests requiring GPU hardware; force DryRun mode.",
    )
    group.addoption(
        "--gpu-arch",
        action="store",
        default=None,
        metavar="ARCH",
        help=(
            "Target GFX architecture (e.g. gfx90a). Used for compilation (--offload-arch),"
            " CMake -DGPU_ARCH, and arch-specific library path resolution."
            " Does not filter GPU slot allocation."
        ),
    )
    group.addoption(
        "--mock-gpu",
        action="store_true",
        default=False,
        help="Use MockGpuDetector (synthetic GPUs) instead of real hardware.",
    )
    group.addoption(
        "--rocm-config",
        action="store",
        default=None,
        metavar="PATH",
        help="Path to rocm-test.toml config file (default: CWD/rocm-test.toml).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Store the GPU detector on the config object for use by NodePool.

    ``NodePool`` (remote_node_plugin) reads ``config._gpu_detector`` to run
    GPU detection on each fleet node at session start.
    """
    if config.getoption("--mock-gpu", default=False):
        config._gpu_detector = MockGpuDetector()  # type: ignore[attr-defined]
        logger.info("GPU mode: MockGpuDetector (--mock-gpu)")
    else:
        import os  # pylint: disable=import-outside-toplevel

        from framework.config.loader import load_config  # pylint: disable=import-outside-toplevel

        rock_dir: str | None = (
            config.getoption("--rock-dir", default=None)
            or os.environ.get("ROCK_DIR")
            or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
        )
        cfg = load_config(config_path=config.getoption("--rocm-config", default=None))
        config._gpu_detector = GpuDetector(rock_dir=rock_dir, artifact_dir=cfg.framework.artifact_dir)  # type: ignore[attr-defined]
        logger.info(
            "GPU mode: real hardware detection (rock_dir=%s)",
            rock_dir or "not set",
        )


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip hw.gpu-marked tests when --no-gpu is active."""
    has_gpu_marker = any(m.name in ("hw.gpu", "hw.multi_gpu") for m in item.iter_markers())
    if has_gpu_marker and item.config.getoption("--no-gpu", default=False):
        pytest.skip("Skipped: --no-gpu flag active")


@pytest.fixture(scope="session")
def gpu_arch(request: pytest.FixtureRequest) -> str | None:
    """Return the target GPU architecture string from ``--gpu-arch``, or ``None``.

    Session-scoped so the option is read once and shared across all tests.
    Consumers — CMake build fixtures, runtime library path helpers — should
    depend on this fixture instead of calling ``request.config.getoption``
    directly.

    Returns:
        Architecture string (e.g. ``"gfx90a"``) or ``None`` when not supplied.
    """
    value = request.config.getoption("--gpu-arch", default=None)
    return str(value) if value is not None else None


@pytest.fixture
def dry_run_executor():
    """Provide a DryRunExecutor for tests that need to exercise framework logic
    without real GPU hardware.

    Returns:
        DryRunExecutor: Returns synthetic ExecutionResult(exit_code=0) for any command.
    """
    from framework.executors.dry_run_executor import DryRunExecutor  # pylint: disable=import-outside-toplevel

    return DryRunExecutor()
