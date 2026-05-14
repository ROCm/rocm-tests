# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
gpu_plugin.py -- GPU acquisition fixture and related pytest options.

Responsibilities:
    - Register ``--no-gpu``, ``--gpu-arch``, ``--mock-gpu``, ``--rocm-config``
      pytest CLI options (via pytest_addoption).
    - Gate tests marked ``@pytest.mark.hw.gpu``: skip when ``--no-gpu`` is active.
    - Provide the ``gpu_fixture`` function-scoped fixture.
    - Provide the ``dry_run_executor`` fixture (DryRunExecutor, no GPU needed).

Loaded automatically via pytest_plugins in conftest.py.

pytest options added:
    --no-gpu        Force DryRunExecutor for all tests; skip hw.gpu-marked tests.
    --gpu-arch ARCH Filter GPU allocation to a specific GFX architecture.
    --mock-gpu      Use MockGpuDetector instead of real hardware detection.
    --rocm-config   Path to an alternate rocm-test.toml config file.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import pytest

from framework.gpu.allocator import GpuAllocator
from framework.gpu.detector import GpuDetector, GpuInfo, MockGpuDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GpuFixture — GPU descriptor exposed to health-check fixtures
# ---------------------------------------------------------------------------


@dataclass
class GpuFixture:
    """Descriptor for an allocated AMD GPU.

    Holds the metadata of the GPU acquired by ``gpu_fixture`` — architecture,
    VRAM, and NUMA node.

    **This class has no** ``.run()`` **method.**
    Use the ``target_executor`` fixture (from ``remote_node_plugin``) to run
    commands on the GPU.  ``GpuFixture`` is a metadata-only descriptor used for
    health checks and arch-specific baseline lookups; it does not execute
    subprocesses.

    Attributes:
        gpu_info: Detected GPU descriptor (arch, VRAM, NUMA node).
    """

    gpu_info: GpuInfo


# ---------------------------------------------------------------------------
# pytest hooks
# ---------------------------------------------------------------------------


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
        help="Restrict GPU allocation to a specific architecture (e.g. gfx1100).",
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
    """Store the GPU detector on the config object for shared fixture access.

    Passes ``rock_dir`` and ``artifact_dir`` (from ``rocm-test.toml``) to
    ``GpuDetector`` so GPU info diagnostic logs are written under the configured
    artifact directory.
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gpu_fixture(request, health_fixture):
    """Allocate one AMD GPU for this test, yield a GpuFixture, release on teardown.

    Automatically:
    - Reads ``@pytest.mark.gpu_vram(N)`` from the test to require at least N GB
      of free VRAM on the allocated GPU.
    - Acquires a GPU from the session pool (NUMA-aware, arch + VRAM filtered).
    - Runs a pre-test GPU health check (temp, ECC, VRAM); skips the test if the
      GPU is already in a degraded state.
    - GPU isolation is set via ROCR_VISIBLE_DEVICES (through ``target_executor`` / ``LocalExecutor``).
    - Releases the GPU back to the pool after the test completes.
    - Runs a post-test GPU health check and logs a warning if health degraded.

    When ``--no-gpu`` is active, tests marked ``hw.gpu`` are skipped before
    this fixture runs, so DryRunExecutor is never needed here.

    Markers read:
        ``@pytest.mark.gpu_vram(N)`` — require at least *N* GB of VRAM.
                                       Skips the test if no GPU meets the
                                       requirement.

    Args:
        health_fixture: GpuHealthChecker configured from framework_config thresholds.

    Yields:
        GpuFixture: GPU descriptor (arch, VRAM, NUMA node).  For command
        execution, declare ``target_executor`` instead — it acquires a GPU
        slot from ``NodePool`` and returns a ready ``LabeledExecutor``.
    """
    config = request.config
    headroom_gb = config.getoption("--vram-headroom-gb", default=2.0)
    detector = getattr(config, "_gpu_detector", None) or GpuDetector()
    allocator = GpuAllocator(detector=detector, headroom_gb=headroom_gb)
    gpu_arch = config.getoption("--gpu-arch", default=None)

    # Read optional VRAM requirement from @pytest.mark.gpu_vram(N)
    vram_marker = request.node.get_closest_marker("gpu_vram")
    vram_required_gb: float = float(vram_marker.args[0]) if vram_marker else 0.0

    try:
        gpu_info = allocator.allocate(arch=gpu_arch, vram_required_gb=vram_required_gb)
    except RuntimeError as exc:
        pytest.skip(f"gpu_fixture: {exc}")

    # Pre-test health gate — skip rather than run on a degraded GPU
    pre_health = health_fixture.check(gpu_info.index)
    logger.info(health_fixture.summary_line(gpu_info, pre_health, "pre"))
    logger.debug("\n%s", health_fixture.detail_block(gpu_info, pre_health, "pre"))
    if not pre_health.passed:
        allocator.release(gpu_info)
        pytest.skip(f"gpu_fixture: GPU {gpu_info.index} failed pre-test health check: " f"{pre_health.message}")

    fixture = GpuFixture(gpu_info=gpu_info)

    try:
        yield fixture
    finally:
        # Post-test health gate — warn if the GPU degraded during the test
        post_health = health_fixture.check(gpu_info.index)
        logger.info(health_fixture.summary_line(gpu_info, post_health, "post"))
        logger.debug("\n%s", health_fixture.detail_block(gpu_info, post_health, "post"))
        if not post_health.passed:
            logger.warning(
                "GPU %d post-test health degraded: %s",
                gpu_info.index,
                post_health.message,
            )
        allocator.release(gpu_info)


@pytest.fixture
def dry_run_executor():
    """Provide a DryRunExecutor for tests that need to exercise framework logic
    without real GPU hardware.

    Returns:
        DryRunExecutor: Returns synthetic ExecutionResult(exit_code=0) for any command.
    """
    from framework.executors.dry_run_executor import DryRunExecutor  # pylint: disable=import-outside-toplevel

    return DryRunExecutor()
