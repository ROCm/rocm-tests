# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
allocator.py -- Per-node GPU slot allocator with file-based locking.

GpuAllocator manages GPU index assignment across parallel xdist workers.
GpuFileLock provides cross-process safety. Used internally by NodePool.
"""

from __future__ import annotations

import logging
import threading
import time

from framework.gpu.detector import AbstractGpuDetector, GpuInfo

logger = logging.getLogger(__name__)


class GpuAllocator:
    """Thread-safe GPU pool with optional architecture and VRAM filtering.

    Attributes:
        _pool:        All GPUs discovered by the detector at construction time.
        _available:   Set of GPU indices currently available for allocation.
        _condition:   ``threading.Condition`` protecting ``_available``.
                      Also exposed as ``_lock`` for backward compatibility.
        _headroom_gb: VRAM headroom reserved per GPU to prevent OOM.
    """

    def __init__(
        self,
        detector: AbstractGpuDetector,
        headroom_gb: float = 0.0,
    ) -> None:
        """Discover GPUs and initialise the allocation pool.

        Args:
            detector:    GpuDetector or MockGpuDetector instance.
            headroom_gb: VRAM headroom in GB to reserve per GPU when evaluating
                         ``vram_required_gb`` requests.  Tests that annotate
                         ``@pytest.mark.gpu_vram(N)`` will only be assigned to
                         GPUs where ``vram_total_gb - headroom_gb >= N``.
        """
        self._pool: list[GpuInfo] = detector.detect()
        self._available: set[int] = {gpu.index for gpu in self._pool}
        self._condition = threading.Condition()
        self._headroom_gb = headroom_gb
        logger.info("GpuAllocator initialised with %d GPU(s)", len(self._pool))

    @property
    def _lock(self) -> threading.Condition:
        """Backward-compatible alias: ``_condition`` acts as a lock guard."""
        return self._condition

    def allocate(
        self,
        vram_required_gb: float = 0.0,
        wait_timeout_secs: float = 0.0,
    ) -> GpuInfo:
        """Allocate one GPU from the pool, preferring NUMA locality.

        GPU allocation is architecture-agnostic: ``--gpu-arch`` is used for
        compilation (``--offload-arch``) and test filtering, not for slot
        selection.  Every GPU in the pool is eligible regardless of its
        detected arch string.

        When *wait_timeout_secs* > 0 and no eligible GPU is currently available,
        the call polls every second until a GPU is freed by another thread or
        the timeout expires.  This supports dynamic rebalancing in xdist sessions
        where multi-GPU tests hold several slots temporarily.

        Args:
            vram_required_gb:  Minimum VRAM the test needs (in GB).  GPUs where
                               ``vram_total_gb - headroom_gb < vram_required_gb``
                               are excluded from the candidate set.
            wait_timeout_secs: Seconds to wait when no GPU is immediately
                               available (default 0.0 = fail immediately).

        Returns:
            ``GpuInfo`` for the allocated GPU.

        Raises:
            RuntimeError: If no eligible GPU is available within
                          *wait_timeout_secs*.
        """
        deadline = time.monotonic() + wait_timeout_secs if wait_timeout_secs > 0 else None

        with self._condition:
            while True:
                candidates = [
                    gpu
                    for gpu in self._pool
                    if gpu.index in self._available
                    and (vram_required_gb == 0.0 or (gpu.vram_mb / 1024) - self._headroom_gb >= vram_required_gb)
                ]
                if candidates:
                    # Prefer lowest NUMA node for memory locality
                    chosen = min(candidates, key=lambda g: (g.numa_node, g.index))
                    self._available.discard(chosen.index)
                    logger.debug(
                        "Allocated GPU %d (%s, NUMA %d, VRAM %d MB)",
                        chosen.index,
                        chosen.arch,
                        chosen.numa_node,
                        chosen.vram_mb,
                    )
                    return chosen

                # No candidate available — check if we should wait
                if deadline is None or time.monotonic() >= deadline:
                    detail = f" with vram_required={vram_required_gb}GB" if vram_required_gb > 0 else ""
                    avail_vram = {g.index: f"{g.vram_mb}MB" for g in self._pool if g.index in self._available}
                    raise RuntimeError(
                        f"No available GPU{detail}. "
                        f"Pool size: {len(self._pool)}, "
                        f"available: {sorted(self._available)}, "
                        f"VRAM per available GPU: {avail_vram}, "
                        f"headroom: {self._headroom_gb}GB"
                    )

                # Wait for a release() notification instead of a blind sleep.
                # _condition.wait() atomically releases _condition and suspends;
                # it reacquires _condition before returning.
                remaining = max(0.01, deadline - time.monotonic()) if deadline else 1.0
                logger.debug(
                    "GpuAllocator: no GPU available, waiting up to %.1fs for release",
                    min(remaining, 1.0),
                )
                self._condition.wait(timeout=min(remaining, 1.0))

    def release(self, gpu: GpuInfo) -> None:
        """Return *gpu* to the allocation pool and wake any waiting allocators.

        Args:
            gpu: GpuInfo previously returned by allocate().
        """
        with self._condition:
            self._available.add(gpu.index)
            logger.debug("Released GPU %d back to pool", gpu.index)
            # Wake all threads waiting in allocate() so they can re-check candidates.
            # In xdist mode each worker is a separate process so this only helps
            # intra-process scenarios (no-xdist runs); cross-process priority is
            # handled by PendingAcquisitionTracker.
            self._condition.notify_all()
