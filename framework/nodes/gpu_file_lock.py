# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
gpu_file_lock.py -- Cross-process GPU slot exclusivity via OS file locks.

Uses the ``filelock`` library (same dependency as BinaryBuilder) to create
OS-level advisory locks so that multiple pytest-xdist workers or concurrent
processes never allocate the same GPU slot simultaneously.

Lock file location:
    output/.gpu-locks/<node_label>_gpu<gpu_index>.lock

One lock file exists per (node, GPU) pair.  The file is created on first
``acquire()`` and persists between pytest runs (it is only a lock token —
no content is written to it).

Usage::

    # Context manager (recommended):
    with GpuFileLock(node_label="HOST_IDX_1", gpu_index=2):
        executor.run("python3 workload.py")

    # Manual acquire / release:
    lock = GpuFileLock("localhost", 0)
    lock.acquire(timeout=30.0)
    try:
        ...
    finally:
        lock.release()
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time

logger = logging.getLogger(__name__)

_LOCK_DIR = os.path.join("output", ".gpu-locks")


class GpuFileLock:
    """Cross-process exclusive lock for a single (node, GPU) slot.

    The lock is backed by ``filelock.FileLock`` which wraps ``fcntl.flock()``
    on Linux and ``LockFileEx`` on Windows.

    Cross-process safety:
        ``flock()`` locks are held per (process, file-descriptor) pair.
        Separate xdist worker processes cannot hold the same lock simultaneously.

    Crash safety:
        When a worker process dies (OOM, SIGKILL, unhandled exception), the kernel
        automatically closes all of its file descriptors, which releases all
        ``flock()`` locks that process held.  Stale lock files are harmless — a new
        process can acquire the lock immediately on its next ``acquire()`` call.
        No manual lock-file cleanup is ever required between pytest sessions.

    Double-acquire prevention:
        ``NodePool.acquire_slot()`` releases the in-memory allocator slot if the
        file lock fails, so no GPU is marked "unavailable" without a file lock
        backing it.

    Attributes:
        node_label: Human-readable node identifier (e.g. ``"HOST_IDX_1"``
                    or ``"localhost"``).  Used as the filename prefix.
        gpu_index:  GPU ordinal on the target node (0-based).
        lock_path:  Absolute or relative path to the lock file.
    """

    def __init__(self, node_label: str, gpu_index: int) -> None:
        safe_label = node_label.replace(" ", "_").replace("/", "_").replace(":", "_")
        pathlib.Path(_LOCK_DIR).mkdir(parents=True, exist_ok=True)
        self.node_label = node_label
        self.gpu_index = gpu_index
        self.lock_path = os.path.join(_LOCK_DIR, f"{safe_label}_gpu{gpu_index}.lock")
        self.info_path = os.path.join(_LOCK_DIR, f"{safe_label}_gpu{gpu_index}.info")
        self._lock: object | None = None  # lazy: created on first acquire

    def _get_lock(self):
        """Lazily create the ``filelock.FileLock`` instance.

        Raises:
            RuntimeError: If the ``filelock`` package is not installed.
        """
        if self._lock is None:
            try:
                from filelock import FileLock  # pylint: disable=import-outside-toplevel
            except ImportError as exc:
                raise RuntimeError("GpuFileLock requires the 'filelock' package: pip install filelock") from exc
            self._lock = FileLock(self.lock_path)
        return self._lock

    def acquire(self, timeout: float = 30.0, test_id: str = "") -> None:
        """Acquire the lock, blocking for up to *timeout* seconds.

        After acquiring, writes a ``.info`` sidecar file recording the test name,
        process ID, and acquisition timestamp — readable via ``read_all_holders()``
        to show which test currently holds each GPU slot.

        Args:
            timeout: Maximum seconds to wait for an available lock
                     (default 30 s).  Pass ``0.0`` for non-blocking (fail
                     immediately if the lock is held).  Pass ``-1`` to block
                     indefinitely.
            test_id: Test function name to record in the ``.info`` file.
                     Used for GPU allocation tracking (requirement 1.m).

        Raises:
            RuntimeError: If the lock cannot be acquired within *timeout*.
        """
        try:
            from filelock import Timeout as FileLockTimeout  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError("GpuFileLock requires the 'filelock' package: pip install filelock") from exc

        lock = self._get_lock()
        logger.debug("GpuFileLock: acquiring %s (timeout=%.1fs)", self.lock_path, timeout)
        try:
            lock.acquire(timeout=timeout)
            logger.debug("GpuFileLock: acquired %s", self.lock_path)
        except FileLockTimeout as exc:
            raise RuntimeError(
                f"GpuFileLock: timed out after {timeout}s waiting for "
                f"{self.node_label} GPU-{self.gpu_index} "
                f"(lock file: {self.lock_path})"
            ) from exc

        # Write sidecar metadata so external tools and the framework itself can
        # identify which test holds each GPU lock cross-process.
        if test_id:
            try:
                with open(self.info_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "test_id": test_id,
                            "pid": os.getpid(),
                            "node_label": self.node_label,
                            "gpu_index": self.gpu_index,
                            "acquired_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        },
                        fh,
                    )
            except OSError:
                pass  # metadata is best-effort; never block test execution

    def release(self) -> None:
        """Release the lock and remove the sidecar ``.info`` file.

        Safe to call when the lock is not held.
        """
        if self._lock is not None:
            try:
                is_locked = getattr(self._lock, "is_locked", False)
                if is_locked:
                    self._lock.release()  # type: ignore[attr-defined]
                    logger.debug("GpuFileLock: released %s", self.lock_path)
            except Exception:  # pylint: disable=broad-exception-caught
                pass  # best-effort release
        # Remove metadata file so GPU shows as free in read_all_holders()
        try:
            os.remove(self.info_path)
        except FileNotFoundError:
            pass
        except OSError:
            pass

    @staticmethod
    def read_all_holders(lock_dir: str = _LOCK_DIR) -> list[dict]:
        """Return metadata for all currently held GPU locks.

        Reads every ``.info`` file in *lock_dir* to build a list of
        ``{test_id, pid, node_label, gpu_index, acquired_at}`` records.
        Used by console visibility prints to show which test holds which GPU.

        Returns:
            List of metadata dicts for held GPU slots (may be empty).
        """
        holders = []
        try:
            for fname in os.listdir(lock_dir):
                if not fname.endswith(".info"):
                    continue
                try:
                    with open(os.path.join(lock_dir, fname), encoding="utf-8") as fh:
                        data = json.load(fh)
                    holders.append(data)
                except (OSError, json.JSONDecodeError):
                    pass
        except OSError:
            pass
        return holders

    def __enter__(self) -> GpuFileLock:
        self.acquire()
        return self

    def __exit__(self, *_) -> None:
        self.release()
