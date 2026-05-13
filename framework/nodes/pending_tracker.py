# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pending_tracker.py -- Cross-process pending GPU acquisition request registry.

When ``acquire_slots(count=N)`` is waiting for N GPU slots, it registers its
intent here.  ``acquire_slot(count=1)`` checks this registry before grabbing a
GPU: if doing so would leave fewer available slots than any pending multi-GPU
request needs, the single-GPU caller yields (sleeps briefly) without consuming
the slot.  This gives high-demand waiters priority when slots free up.

Storage layout (per node label, in ``output/.gpu-locks/``)::

    <node_label>_pending.json   -- active request map (JSON)
    <node_label>_pending.lock   -- filelock protecting the JSON

JSON schema::

    {
        "<uuid>": {"count": 2, "timeout_at": 1715472525.6},
        ...
    }

Crash / stale-entry safety:
    Each entry carries ``timeout_at`` (absolute monotonic time).  Entries
    whose ``timeout_at`` has passed are pruned silently on every read.  If a
    worker crashes with an entry registered, it expires automatically after
    the acquisition timeout elapses — no manual cleanup needed.

    Session-start cleanup:
    ``cleanup_session_start()`` is called once by the master ``pytest_configure``
    hook before any xdist workers are spawned.  It removes all ``*_pending.json``
    and ``*_pending.lock`` files so that stale registrations from a crashed session
    cannot block the next session.  The ``time.monotonic()`` clock resets after a
    system reboot, which would otherwise make old ``timeout_at`` values appear
    perpetually valid and cause ``should_yield()`` to block single-GPU tests
    indefinitely.

Error handling:
    All file-I/O errors are swallowed: ``max_pending_count()`` returns 0 and
    ``should_yield()`` returns ``False`` on any exception, degrading gracefully
    to the previous no-priority behaviour.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import time
import uuid

logger = logging.getLogger(__name__)

_LOCK_DIR = os.path.join("output", ".gpu-locks")
_META_LOCK_TIMEOUT = 0.5  # seconds: cap filelock wait to avoid blocking callers


class PendingAcquisitionTracker:
    """Cross-process registry of pending multi-GPU slot acquisition requests.

    Uses the same ``filelock`` library and ``output/.gpu-locks/`` directory as
    ``GpuFileLock`` so no extra infrastructure is required.

    Typical usage in ``NodePool.acquire_slots``::

        tracker = PendingAcquisitionTracker("localhost")
        req_id = tracker.register(count=2, timeout_at=deadline)
        try:
            while True:
                # … try to acquire 2 slots …
        finally:
            tracker.deregister(req_id)

    And in ``NodePool.acquire_slot``::

        tracker = PendingAcquisitionTracker("localhost")
        if tracker.should_yield(available_after_grab=available - 1):
            time.sleep(0.5)
            continue   # skip this acquisition round

    Attributes:
        node_label: Node identifier string (same as used in ``GpuFileLock``).
        _json_path: Path to the pending-request JSON file.
        _lock_path: Path to the meta-filelock protecting the JSON file.
    """

    def __init__(self, node_label: str) -> None:
        safe = node_label.replace(" ", "_").replace("/", "_").replace(":", "_")
        pathlib.Path(_LOCK_DIR).mkdir(parents=True, exist_ok=True)
        self.node_label = node_label
        self._json_path = os.path.join(_LOCK_DIR, f"{safe}_pending.json")
        self._lock_path = os.path.join(_LOCK_DIR, f"{safe}_pending.lock")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, count: int, timeout_at: float) -> str:
        """Register a pending request for *count* GPU slots.

        Args:
            count:      Number of GPU slots needed simultaneously.
            timeout_at: Absolute ``time.monotonic()`` deadline for this request.
                        Entries whose deadline has passed are treated as expired.

        Returns:
            A UUID string that identifies this request.  Pass it to
            :meth:`deregister` when the request succeeds or times out.
        """
        request_id = str(uuid.uuid4())
        try:
            with self._meta_lock():
                data = self._read_unsafe()
                data[request_id] = {"count": count, "timeout_at": timeout_at}
                self._write_unsafe(data)
            logger.debug(
                "PendingTracker [%s]: registered req %s (count=%d, ttl=%.1fs)",
                self.node_label,
                request_id[:8],
                count,
                max(0.0, timeout_at - time.monotonic()),
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("PendingTracker [%s]: register failed (ignored): %s", self.node_label, exc)
        return request_id

    def deregister(self, request_id: str) -> None:
        """Remove a pending request entry.

        Safe to call if the entry was never successfully registered (e.g. the
        :meth:`register` call itself raised).

        Args:
            request_id: UUID returned by :meth:`register`.
        """
        try:
            with self._meta_lock():
                data = self._read_unsafe()
                removed = data.pop(request_id, None)
                if removed is not None:
                    self._write_unsafe(data)
            logger.debug(
                "PendingTracker [%s]: deregistered req %s",
                self.node_label,
                request_id[:8],
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug("PendingTracker [%s]: deregister failed (ignored): %s", self.node_label, exc)

    def max_pending_count(self) -> int:
        """Return the highest GPU count any live (non-expired) waiter needs.

        Expired entries (``timeout_at < time.monotonic()``) are pruned
        opportunistically each time this method is called.

        Returns:
            Maximum pending count (>= 2 when any multi-GPU waiter is active),
            or 0 if no live requests exist.  Returns 0 on any I/O error.
        """
        try:
            now = time.monotonic()
            with self._meta_lock():
                data = self._read_unsafe()
                live = {k: v for k, v in data.items() if v.get("timeout_at", 0) > now}
                if len(live) != len(data):
                    self._write_unsafe(live)  # prune expired
            if not live:
                return 0
            return max(int(v["count"]) for v in live.values())
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.debug(
                "PendingTracker [%s]: max_pending_count error (returning 0): %s",
                self.node_label,
                exc,
            )
            return 0

    def true_available_count(self, gpu_indices: list[int]) -> int:
        """Probe OS file locks to get the TRUE cross-process available GPU count.

        Unlike reading the in-memory ``GpuAllocator._available`` set (which is
        per-worker and stale in xdist mode), this method probes each GPU's
        ``GpuFileLock`` file with a non-blocking ``acquire(timeout=0)`` +
        immediate ``release()``.  A successful probe means no other process
        holds that lock — the GPU is truly free.

        This is the correct source of truth under xdist because ``flock()``
        locks are OS-level and visible across all worker processes.

        Args:
            gpu_indices: List of GPU ordinals (``GpuInfo.index``) on this node.

        Returns:
            Number of GPUs whose file lock is currently unheld (truly available).
            Falls back to ``len(gpu_indices)`` on ``filelock`` import error.
        """
        try:
            from filelock import FileLock, Timeout as FileLockTimeout  # pylint: disable=import-outside-toplevel
        except ImportError:
            # filelock not installed — degrade gracefully (no priority)
            return len(gpu_indices)

        safe = self.node_label.replace(" ", "_").replace("/", "_").replace(":", "_")
        available = 0
        for idx in gpu_indices:
            lock_path = os.path.join(_LOCK_DIR, f"{safe}_gpu{idx}.lock")
            lk = FileLock(lock_path)
            try:
                lk.acquire(timeout=0)  # non-blocking probe
                lk.release()
                available += 1
            except FileLockTimeout:
                pass  # held by another process — GPU is in use
            except Exception:  # pylint: disable=broad-exception-caught
                # Any unexpected I/O error: assume free (safe degradation)
                available += 1
        return available

    def should_yield(self, gpu_indices: list[int]) -> bool:
        """Return True when a single-GPU grab should yield to a pending multi-GPU waiter.

        Uses true cross-process availability (file lock probing) instead of
        in-memory ``GpuAllocator._available``, which is stale in xdist mode
        because each worker process has its own independent copy.

        Yields when ALL of the following hold:
        - At least one live multi-GPU request (count > 1) is registered.
        - The pending request is satisfiable (count ≤ total GPUs on node).
        - Taking one GPU would leave fewer truly-free slots than that request needs.

        The satisfiability guard prevents a deadlock where a multi-GPU test
        registers for more GPUs than the node has, causing single-GPU tests to
        yield indefinitely while the multi-GPU test can never be satisfied.

        Args:
            gpu_indices: List of all GPU ordinals on this node (e.g. ``[0,1,2,3]``).
                         Passed to :meth:`true_available_count` for lock probing.

        Returns:
            ``True`` → caller should sleep and retry without grabbing.
            ``False`` → caller may proceed normally.
        """
        max_pending = self.max_pending_count()
        if max_pending <= 1:
            return False

        total_gpus = len(gpu_indices)
        if max_pending > total_gpus:
            # The pending request requires more GPUs than exist on this node.
            # It can never be satisfied — do NOT block single-GPU tests or the
            # session will deadlock until --gpu-acquire-timeout fires.
            logger.debug(
                "PendingTracker [%s]: max_pending=%d > total_gpus=%d; request unsatisfiable — not yielding",
                self.node_label,
                max_pending,
                total_gpus,
            )
            return False

        available = self.true_available_count(gpu_indices)
        # available - 1: how many slots remain if this caller takes one GPU
        result = (available - 1) < max_pending
        if result:
            logger.debug(
                "PendingTracker [%s]: should_yield=True "
                "(true_available=%d, available_after_grab=%d, max_pending=%d)",
                self.node_label,
                available,
                available - 1,
                max_pending,
            )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meta_lock(self):
        """Return a filelock context manager for the meta-lock file."""
        try:
            from filelock import FileLock, Timeout as FileLockTimeout  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise RuntimeError(
                "PendingAcquisitionTracker requires the 'filelock' package: pip install filelock"
            ) from exc

        class _TimedLock:
            """Acquire with a short timeout; raise RuntimeError on timeout."""

            def __init__(self, path: str) -> None:
                self._lock = FileLock(path)

            def __enter__(self):
                try:
                    self._lock.acquire(timeout=_META_LOCK_TIMEOUT)
                except FileLockTimeout as exc:
                    raise RuntimeError(f"PendingTracker: meta-lock timeout after {_META_LOCK_TIMEOUT}s") from exc
                return self

            def __exit__(self, *_):
                try:
                    self._lock.release()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass

        return _TimedLock(self._lock_path)

    def _read_unsafe(self) -> dict:
        """Read JSON file without acquiring meta-lock (caller holds it)."""
        try:
            with open(self._json_path, encoding="utf-8") as fh:
                return json.load(fh)  # type: ignore[no-any-return]
        except FileNotFoundError:
            return {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_unsafe(self, data: dict) -> None:
        """Write JSON file without acquiring meta-lock (caller holds it)."""
        with open(self._json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)


# ---------------------------------------------------------------------------
# Session-start cleanup
# ---------------------------------------------------------------------------


def cleanup_session_start(lock_dir: str = _LOCK_DIR) -> int:
    """Remove all pending-tracker JSON and meta-lock files at session start.

    Called once by the master process in ``pytest_configure`` (before any
    xdist workers are spawned).  Removes stale ``*_pending.json`` and
    ``*_pending.lock`` files left by a crashed or killed previous session
    so that ``should_yield()`` starts with a clean slate.

    Without this call, two scenarios cause indefinite blocking:

    1. A previous session crashed while ``acquire_slots()`` had a multi-GPU
       request registered.  The ``finally: deregister()`` never ran, so the
       JSON entry persists.  Single-GPU ``acquire_slot()`` callers see
       ``should_yield() = True`` and spin until ``--gpu-acquire-timeout``
       expires → ``pytest.skip()``.

    2. The system was rebooted between sessions.  ``time.monotonic()`` resets
       near zero after reboot, making old ``timeout_at`` values (set during
       the previous boot) appear to be far in the future.  The pruning check
       ``v["timeout_at"] > now`` never fires, so stale entries are never
       removed by the normal expiry path.

    The ``*_pending.lock`` (meta-lock) files are also removed: a process that
    died while holding one would leave the lock file in an inconsistent state
    for some ``filelock`` implementations, causing the 0.5 s meta-lock timeout
    to fire on the first acquisition call of the new session.

    Args:
        lock_dir: Directory containing pending-tracker files
                  (default ``output/.gpu-locks``).

    Returns:
        Number of files successfully removed.
    """
    import glob as _glob  # pylint: disable=import-outside-toplevel

    removed = 0
    for pattern in ("*_pending.json", "*_pending.lock"):
        for path in _glob.glob(os.path.join(lock_dir, pattern)):
            try:
                os.remove(path)
                logger.debug("cleanup_session_start: removed stale %s", path)
                removed += 1
            except OSError as exc:
                logger.debug("cleanup_session_start: could not remove %s: %s", path, exc)

    if removed:
        logger.info(
            "cleanup_session_start: removed %d stale pending-tracker file(s) from %s",
            removed,
            lock_dir,
        )
    return removed
