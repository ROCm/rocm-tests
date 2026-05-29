# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
node_pool.py -- Fleet manager: GPU topology discovery, slot acquisition, and release.

NodePool discovers GPUs at session start, maintains per-node GpuAllocator instances,
and provides file-locked slot acquisition (acquire_slot, acquire_slots,
acquire_multi_node). Used exclusively via target_executor — not called from tests.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import threading
import time

from framework.gpu.allocator import GpuAllocator
from framework.gpu.detector import GpuDetector, GpuInfo
from framework.nodes.gpu_file_lock import GpuFileLock
from framework.nodes.node_spec import NodeSpec
from framework.nodes.pending_tracker import PendingAcquisitionTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NodeSlot — atomic (node, gpu) resource unit
# ---------------------------------------------------------------------------


@dataclass
class NodeSlot:
    """Atomic resource: one (node, gpu) pair acquired from the pool.

    Returned by ``NodePool.acquire_slot()`` and ``acquire_slots()``.
    Call ``make_executor()`` to get a fully configured executor for this slot.

    Attributes:
        node_spec:  Descriptor for the hosting node.
        gpu_info:   GPU metadata (arch, VRAM, NUMA node).
        _file_lock: Active ``GpuFileLock`` held by this slot.
        _ssh:       ``SshExecutor`` for remote nodes; ``None`` for local.
    """

    node_spec: NodeSpec
    gpu_info: GpuInfo
    _file_lock: GpuFileLock
    _ssh: object | None = field(default=None, repr=False)  # SshExecutor | None

    @property
    def gpu_label(self) -> str:
        """Short label for this GPU, e.g. ``"GPU-0"``."""
        return f"GPU-{self.gpu_info.index}"

    def make_executor(
        self,
        test_id: str = "",
        log_path: str | None = None,
        session_log_path: str | None = None,
    ):
        """Return an executor with logging context attached.

        LOCAL slot:  ``LocalExecutor(gpu_index=N, log_config=LogConfig(...))``
        REMOTE slot: ``SshExecutor(gpu_indices=[N], log_config=LogConfig(...))``

        ``LogConfig`` carries the test, node, and GPU labels together with the
        log file paths.  ``run()`` on the returned executor applies the shared
        7-step logging protocol (prefixed console output + timestamped log files)
        without any additional wrapper class.

        Args:
            test_id:          Test function name for the label prefix.
            log_path:         Per-test log file (append mode).
            session_log_path: Session-wide aggregate log file (append mode).

        Returns:
            ``LocalExecutor`` or ``SshExecutor`` with ``log_config`` set.
        """
        from framework.executors.log_config import LogConfig  # pylint: disable=import-outside-toplevel

        log_cfg = LogConfig(
            test_id=test_id,
            node_label=self.node_spec.label,
            gpu_label=self.gpu_label,
            log_path=log_path,
            session_log_path=session_log_path,
        )

        if self.node_spec.is_local:
            from framework.executors.local_executor import LocalExecutor  # pylint: disable=import-outside-toplevel

            return LocalExecutor(
                gpu_index=self.gpu_info.index,
                stream_stdout=True,
                stream_stderr=False,
                log_path=None,
                log_config=log_cfg,
            )

        from framework.executors.ssh_executor import (  # pylint: disable=import-outside-toplevel
            SshExecutor as _SshExecutor,
        )

        ssh: _SshExecutor = self._ssh  # type: ignore[assignment]
        ssh.gpu_indices = [self.gpu_info.index]
        ssh.log_config = log_cfg
        return ssh


@dataclass
class MultiGpuSlots:
    """Group of ``NodeSlot`` objects from the SAME node (intra-node multi-GPU).

    Returned by ``NodePool.acquire_slots()`` for
    ``@pytest.mark.hw.multi_gpu`` tests.

    Attributes:
        slots:      Ordered list of acquired slots (same node, different GPUs).
        node_spec:  The shared node descriptor.
    """

    slots: list[NodeSlot]
    node_spec: NodeSpec
    _log_path: str | None = field(default=None, repr=False)
    _session_log_path: str | None = field(default=None, repr=False)

    @property
    def gpu_indices(self) -> list[int]:
        """GPU ordinals in allocation order."""
        return [s.gpu_info.index for s in self.slots]

    @property
    def count(self) -> int:
        """Number of GPUs in this group."""
        return len(self.slots)

    def make_executor(
        self,
        test_id: str = "",
        log_path: str | None = None,
        session_log_path: str | None = None,
    ):
        """Return a ``LocalExecutor`` or ``SshExecutor`` with LogConfig attached and all GPUs visible.

        LOCAL: ``LocalExecutor(gpu_index=[0,1,...], stream_stdout=True, log_config=...)``
        REMOTE: ``SshExecutor(ssh, gpu_indices=[0,1,...], log_config=...)``

        The inner executor streams stdout live to the console. stderr is captured
        via ``LogConfig`` for post-call logging and Allure attachment.

        Log paths are resolved in priority order: explicit argument → fixture-injected
        attribute (``_log_path`` / ``_session_log_path``) → default (``None``).
        The fixture injects these so tests only need to pass ``test_id``.

        Args:
            test_id:          Test function name for the label prefix.
            log_path:         Per-test log file (append mode).  Falls back to
                              ``self._log_path`` when ``None``.
            session_log_path: Session-wide aggregate log file (append mode).  Falls
                              back to ``self._session_log_path`` when ``None``.

        Returns:
            ``LocalExecutor`` or ``SshExecutor`` with ``log_config`` attached (all GPUs visible).
        """
        from framework.executors.log_config import LogConfig  # pylint: disable=import-outside-toplevel

        effective_log = log_path or getattr(self, "_log_path", None)
        effective_session_log = session_log_path or getattr(self, "_session_log_path", None)

        gpu_label = f"GPU-{','.join(str(i) for i in self.gpu_indices)}"
        log_cfg = LogConfig(
            test_id=test_id,
            node_label=self.node_spec.label,
            gpu_label=gpu_label,
            log_path=effective_log,
            session_log_path=effective_session_log,
        )

        if self.node_spec.is_local:
            from framework.executors.local_executor import LocalExecutor  # pylint: disable=import-outside-toplevel

            return LocalExecutor(
                gpu_index=self.gpu_indices,  # list[int] → ROCR_VISIBLE_DEVICES=N,M,...
                stream_stdout=True,
                stream_stderr=False,
                log_path=None,
                log_config=log_cfg,
            )

        from framework.executors.ssh_executor import (  # pylint: disable=import-outside-toplevel
            SshExecutor as _SshExecutorMulti,
        )

        ssh_multi: _SshExecutorMulti = self.slots[0]._ssh  # type: ignore[assignment]
        ssh_multi.gpu_indices = self.gpu_indices
        ssh_multi.log_config = log_cfg
        return ssh_multi


# ---------------------------------------------------------------------------
# NodePool — fleet manager
# ---------------------------------------------------------------------------


class NodePool:
    """Fleet manager: discovers GPUs, provides file-locked slot acquisition.

    At construction time, ``NodePool`` runs GPU detection on all nodes in
    parallel (via ``ThreadPoolExecutor``).  Each node gets its own
    ``GpuAllocator``.  Slot acquisition (``acquire_slot()``) picks the best
    available (node, GPU) pair, acquires a ``GpuFileLock`` for cross-process
    exclusivity, and returns a ``NodeSlot``.

    LOCAL mode (no ``--remote-node``):
        Pass ``node_specs=[NodeSpec(hostname="localhost", label="localhost")]``.
        No SSH connections are opened.  ``GpuDetector(ssh_executor=None)``
        runs detection locally.

    REMOTE mode:
        Pass ``node_specs`` from ``HostConfigLoader.load(host_yaml_path)``.
        One ``SshExecutor`` is opened per remote node for GPU detection
        and for subsequent test execution.

    Attributes:
        node_specs:  Ordered list of node descriptors.
        _allocators: Per-node ``GpuAllocator`` keyed by ``node_spec.label``.
        _ssh_sessions: Per-node ``SshExecutor`` (remote only).
        _pool_lock:  Mutex protecting cross-node slot selection.
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        node_specs: list[NodeSpec],
        rock_dir: str | None = None,
        headroom_gb: float = 0.0,
        detect_timeout: float = 60.0,
        prefilled_gpus: dict[str, list[GpuInfo]] | None = None,
        artifact_dir: str | None = None,
    ) -> None:
        """Discover GPUs on all nodes and initialise per-node allocators.

        Args:
            node_specs:     Ordered list of node descriptors to manage.
            rock_dir:       TheRock/ROCm install path passed to ``GpuDetector``
                            for container fallback.
            headroom_gb:    VRAM headroom per GPU (passed to ``GpuAllocator``).
            detect_timeout: Seconds allowed for parallel GPU detection
                            (default 60 s; increase for slow SSH connections).
            prefilled_gpus: Pre-detected GPU lists keyed by node label.  When
                            provided, GPU detection is skipped and allocators
                            are built directly from this mapping.  Used by
                            xdist workers that receive topology from the master
                            to avoid redundant and potentially flaky re-detection.
            artifact_dir:   Directory for GPU info diagnostic logs (forwarded to
                            ``GpuDetector``).  When ``None``, the value is read
                            from ``FrameworkSection.artifact_dir`` in ``loader.py``.
        """
        self.node_specs = list(node_specs)
        self._allocators: dict[str, GpuAllocator] = {}
        self._ssh_sessions: dict[str, object] = {}  # label → SshExecutor
        self._pool_lock = threading.Lock()
        self._rock_dir = rock_dir
        self._headroom_gb = headroom_gb
        if artifact_dir is None:
            from framework.config.loader import FrameworkSection  # pylint: disable=import-outside-toplevel

            artifact_dir = FrameworkSection().artifact_dir
        self._artifact_dir = artifact_dir

        if prefilled_gpus is not None:
            self._init_from_prefilled(prefilled_gpus)
        else:
            self._detect_all(detect_timeout)

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _init_from_prefilled(self, prefilled: dict[str, list[GpuInfo]]) -> None:
        """Build per-node allocators from a pre-detected GPU map (xdist worker mode).

        Skips SSH connections and ``lspci``/``amd-smi`` detection entirely.
        Called instead of ``_detect_all`` when the master already serialized
        the GPU topology and passed it via xdist ``workerinput``.

        Args:
            prefilled: Mapping of node label → list of ``GpuInfo`` as detected
                       by the master process.
        """

        class _StaticDetector:
            def __init__(self, gpu_list: list) -> None:
                self._gpus = gpu_list

            def detect(self) -> list:
                return list(self._gpus)

        for spec in self.node_specs:
            gpus = prefilled.get(spec.label, [])
            allocator = GpuAllocator(
                detector=_StaticDetector(gpus),  # type: ignore[arg-type]
                headroom_gb=self._headroom_gb,
            )
            self._allocators[spec.label] = allocator
            logger.info(
                "NodePool (worker): %s — using %d pre-detected GPU(s): %s",
                spec.label,
                len(gpus),
                [f"GPU-{g.index}({g.arch})" for g in gpus],
            )

    def _detect_all(self, timeout: float) -> None:
        """Detect GPUs on all nodes in parallel and build per-node allocators."""
        with ThreadPoolExecutor(max_workers=len(self.node_specs) or 1) as pool:
            futures = {pool.submit(self._detect_node, spec): spec for spec in self.node_specs}
            for fut in as_completed(futures, timeout=timeout):
                spec = futures[fut]
                try:
                    fut.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    logger.warning("GPU detection failed for node %s: %s", spec.label, exc)

    def _detect_node(self, spec: NodeSpec) -> None:
        """Detect GPUs on *spec* and register the allocator."""
        ssh = None
        if not spec.is_local:
            from framework.executors.ssh_executor import SshExecutor  # pylint: disable=import-outside-toplevel

            ssh = SshExecutor(
                host=spec.hostname,
                user=spec.username,
                key_path=spec.ssh_key,
                password=spec.password,
            )
            self._ssh_sessions[spec.label] = ssh
            logger.info("NodePool: opened SSH session to %s (%s)", spec.label, spec.hostname)

        detector = GpuDetector(rock_dir=self._rock_dir, ssh_executor=ssh, artifact_dir=self._artifact_dir)
        gpus = detector.detect()

        class _StaticDetector:
            """Wrap a pre-detected list so GpuAllocator can call detect()."""

            def __init__(self, gpu_list):
                self._gpus = gpu_list

            def detect(self):
                return list(self._gpus)

        allocator = GpuAllocator(
            detector=_StaticDetector(gpus),  # type: ignore[arg-type]
            headroom_gb=self._headroom_gb,
        )
        self._allocators[spec.label] = allocator
        logger.info(
            "NodePool: %s — detected %d GPU(s): %s",
            spec.label,
            len(gpus),
            [f"GPU-{g.index}({g.arch})" for g in gpus],
        )

    # ------------------------------------------------------------------
    # Topology reporting
    # ------------------------------------------------------------------

    def total_gpu_slots(self) -> int:
        """Return the total number of GPU slots across all nodes."""
        total = 0
        for allocator in self._allocators.values():
            total += len(allocator._pool)
        return total

    def pool_status(self) -> tuple[int, int]:
        """Return ``(available_slots, total_slots)`` across all nodes.

        Thread-safe: reads each allocator's ``_available`` set under its lock
        so the snapshot is consistent even when xdist workers are concurrently
        acquiring and releasing slots.

        Returns:
            Tuple of ``(available, total)`` GPU slot counts.
        """
        available = 0
        total = 0
        for allocator in self._allocators.values():
            with allocator._lock:
                available += len(allocator._available)
            total += len(allocator._pool)
        return available, total

    def topology_summary(self) -> str:
        """Return a one-line topology description for session start output.

        Example: ``"localhost(2)"`` or ``"AXA-01(4) + AXA-02(4)"``

        Returns:
            Human-readable topology string.
        """
        parts = []
        for spec in self.node_specs:
            alloc = self._allocators.get(spec.label)
            count = len(alloc._pool) if alloc else 0
            parts.append(f"{spec.label}({count})")
        return " + ".join(parts)

    # ------------------------------------------------------------------
    # Slot acquisition
    # ------------------------------------------------------------------

    def acquire_slot(  # noqa: C901  # pylint: disable=too-many-locals,too-many-branches
        self,
        vram_required_gb: float = 0.0,
        wait_timeout_secs: float = 300.0,
        test_id: str = "",
    ) -> NodeSlot:
        """Acquire one GPU slot from any available node.

        Iterates ``node_specs`` in order, attempting to acquire from each
        node's ``GpuAllocator``.  The first successful allocation acquires
        the corresponding ``GpuFileLock`` and returns a ``NodeSlot``.

        When all nodes are exhausted without a successful allocation, the
        call polls every 0.5 seconds until *wait_timeout_secs* expires.

        GPU allocation is architecture-agnostic.  ``--gpu-arch`` is passed
        to compilation (``--offload-arch``) but does not filter slot selection.

        Args:
            vram_required_gb:  Minimum VRAM needed (GB).
            wait_timeout_secs: Seconds to wait when no slot is immediately
                               available (default 30 s).
            test_id:           Test function name recorded in the GPU lock
                               metadata file for cross-process tracking.

        Returns:
            ``NodeSlot`` with an active ``GpuFileLock``.

        Raises:
            RuntimeError: If no slot is available within *wait_timeout_secs*.
        """
        deadline = time.monotonic() + wait_timeout_secs

        while True:
            # ------------------------------------------------------------------
            # Priority check — before attempting acquisition.
            # If a multi-GPU waiter is registered in PendingAcquisitionTracker
            # and taking one GPU would leave fewer slots than it needs, yield.
            # This is cross-process: the tracker file is shared across xdist workers.
            # ------------------------------------------------------------------
            yielded = False
            for spec in self.node_specs:
                alloc = self._allocators.get(spec.label)
                gpu_indices = [g.index for g in alloc._pool] if alloc else []
                tracker = PendingAcquisitionTracker(spec.label)
                if tracker.should_yield(gpu_indices):
                    logger.debug(
                        "acquire_slot: yielding to pending multi-GPU waiter on %s",
                        spec.label,
                    )
                    yielded = True
                    break

            if not yielded:
                with self._pool_lock:
                    for spec in self.node_specs:
                        alloc = self._allocators.get(spec.label)
                        if alloc is None:
                            continue

                        # Try every GPU on this node before moving to the next.
                        # When a file lock fails, keep the GPU "allocated" in-memory
                        # (out of _available) so alloc.allocate() skips it and picks
                        # the next GPU.  Release all temporarily-held GPUs after the
                        # scan completes (success or exhaustion).
                        skipped: list[GpuInfo] = []
                        found_slot: NodeSlot | None = None

                        while found_slot is None:
                            try:
                                gpu_info = alloc.allocate(
                                    vram_required_gb=vram_required_gb,
                                    wait_timeout_secs=0.0,  # non-blocking
                                )
                            except RuntimeError:
                                break  # No more GPUs available on this node

                            # Acquire the file lock for cross-process safety.
                            # Use non-blocking (timeout=0.0) — the outer polling loop
                            # handles the wait.  This prevents blocking inside _pool_lock
                            # and avoids the 5 s x N stall that caused worker deadlocks.
                            file_lock = GpuFileLock(
                                node_label=spec.label,
                                gpu_index=gpu_info.index,
                            )
                            try:
                                file_lock.acquire(timeout=0.0, test_id=test_id)
                                # Success — release any GPUs we held during the scan.
                                for skipped_gpu in skipped:
                                    alloc.release(skipped_gpu)
                                ssh = self._ssh_sessions.get(spec.label)
                                found_slot = NodeSlot(
                                    node_spec=spec,
                                    gpu_info=gpu_info,
                                    _file_lock=file_lock,
                                    _ssh=ssh,
                                )
                            except RuntimeError:
                                # Lock held by another process — do NOT release
                                # gpu_info yet.  Keeping it out of _available forces
                                # the next alloc.allocate() call to pick a different
                                # GPU on this node instead of retrying the same one.
                                skipped.append(gpu_info)

                        # Restore any GPUs temporarily held during the scan.
                        for skipped_gpu in skipped:
                            alloc.release(skipped_gpu)

                        if found_slot is not None:
                            return found_slot

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"NodePool: no GPU slot available after {wait_timeout_secs}s. "
                    f"Topology: {self.topology_summary()}. "
                    "Consider using --gpu-acquire-timeout to increase the timeout."
                )

            logger.debug(
                "NodePool: no slot available, waiting 0.5s (%.1fs remaining)",
                max(0.0, deadline - time.monotonic()),
            )
            time.sleep(0.5)

    def acquire_slots(  # pylint: disable=too-many-locals,too-many-positional-arguments
        self,
        count: int,
        node_label: str | None = None,
        vram_required_gb: float = 0.0,
        wait_timeout_secs: float = 300.0,
        test_id: str = "",
    ) -> MultiGpuSlots:
        """Acquire *count* GPU slots from the SAME node (intra-node multi-GPU).

        All acquired slots come from one node.  If *node_label* is specified,
        only that node is tried.  Otherwise, the first node with enough
        available slots is chosen.

        The acquisition is all-or-nothing: if a node cannot provide all
        *count* GPUs simultaneously, the call waits and retries.

        GPU allocation is architecture-agnostic.  ``--gpu-arch`` is used for
        compilation (``--offload-arch``) but does not filter slot selection.

        Args:
            count:             Number of GPUs to acquire.
            node_label:        Specific node to allocate from (``None`` = any).
            vram_required_gb:  Minimum VRAM per GPU (GB).
            wait_timeout_secs: Seconds to wait when not enough slots are
                               immediately available (default 30 s).

        Returns:
            ``MultiGpuSlots`` with ``count`` active slots from one node.

        Raises:
            RuntimeError: If *count* slots are unavailable within the timeout.
        """
        deadline = time.monotonic() + wait_timeout_secs

        # Register intent in the cross-process pending tracker so that
        # single-GPU callers (acquire_slot) yield rather than taking slots
        # we need.  Each candidate node gets its own registration because
        # we don't yet know which node will satisfy the request.
        candidates_for_reg = [s for s in self.node_specs if s.label == node_label] if node_label else self.node_specs
        tracker_regs: dict[str, str] = {}  # node_label → request_id
        for spec in candidates_for_reg:
            tracker = PendingAcquisitionTracker(spec.label)
            req_id = tracker.register(count=count, timeout_at=deadline)
            tracker_regs[spec.label] = req_id
            logger.debug(
                "acquire_slots: registered pending request for %d GPU(s) on %s (req=%s)",
                count,
                spec.label,
                req_id[:8],
            )

        try:
            while True:
                with self._pool_lock:
                    candidates = (
                        [s for s in self.node_specs if s.label == node_label] if node_label else self.node_specs
                    )
                    for spec in candidates:
                        alloc = self._allocators.get(spec.label)
                        if alloc is None:
                            continue
                        acquired: list[NodeSlot] = []
                        try:
                            for _ in range(count):
                                gpu = alloc.allocate(
                                    vram_required_gb=vram_required_gb,
                                    wait_timeout_secs=0.0,
                                )
                                # CRITICAL: acquire file lock non-blocking. If it
                                # fails, release the in-memory slot BEFORE raising so
                                # that the GPU is not phantom-allocated in this worker's
                                # _available set.  Without this, workers accumulate
                                # "ghost" allocations and eventually see an empty
                                # _available set while no file locks are actually held
                                # — causing all workers to hang until timeout.
                                file_lock = GpuFileLock(
                                    node_label=spec.label,
                                    gpu_index=gpu.index,
                                )
                                try:
                                    file_lock.acquire(timeout=0.0, test_id=test_id)
                                except RuntimeError:
                                    alloc.release(gpu)  # restore before raising
                                    raise
                                ssh = self._ssh_sessions.get(spec.label)
                                acquired.append(
                                    NodeSlot(
                                        node_spec=spec,
                                        gpu_info=gpu,
                                        _file_lock=file_lock,
                                        _ssh=ssh,
                                    )
                                )
                        except RuntimeError:
                            # Not enough GPUs on this node — release fully-acquired slots.
                            # The GPU that failed the file lock is already released above.
                            for slot in acquired:
                                slot._file_lock.release()
                                alloc.release(slot.gpu_info)
                            acquired = []
                            continue

                        if acquired:
                            return MultiGpuSlots(slots=acquired, node_spec=spec)

                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"NodePool: cannot acquire {count} GPU slots from one node "
                        f"after {wait_timeout_secs}s. "
                        f"Topology: {self.topology_summary()}"
                    )
                logger.debug(
                    "NodePool: %d-GPU slots not available, waiting 0.5s (%.1fs remaining)",
                    count,
                    max(0.0, deadline - time.monotonic()),
                )
                time.sleep(0.5)

        finally:
            # Deregister on success OR timeout so single-GPU callers
            # stop yielding once the competition for these slots is resolved.
            for lbl, req_id in tracker_regs.items():
                PendingAcquisitionTracker(lbl).deregister(req_id)

    def acquire_multi_node(  # pylint: disable=too-many-positional-arguments
        self,
        gpu_count_per_node: int = 1,
        vram_required_gb: float = 0.0,
        wait_timeout_secs: float = 30.0,
        test_id: str = "",
    ) -> list[MultiGpuSlots]:
        """Acquire *gpu_count_per_node* GPU slots from EACH node in the fleet.

        Used for multi-node tests (``@pytest.mark.e2e.multinode``) that need
        one or more GPUs from every available node simultaneously.

        GPU allocation is architecture-agnostic.  ``--gpu-arch`` is used for
        compilation (``--offload-arch``) but does not filter slot selection.

        Args:
            gpu_count_per_node: GPUs to acquire from each node (default 1).
            vram_required_gb:   Minimum VRAM per GPU (GB).
            wait_timeout_secs:  Total wait timeout for acquiring all nodes (s).

        Returns:
            List of ``MultiGpuSlots``, one per node.

        Raises:
            RuntimeError: If any node cannot provide the requested slots.
        """
        result: list[MultiGpuSlots] = []
        for spec in self.node_specs:
            slots = self.acquire_slots(
                count=gpu_count_per_node,
                node_label=spec.label,
                vram_required_gb=vram_required_gb,
                wait_timeout_secs=wait_timeout_secs,
                test_id=test_id,
            )
            result.append(slots)
        return result

    # ------------------------------------------------------------------
    # Release
    # ------------------------------------------------------------------

    def release(self, slots: list[NodeSlot]) -> None:
        """Return *slots* to their per-node allocators and release file locks.

        Args:
            slots: List of ``NodeSlot`` previously returned by an acquire method.
        """
        for slot in slots:
            alloc = self._allocators.get(slot.node_spec.label)
            if alloc is not None:
                alloc.release(slot.gpu_info)
            slot._file_lock.release()
            logger.debug(
                "NodePool: released %s GPU-%d",
                slot.node_spec.label,
                slot.gpu_info.index,
            )

    def release_multi(self, multi: MultiGpuSlots) -> None:
        """Release all slots in a ``MultiGpuSlots`` group.

        Args:
            multi: ``MultiGpuSlots`` previously returned by ``acquire_slots()``.
        """
        self.release(multi.slots)

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def close_ssh_sessions(self) -> None:
        """Close all SSH sessions opened for remote GPU detection and execution.

        Called by ``remote_node_plugin.pytest_sessionfinish``.
        Safe to call when no SSH sessions have been opened (local mode).
        """
        for label, ssh in list(self._ssh_sessions.items()):
            try:
                ssh.close()  # type: ignore[attr-defined]
                logger.debug("NodePool: closed SSH session to %s", label)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("NodePool: error closing SSH session to %s: %s", label, exc)
        self._ssh_sessions.clear()
