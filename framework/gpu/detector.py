# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
detector.py -- AMD GPU detection via lspci (primary) + KFD sysfs (fallback) with amd-smi diagnostics.

Provides:
    GpuInfo         -- Immutable descriptor for a single detected AMD GPU.
    GpuDetector     -- Detects real AMD GPUs from the host system.
    MockGpuDetector -- Returns synthetic GpuInfo list for unit tests (--mock-gpu).

Detection strategy:
    Primary: ``lspci -d 1002: -nn`` — kernel PCI bus enumeration; requires no AMD
    driver and no ROCm installation; works locally and over SSH.  Returns
    GpuInfo with ``arch="unknown"`` and ``vram_mb=0``.

    Fallback: KFD sysfs (``/sys/class/kfd/kfd/topology/nodes``) — used when
    ``lspci`` is absent (e.g. inside containers without pciutils).  Requires
    no binary and no elevated permissions; also populates ``arch`` and ``vram_mb``.

    amd-smi detection is currently disabled (commented out in ``detect()``).

    Diagnostics: when GPUs are detected, ``amd-smi list`` is executed once
    on the same node for human inspection.  Its output is written to the console
    and to ``output/artifacts/gpu-info-<node>.log``.  The result is NEVER used
    for scheduling or allocation decisions.

Usage:
    detector = GpuDetector()
    gpus = detector.detect()
    for gpu in gpus:
        print(f"GPU {gpu.index}: {gpu.arch} — {gpu.vram_mb} MB VRAM")
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import json
import logging
import pathlib
import subprocess
from typing import TYPE_CHECKING

from framework.config.loader import FrameworkSection
from framework.rocm.libs.amd_smi import _get

if TYPE_CHECKING:
    from framework.executors.ssh_executor import SshExecutor

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    """Immutable descriptor for a single AMD GPU.

    Attributes:
        index:   Zero-based ordinal used for HIP_VISIBLE_DEVICES.
        arch:    GFX architecture string, e.g. ``"gfx942"``, ``"gfx1100"``.
        vram_mb: Total VRAM in megabytes.
        numa_node: NUMA node affinity (-1 if unknown).
    """

    index: int
    arch: str  # "unknown" when detected via lspci only; populated by KFD/amd-smi when re-enabled
    vram_mb: int  # 0 when detected via lspci only; populated by KFD/amd-smi when re-enabled
    numa_node: int = -1


class AbstractGpuDetector(abc.ABC):
    """Base class for GPU detectors."""

    @abc.abstractmethod
    def detect(self) -> list[GpuInfo]:
        """Return a list of available AMD GPUs.

        Returns:
            List of GpuInfo, one per GPU. Empty list if no GPUs found.
        """


def _kfd_gfx_version(raw: str) -> str:
    """Convert KFD decimal gfx_target_version to 'gfxXXX' string.

    KFD sysfs encodes the GFX target as ``major*10000 + minor*100 + stepping``
    in decimal (e.g. gfx942 → ``90402``, gfx1100 → ``110000``).
    Minor and stepping are rendered as lowercase hex to match the canonical GFX
    naming convention (e.g. stepping=10 → ``'a'`` for gfx90a).

    If *raw* is not a plain decimal integer (already "gfxXXX" or "unknown"),
    it is returned unchanged.
    """
    try:
        v = int(raw)
        major = v // 10000
        minor = (v // 100) % 100
        step = v % 100
        return f"gfx{major}{minor:x}{step:x}"
    except ValueError:
        return raw


class GpuDetector(AbstractGpuDetector):
    """Detect AMD GPUs from the host system (local) or a remote node (SSH).

    Detection strategy:
        Primary: ``lspci -d 1002: -nn`` — kernel PCI bus enumeration; requires no
        AMD driver; works locally and over SSH.

        Fallback: KFD sysfs (``/sys/class/kfd/kfd/topology/nodes``) — activated
        when ``lspci`` is absent (e.g. containers without pciutils installed).
        Requires no binary and no elevated permissions; also returns ``arch``
        and ``vram_mb``.

        amd-smi detection is disabled (commented out in ``detect()``).

        Diagnostics: ``amd-smi list`` runs once when GPUs are detected and its
        output is captured to ``output/artifacts/gpu-info-<node>.log`` for
        human inspection only.

    The detection result is cached after the first ``detect()`` call.

    Args:
        rock_dir:     Unused in this phase (reserved for amd-smi re-enablement).
        ssh_executor: When set, detection runs on the remote host via SSH.
                      When ``None`` (default), detection runs locally.
    """

    def __init__(
        self,
        rock_dir: str | None = None,
        ssh_executor: SshExecutor | None = None,
        artifact_dir: str | None = None,
    ) -> None:
        self._rock_dir = rock_dir
        self._ssh = ssh_executor
        if artifact_dir is None:
            artifact_dir = FrameworkSection().artifact_dir
        self._artifact_dir = artifact_dir
        self._cached: list[GpuInfo] | None = None

    def detect(self) -> list[GpuInfo]:
        """Detect AMD GPUs and return their descriptors.

        Results are cached after the first call.  GPU topology does not change
        during a pytest session so subsequent calls return the cached list
        without repeating detection commands.

        Primary: ``lspci -d 1002: -nn`` (works locally and over SSH; requires no
        AMD driver).  Fallback: KFD sysfs when ``lspci`` is absent (e.g. inside
        containers without pciutils).  When GPUs are detected by either method,
        ``amd-smi list`` runs once for diagnostic output only (never used for
        scheduling/allocation).  amd-smi detection is disabled below.

        Returns:
            List of ``GpuInfo``.  Empty list if no AMD GPUs are found.
        """
        if self._cached is not None:
            return list(self._cached)

        target = f"remote({self._ssh.session_key})" if self._ssh else "local"
        node_label = self._ssh.session_key if self._ssh else "localhost"

        # PRIMARY: lspci hardware enumeration (local or SSH)
        logger.info("GPU detection [%s]: trying lspci", target)
        try:
            gpus = self._detect_via_lspci()
            if gpus:
                self._cached = gpus
                self._run_amd_smi_diagnostic(node_label=node_label)
                return list(gpus)
            logger.warning("GPU detection [%s]: lspci returned 0 AMD GPUs", target)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("GPU detection [%s]: lspci failed: %s", target, exc)

        # FALLBACK: KFD sysfs — no binary required, works when lspci is absent
        try:
            gpus = self._detect_via_kfd()
            if gpus:
                self._cached = gpus
                self._run_amd_smi_diagnostic(node_label=node_label)
                return list(gpus)
            logger.warning("GPU detection [%s]: KFD sysfs returned 0 GPUs", target)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.info("GPU detection [%s]: KFD sysfs failed (%s)", target, exc)

        # --- amd-smi detection deferred; commented for future re-enablement ---
        # try:
        #     gpus = self._detect_via_amd_smi()
        #     if gpus:
        #         self._cached = gpus
        #         return list(gpus)
        # except Exception as exc:
        #     logger.info("GPU detection [%s]: system amd-smi failed (%s)", target, exc)
        #
        # if self._rock_dir:
        #     rock_amd_smi = os.path.join(self._rock_dir, "bin", "amd-smi")
        #     try:
        #         gpus = self._detect_via_amd_smi_at(rock_amd_smi)
        #         self._cached = gpus
        #         return list(gpus)
        #     except Exception as exc:
        #         logger.warning("GPU detection [%s]: rock_dir amd-smi failed: %s", target, exc)

        self._cached = []
        return []

    def _run_command(self, cmd: str) -> str:
        """Run *cmd* locally or via SSH and return stdout.

        Args:
            cmd: Shell command to run.

        Returns:
            Decoded stdout string.

        Raises:
            RuntimeError: If the command exits non-zero.
        """
        if self._ssh is not None:
            result = self._ssh.run(cmd, timeout=30)
            if result.exit_code != 0:
                raise RuntimeError(f"Remote command failed (rc={result.exit_code}): {result.stderr}")
            return result.stdout
        # Local subprocess execution
        proc = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Local command failed: {proc.stderr}")
        return proc.stdout

    def _detect_via_lspci(self) -> list[GpuInfo]:
        """Count AMD GPUs via ``lspci -d 1002: -nn`` (works locally and over SSH).

        ``lspci -d 1002:`` lists all AMD PCI devices; we count lines that contain
        "Display controller" to identify GPU entries (same method as nodelib.py).

        Returns GpuInfo with ``arch="unknown"`` and ``vram_mb=0`` — these fields
        will be populated when KFD/amd-smi detection is re-enabled in a future phase.

        Returns:
            List of GpuInfo with sequential indices 0..N-1.

        Raises:
            RuntimeError: If ``lspci`` exits non-zero or is not available.
        """
        out = self._run_command("lspci -d 1002: -nn")
        gpu_lines = [line for line in out.splitlines() if "Display controller" in line]
        logger.info("lspci detected %d AMD GPU(s)", len(gpu_lines))
        return [GpuInfo(index=i, arch="unknown", vram_mb=0) for i, _ in enumerate(gpu_lines)]

    def _run_amd_smi_diagnostic(self, node_label: str) -> None:
        """Run ``amd-smi list`` once for diagnostic output only.

        The output is written to the console (via logger.info) and to
        ``output/artifacts/gpu-info-<node_label>.log``.  The result is NEVER
        used for scheduling or allocation decisions — it is for human inspection
        and CI log archives only.

        Args:
            node_label: Human-readable node name used in the log file name
                        (e.g. ``"localhost"`` or ``"HOST_IDX_1"``).
        """
        safe_label = node_label.replace(" ", "_").replace("/", "_").replace(":", "_")
        log_path = pathlib.Path(self._artifact_dir) / f"{safe_label}_gpu_info.log"
        try:
            out = self._run_command("amd-smi list")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(out, encoding="utf-8")
            logger.info("[%s] amd-smi diagnostic:\n%s", node_label, out)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.info("[%s] amd-smi diagnostic skipped: %s", node_label, exc)

    def _detect_via_kfd(self) -> list[GpuInfo]:
        """Detect GPUs by reading Linux KFD sysfs topology nodes.

        For remote detection, runs ``cat`` on sysfs files via SSH.
        For local detection, reads the filesystem directly.
        """
        if self._ssh is not None:
            return self._detect_via_kfd_remote()

        kfd_base = pathlib.Path("/sys/class/kfd/kfd/topology/nodes")
        if not kfd_base.exists():
            raise OSError("KFD sysfs path not found")

        gpus: list[GpuInfo] = []
        for node_dir in sorted(kfd_base.iterdir()):
            prop_file = node_dir / "properties"
            if not prop_file.exists():
                continue
            props = dict(line.split(None, 1) for line in prop_file.read_text().splitlines() if " " in line)
            # Skip CPU-only nodes (gpu_id == 0)
            if props.get("gpu_id", "0").strip() == "0":
                continue
            gpus.append(
                GpuInfo(
                    index=len(gpus),
                    arch=_kfd_gfx_version(props.get("gfx_target_version", "unknown").strip()),
                    vram_mb=int(props.get("local_mem_size", "0").strip()) // (1024 * 1024),
                    numa_node=int(props.get("numa_node", "-1").strip()),
                )
            )
        return gpus

    def _detect_via_kfd_remote(self) -> list[GpuInfo]:
        """Detect GPUs on a remote host via KFD sysfs over SSH."""
        kfd_base = "/sys/class/kfd/kfd/topology/nodes"

        # List node directories
        try:
            dirs_out = self._run_command(f"ls {kfd_base}")
        except RuntimeError as exc:
            raise OSError(f"KFD sysfs not available on remote: {exc}") from exc

        node_dirs = sorted(d.strip() for d in dirs_out.splitlines() if d.strip())
        gpus: list[GpuInfo] = []
        for node_name in node_dirs:
            prop_path = f"{kfd_base}/{node_name}/properties"
            try:
                content = self._run_command(f"cat {prop_path} 2>/dev/null")
            except RuntimeError:
                continue
            if not content:
                continue
            props = dict(line.split(None, 1) for line in content.splitlines() if " " in line)
            if props.get("gpu_id", "0").strip() == "0":
                continue
            gpus.append(
                GpuInfo(
                    index=len(gpus),
                    arch=_kfd_gfx_version(props.get("gfx_target_version", "unknown").strip()),
                    vram_mb=int(props.get("local_mem_size", "0").strip()) // (1024 * 1024),
                    numa_node=int(props.get("numa_node", "-1").strip()),
                )
            )
        return gpus

    def _detect_via_amd_smi(self) -> list[GpuInfo]:
        """Detect GPUs using ``amd-smi list`` from system PATH."""
        return self._detect_via_amd_smi_at("amd-smi")

    def _detect_via_amd_smi_at(self, amd_smi_path: str) -> list[GpuInfo]:
        """Detect GPUs using ``amd-smi list`` at an explicit binary path.

        Works for both local and remote execution — the command is run through
        ``_run_command()`` which delegates to SSH when ``ssh_executor`` is set.

        Args:
            amd_smi_path: Absolute or resolvable path to the ``amd-smi`` binary.
                          Pass ``"amd-smi"`` to use the system PATH entry.

        Returns:
            List of GpuInfo parsed from ``amd-smi list --json`` output.

        Raises:
            RuntimeError: If ``amd-smi`` exits non-zero.
            FileNotFoundError: If the binary is not found at *amd_smi_path*.
        """
        raw = self._run_command(f"{amd_smi_path} list --json")
        devices = json.loads(raw)
        gpus: list[GpuInfo] = []
        for i, dev in enumerate(devices):
            total_raw = _get(
                dev,
                ("vram", "total"),  # 6.x nested {"value": N, "unit": "MB"}
                ("vram_total_mb",),  # 5.x flat MB int
                ("vram_info", "vram_total_mb"),
                default=0,
            )
            if isinstance(total_raw, dict):
                vram_mb = total_raw.get("value", 0)
            elif isinstance(total_raw, int):
                vram_mb = total_raw // (1024 * 1024) if total_raw > 1024 * 1024 else total_raw
            else:
                vram_mb = 0
            arch = _get(
                dev,
                ("asic", "target_graphics_version"),
                ("asic", "arch"),
                default="unknown",
            )
            gpus.append(GpuInfo(index=i, arch=arch, vram_mb=vram_mb))
        return gpus


class MockGpuDetector(AbstractGpuDetector):
    """Synthetic GPU detector for unit tests and ``--mock-gpu`` mode.

    Returns a configurable list of fake GpuInfo objects without touching
    any hardware or system paths.

    Args:
        gpus: List of GpuInfo to return from detect(). Defaults to two
              synthetic gfx942 GPUs with 32 GB VRAM each.
    """

    def __init__(self, gpus: list[GpuInfo] | None = None) -> None:
        self._gpus = gpus or [
            GpuInfo(index=0, arch="gfx942", vram_mb=32768, numa_node=0),
            GpuInfo(index=1, arch="gfx942", vram_mb=32768, numa_node=1),
        ]

    def detect(self) -> list[GpuInfo]:
        """Return the preconfigured synthetic GPU list.

        Returns:
            List of GpuInfo as supplied at construction time.
        """
        logger.debug("MockGpuDetector returning %d synthetic GPUs", len(self._gpus))
        return list(self._gpus)
