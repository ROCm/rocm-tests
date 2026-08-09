# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
detector.py -- AMD GPU detection via lspci, KFD sysfs, and amd-smi.

Detection strategy (layered enrichment):
  1. lspci -d 1002: -nn  — counts AMD GPUs; no driver required, works over SSH.
                           Returns arch="unknown", vram_mb=0 (lspci cannot read VRAM).
  2. KFD sysfs enrichment — run after lspci to populate arch and vram_mb per GPU.
                           On discrete GPUs (MI308X, etc.) local_mem_size is correct.
                           On APUs (MI300A), local_mem_size is 0 (kernel driver TODO).
  3. amd-smi enrichment  — run for any GPU whose vram_mb is still 0 after KFD,
                           covering MI300A unified-memory and any other APU variant.
  Fallback (no lspci):   KFD sysfs → system amd-smi → rock_dir amd-smi, in order.

Diagnostics: amd-smi list runs once when GPUs are detected and its output is
captured to output/artifacts/gpu-info-<node>.log for human inspection only.
Use MockGpuDetector for --mock-gpu.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import json
import logging
import os
import pathlib
import subprocess
from typing import TYPE_CHECKING

from framework.config.loader import FrameworkSection
from framework.rocm.libs.amd_smi import _get

if TYPE_CHECKING:
    from framework.executors.ssh_executor import SshExecutor

logger = logging.getLogger(__name__)

# PCI class name substrings that identify AMD GPU devices in lspci output.
# "Display controller" — discrete GPUs on consumer/workstation cards.
# "VGA compatible controller" — older Radeon discrete GPUs.
# "Processing accelerators" — AMD Instinct compute GPUs (MI200/MI300 family).
_GPU_PCI_CLASSES = ("Display controller", "VGA compatible controller", "Processing accelerators")


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
    arch: str  # "unknown" until enriched by KFD/amd-smi; gfx-arch string after enrichment
    vram_mb: int  # 0 until enriched; populated by KFD (discrete GPUs) or amd-smi (APUs like MI300A)
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

    Detection strategy (layered enrichment):
        1. ``lspci -d 1002: -nn`` — counts AMD GPUs; no driver required; works
           locally and over SSH.  Returns ``arch="unknown"``, ``vram_mb=0``.
        2. KFD sysfs enrichment — run after lspci to populate ``arch`` and
           ``vram_mb`` from ``/sys/class/kfd/kfd/topology/nodes``.  Correct for
           discrete GPUs; returns ``vram_mb=0`` for APUs (MI300A) because the
           Linux KFD driver does not implement ``local_mem_size`` for APU nodes.
        3. ``amd-smi`` enrichment — run for any GPU with ``vram_mb=0`` after KFD,
           resolving unified-memory APUs like MI300A.  Tries system ``amd-smi``
           first, then ``rock_dir/bin/amd-smi`` if configured.

        Fallback (when lspci finds 0 GPUs): KFD sysfs → system amd-smi →
        rock_dir amd-smi, in order.

        Diagnostics: ``amd-smi list`` runs once when GPUs are detected and its
        output is captured to ``output/artifacts/gpu-info-<node>.log`` for
        human inspection only.

    The detection result is cached after the first ``detect()`` call.

    Args:
        rock_dir:     Path to TheRock/ROCm install; used as fallback amd-smi
                      location (``<rock_dir>/bin/amd-smi``) during enrichment.
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

    def detect(self) -> list[GpuInfo]:  # noqa: C901
        """Detect AMD GPUs and return their descriptors.

        Results are cached after the first call.  GPU topology does not change
        during a pytest session so subsequent calls return the cached list
        without repeating detection commands.

        Detection uses a layered enrichment strategy:

        1. ``lspci -d 1002: -nn`` counts AMD GPUs (works locally and over SSH,
           no driver required).  Returns ``arch="unknown"`` and ``vram_mb=0``
           because lspci cannot read VRAM.
        2. KFD sysfs enrichment runs after lspci to populate ``arch`` and
           ``vram_mb`` using ``local_mem_size``.  Correct for discrete GPUs
           (MI308X, etc.); returns ``vram_mb=0`` for APUs (MI300A) because the
           Linux KFD driver does not implement ``local_mem_size`` for APU nodes.
        3. ``amd-smi`` enrichment runs for any GPU whose ``vram_mb`` is still 0
           after KFD, covering MI300A unified-memory and any other APU variant.

        Fallback (when lspci finds 0 GPUs): KFD sysfs → system amd-smi →
        rock_dir amd-smi, in order.

        Returns:
            List of ``GpuInfo``.  Empty list if no AMD GPUs are found.
        """
        if self._cached is not None:
            return list(self._cached)

        target = f"remote({self._ssh.session_key})" if self._ssh else "local"
        node_label = self._ssh.session_key if self._ssh else "localhost"

        # PRIMARY: lspci hardware enumeration (local or SSH).
        # Returns GpuInfo with arch="unknown", vram_mb=0 — enriched below.
        logger.info("GPU detection [%s]: trying lspci", target)
        lspci_gpus: list[GpuInfo] = []
        try:
            lspci_gpus = self._detect_via_lspci()
            if not lspci_gpus:
                logger.warning("GPU detection [%s]: lspci returned 0 AMD GPUs", target)
                # Print raw lspci output so engineers can see what PCI class names are present.
                try:
                    raw = self._run_command("lspci -d 1002: -nn")
                    if raw.strip():
                        print(f"\n[rocm-test] lspci AMD devices (all classes, unfiltered):\n{raw.strip()}")
                    else:
                        print(
                            "\n[rocm-test] lspci found NO AMD PCI devices (vendor 1002). "
                            "Check that the GPU is passed through to this container "
                            "(--device /dev/kfd --device /dev/dri)."
                        )
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("GPU detection [%s]: lspci failed: %s", target, exc)

        if lspci_gpus:
            # lspci found GPUs — enrich arch and vram_mb via KFD, then amd-smi.
            gpus = self._enrich_via_kfd(lspci_gpus, target)
            gpus = self._enrich_via_amd_smi(gpus, target)
            self._cached = gpus
            self._run_amd_smi_diagnostic(node_label=node_label)
            return list(gpus)

        # FALLBACK: KFD sysfs — no binary required, works when lspci is absent
        # (e.g. containers without pciutils). Also enriches arch and vram_mb.
        try:
            gpus = self._detect_via_kfd()
            if gpus:
                # KFD returns vram_mb=0 for APU nodes — still try amd-smi enrichment.
                gpus = self._enrich_via_amd_smi(gpus, target)
                self._cached = gpus
                self._run_amd_smi_diagnostic(node_label=node_label)
                return list(gpus)
            logger.warning("GPU detection [%s]: KFD sysfs returned 0 GPUs", target)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.info("GPU detection [%s]: KFD sysfs failed (%s)", target, exc)

        # Print KFD and DRI diagnostics (local only — sysfs not traversable over SSH).
        if self._ssh is None:
            self._print_kfd_dri_diagnostics()

        # THIRD FALLBACK: amd-smi from PATH. This covers direct host/container
        # runs where pciutils/KFD sysfs are unavailable but ROCm tools are usable.
        try:
            gpus = self._detect_via_amd_smi()
            if gpus:
                logger.info("GPU detection [%s]: system amd-smi detected %d GPU(s)", target, len(gpus))
                self._cached = gpus
                self._run_amd_smi_diagnostic(node_label=node_label)
                return list(gpus)
            logger.warning("GPU detection [%s]: system amd-smi returned 0 GPU(s)", target)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("GPU detection [%s]: system amd-smi failed: %s", target, exc)

        # FOURTH FALLBACK: amd-smi from TheRock rock_dir.
        # Activated when lspci/KFD/system amd-smi all return 0.
        if self._rock_dir:
            rock_amd_smi = os.path.join(self._rock_dir, "bin", "amd-smi")
            try:
                gpus = self._detect_via_amd_smi_at(rock_amd_smi)
                if gpus:
                    logger.info("GPU detection [%s]: rock_dir amd-smi detected %d GPU(s)", target, len(gpus))
                    self._cached = gpus
                    self._run_amd_smi_diagnostic(node_label=node_label)
                    return list(gpus)
                logger.warning("GPU detection [%s]: rock_dir amd-smi returned 0 GPU(s)", target)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.warning("GPU detection [%s]: rock_dir amd-smi failed: %s", target, exc)

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
        proc = subprocess.run(  # nosec B602 — shell=True required; detection commands are framework-controlled system calls, not user input
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
        are enriched by ``_enrich_via_kfd()`` and ``_enrich_via_amd_smi()`` in ``detect()``.

        Returns:
            List of GpuInfo with sequential indices 0..N-1.

        Raises:
            RuntimeError: If ``lspci`` exits non-zero or is not available.
        """
        out = self._run_command("lspci -d 1002: -nn")
        all_amd_lines = [line for line in out.splitlines() if line.strip()]
        gpu_lines = [line for line in all_amd_lines if any(cls in line for cls in _GPU_PCI_CLASSES)]

        if all_amd_lines and not gpu_lines:
            # AMD PCI devices present but none match known GPU class names.
            # This can happen with unrecognised class strings — log the raw output.
            logger.warning(
                "lspci found %d AMD device(s) but none match GPU PCI classes %s. " "Raw lspci output:\n%s",
                len(all_amd_lines),
                list(_GPU_PCI_CLASSES),
                out.strip(),
            )

        logger.info(
            "lspci detected %d AMD GPU(s) (from %d total AMD PCI device(s))",
            len(gpu_lines),
            len(all_amd_lines),
        )
        return [GpuInfo(index=i, arch="unknown", vram_mb=0) for i, _ in enumerate(gpu_lines)]

    def _enrich_via_kfd(self, gpus: list[GpuInfo], target: str) -> list[GpuInfo]:
        """Enrich *gpus* (from lspci) with ``arch`` and ``vram_mb`` from KFD sysfs.

        lspci returns ``arch="unknown"`` and ``vram_mb=0``.  KFD sysfs provides
        real arch and VRAM for discrete GPUs.  For APUs (e.g. MI300A), the Linux
        KFD driver does not implement ``local_mem_size`` so ``vram_mb`` stays 0;
        ``_enrich_via_amd_smi`` handles those cases afterwards.

        GPU count from lspci is authoritative.  If KFD returns a different count
        (e.g. topology nodes include CPU nodes filtered by gpu_id) the KFD data
        is used only when counts match; otherwise the original lspci list is
        returned unchanged and a warning is logged.

        Args:
            gpus:   GpuInfo list produced by ``_detect_via_lspci()``.
            target: Human-readable target label for log messages.

        Returns:
            Enriched list with the same length as *gpus*.  Individual GPUs that
            could not be matched retain their original ``arch`` and ``vram_mb``.
        """
        try:
            kfd_gpus = self._detect_via_kfd()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.info("GPU detection [%s]: KFD enrichment skipped (%s)", target, exc)
            return gpus

        if not kfd_gpus:
            logger.info("GPU detection [%s]: KFD returned 0 nodes — skipping enrichment", target)
            return gpus

        if len(kfd_gpus) != len(gpus):
            logger.warning(
                "GPU detection [%s]: lspci found %d GPU(s) but KFD found %d — "
                "skipping KFD enrichment to avoid index mismatch",
                target,
                len(gpus),
                len(kfd_gpus),
            )
            return gpus

        enriched = [
            GpuInfo(
                index=lspci_gpu.index,
                arch=kfd_gpu.arch if kfd_gpu.arch != "unknown" else lspci_gpu.arch,
                vram_mb=kfd_gpu.vram_mb,
                numa_node=kfd_gpu.numa_node if kfd_gpu.numa_node != -1 else lspci_gpu.numa_node,
            )
            for lspci_gpu, kfd_gpu in zip(gpus, kfd_gpus, strict=True)
        ]
        vram_populated = sum(1 for g in enriched if g.vram_mb > 0)
        logger.info(
            "GPU detection [%s]: KFD enriched %d/%d GPU(s) with VRAM data",
            target,
            vram_populated,
            len(enriched),
        )
        return enriched

    def _enrich_via_amd_smi(self, gpus: list[GpuInfo], target: str) -> list[GpuInfo]:
        """Enrich any GPU in *gpus* that still has ``vram_mb=0`` using ``amd-smi``.

        Called after lspci+KFD enrichment.  Targets APU variants (e.g. MI300A)
        where the KFD driver does not populate ``local_mem_size``.  If all GPUs
        already have ``vram_mb > 0`` this method is a no-op.

        Tries system ``amd-smi`` first; falls back to ``rock_dir/bin/amd-smi``
        when ``rock_dir`` is configured.  If ``amd-smi`` returns a different GPU
        count than *gpus*, enrichment is skipped and a warning is logged.

        Args:
            gpus:   GpuInfo list (may have ``vram_mb=0`` for some entries).
            target: Human-readable target label for log messages.

        Returns:
            List with the same length as *gpus*.  GPUs that already had
            ``vram_mb > 0`` are returned unchanged.  APU GPUs with
            ``vram_mb=0`` are updated with ``amd-smi`` data where available.
        """
        if all(g.vram_mb > 0 for g in gpus):
            return gpus

        zero_count = sum(1 for g in gpus if g.vram_mb == 0)
        logger.info(
            "GPU detection [%s]: %d/%d GPU(s) have vram_mb=0 after KFD — attempting amd-smi enrichment",
            target,
            zero_count,
            len(gpus),
        )

        smi_candidates: list[str] = ["amd-smi"]
        if self._rock_dir:
            smi_candidates.append(os.path.join(self._rock_dir, "bin", "amd-smi"))

        for smi_path in smi_candidates:
            try:
                smi_gpus = self._detect_via_amd_smi_at(smi_path)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                logger.info("GPU detection [%s]: amd-smi enrichment (%s) failed: %s", target, smi_path, exc)
                continue

            if not smi_gpus:
                logger.warning("GPU detection [%s]: amd-smi enrichment (%s) returned 0 GPU(s)", target, smi_path)
                continue

            if len(smi_gpus) != len(gpus):
                logger.warning(
                    "GPU detection [%s]: amd-smi returned %d GPU(s) but pool has %d — "
                    "skipping amd-smi enrichment to avoid index mismatch",
                    target,
                    len(smi_gpus),
                    len(gpus),
                )
                continue

            enriched = [
                GpuInfo(
                    index=g.index,
                    arch=smi_gpu.arch if smi_gpu.arch != "unknown" else g.arch,
                    vram_mb=smi_gpu.vram_mb if g.vram_mb == 0 else g.vram_mb,
                    numa_node=g.numa_node,
                )
                for g, smi_gpu in zip(gpus, smi_gpus, strict=True)
            ]
            vram_populated = sum(1 for g in enriched if g.vram_mb > 0)
            logger.info(
                "GPU detection [%s]: amd-smi enrichment via '%s' resolved %d/%d GPU(s) with VRAM data",
                target,
                smi_path,
                vram_populated,
                len(enriched),
            )
            return enriched

        logger.warning(
            "GPU detection [%s]: amd-smi enrichment exhausted all candidates — %d GPU(s) retain vram_mb=0",
            target,
            zero_count,
        )
        return gpus

    def _print_kfd_dri_diagnostics(self) -> None:
        """Print KFD sysfs and /dev/dri state to the console when local detection returns 0.

        Runs only on the local node (no SSH). Output helps engineers distinguish between
        "KFD sysfs not mounted in container", "all nodes are CPU-only", and "GPU not passed through".
        """
        kfd_path = pathlib.Path("/sys/class/kfd/kfd/topology/nodes")
        if kfd_path.exists():
            try:
                nodes = sorted(kfd_path.iterdir())
                print(
                    f"\n[rocm-test] KFD topology: {len(nodes)} node(s) found "
                    "(all skipped because gpu_id=0 indicates CPU-only nodes):"
                )
                for node_dir in nodes[:8]:  # cap to avoid flooding the console
                    prop_file = node_dir / "properties"
                    if prop_file.exists():
                        props = dict(line.split(None, 1) for line in prop_file.read_text().splitlines() if " " in line)
                        print(
                            f"  node {node_dir.name}: "
                            f"gpu_id={props.get('gpu_id', '?').strip()}, "
                            f"gfx_target_version={props.get('gfx_target_version', '?').strip()}, "
                            f"name={props.get('name', '?').strip()}"
                        )
            except OSError as exc:
                print(f"\n[rocm-test] KFD sysfs exists but could not be read: {exc}")
        else:
            print(
                "\n[rocm-test] /sys/class/kfd not found. The ROCm kernel driver (amdgpu) "
                "may not be loaded, or the KFD sysfs tree is not exposed inside this container. "
                "Verify the host has amdgpu loaded: lsmod | grep amdgpu"
            )

        dri_path = pathlib.Path("/dev/dri")
        if dri_path.exists():
            try:
                dri_devices = sorted(d.name for d in dri_path.iterdir())
                print(f"[rocm-test] /dev/dri devices present: {dri_devices}")
            except OSError:
                print("[rocm-test] /dev/dri exists but could not be listed.")
        else:
            print("[rocm-test] /dev/dri not found — GPU device may not be passed through to this container.")

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
        """Detect GPUs using ``amd-smi static`` from system PATH."""
        return self._detect_via_amd_smi_at("amd-smi")

    def _detect_via_amd_smi_at(self, amd_smi_path: str) -> list[GpuInfo]:
        """Detect GPUs using ``amd-smi static --json`` at an explicit binary path.

        Uses ``amd-smi static`` (not ``amd-smi list``) because only ``static``
        exposes VRAM size and ASIC architecture in its JSON output.
        ``amd-smi list`` returns only identifiers (BDF, UUID) with no VRAM/arch fields.

        JSON schema across ROCm versions (priority order in ``_get()`` fallbacks):

        VRAM total (field: ``vram_mb``):
          - ROCm 7.x: ``vram.size.value``  (int MB, unit confirmed in ``vram.size.unit``)
          - ROCm 6.x: ``vram.total.value`` (nested ``{"value": N, "unit": "MB"}``)
          - ROCm 5.x: ``vram_total_mb``    (flat int, already in MB)

        Architecture (field: ``arch``):
          - All versions: ``asic.target_graphics_version`` (e.g. ``"gfx942"``)
          - Fallback:     ``asic.arch``

        Works for both local and remote execution — the command is run through
        ``_run_command()`` which delegates to SSH when ``ssh_executor`` is set.

        Args:
            amd_smi_path: Absolute or resolvable path to the ``amd-smi`` binary.
                          Pass ``"amd-smi"`` to use the system PATH entry.

        Returns:
            List of GpuInfo parsed from ``amd-smi static --json`` output.

        Raises:
            RuntimeError: If ``amd-smi`` exits non-zero.
            FileNotFoundError: If the binary is not found at *amd_smi_path*.
        """
        raw = self._run_command(f"{amd_smi_path} static --json")
        devices = json.loads(raw)
        gpus: list[GpuInfo] = []
        for i, dev in enumerate(devices):
            total_raw = _get(
                dev,
                ("vram", "size"),  # ROCm 7.x: {"value": N, "unit": "MB"}
                ("vram", "total"),  # ROCm 6.x: {"value": N, "unit": "MB"}
                ("vram_total_mb",),  # ROCm 5.x: flat int MB
                ("vram_info", "vram_total_mb"),
                default=0,
            )
            if isinstance(total_raw, dict):
                vram_mb = int(total_raw.get("value", 0))
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
