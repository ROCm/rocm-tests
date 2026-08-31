# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm and GPU environment detection for the pytest port."""

from __future__ import annotations

import contextlib
import glob
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_pci_map import GPU_DEVICE_MAP


def rocm_version_from_path(p: Path) -> str:
    """Read ROCm version from .info/version, or follow lib symlink, or guess from path."""
    info = p / ".info" / "version"
    if info.exists():
        try:
            return info.read_text().strip()
        except Exception:
            pass
    lib = p / "lib"
    if lib.is_symlink():
        try:
            real_root = Path(os.path.realpath(lib)).parent
            info = real_root / ".info" / "version"
            if info.exists():
                return info.read_text().strip()
        except Exception:
            pass
    m = re.search(r"\d+\.\d+[\d.a-zA-Z~_-]*", str(p))
    if m:
        return m.group(0)
    return "unknown"


def _render_node_index(path: str) -> int:
    """Sort key for ``/sys/class/drm/renderD<N>/...`` paths."""
    m = re.search(r"renderD(\d+)", path)
    return int(m.group(1)) if m else -1


def detect_gpu_device_id() -> str:
    """Read PCI device + revision IDs from sysfs; fall back to amd-smi.

    Returns the ``<device>_<revision>`` pair of the lowest-numbered render
    node. That pair selects the RVS config (see ``tests/_rvs_based.py``),
    so which node wins must not vary between runs on the same host.
    """
    found: list[tuple[int, str]] = []
    for f in glob.glob("/sys/class/drm/renderD*/device/device"):
        f_path = Path(f)
        try:
            dev_id = f_path.read_text().strip().replace("0x", "").lower()
        except Exception:
            continue
        if not dev_id:
            continue
        rev_file = f_path.parent / "revision"
        rev_id = "00"
        with contextlib.suppress(Exception):
            rev_id = rev_file.read_text().strip().replace("0x", "").lower() or "00"
        found.append((_render_node_index(f), f"{dev_id}_{rev_id}"))

    if found:
        found.sort()
        distinct = sorted({did for _idx, did in found})
        if len(distinct) > 1:
            print(
                f"  WARNING: render nodes report {len(distinct)} distinct PCI "
                f"device IDs ({', '.join(distinct)}); selecting "
                f"{found[0][1]} from the lowest-numbered node. RVS configs "
                f"are chosen from this one ID, so verify it matches the "
                f"GPUs under test.",
                file=sys.stderr,
            )
        return found[0][1]

    try:
        out = subprocess.run(["amd-smi", "static", "-a", "-g", "0"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if re.search(r"DEVICE_ID|DEV_ID", line, re.I):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    raw = parts[1].strip().lower()
                    match = re.fullmatch(
                        r"(?:0x)?([0-9a-f]{4})(?:[_:\-/](?:0x)?([0-9a-f]{2}))?",
                        raw,
                    )
                    if match:
                        if match.group(2) is None:
                            print(
                                f"  WARNING: amd-smi reported device "
                                f"{match.group(1)} without a revision; "
                                f"assuming revision 00. RVS config "
                                f"selection may be wrong on parts whose "
                                f"actual revision is non-zero.",
                                file=sys.stderr,
                            )
                        return f"{match.group(1)}_{match.group(2) or '00'}"
    except Exception:
        pass
    return ""


def match_rvs_gpu_dir(gpu_short_name: str, gpu_model: str, rocm_root: Path, build_dir: Path) -> str:
    """Find the RVS per-GPU config subdirectory matching this GPU."""
    if gpu_short_name:
        return gpu_short_name
    model_lower = gpu_model.lower()
    for conf_root in [
        build_dir / "rocm_validation_suite" / "build" / "bin" / "conf",
        rocm_root / "share" / "rocm-validation-suite" / "conf",
    ]:
        if not conf_root.is_dir():
            continue
        best = ""
        for d in conf_root.iterdir():
            if not d.is_dir():
                continue
            name = d.name
            if name.lower() in model_lower and len(name) > len(best):
                best = name
        if best:
            return best
    return ""


def apply_framework_environment(
    config: Config,
    *,
    rock_dir: str,
    ld_path: dict[str, str],
) -> None:
    """Apply ROCm paths from framework fixtures; fill RVS-specific fields only.

    GPU count/arch/model are supplied by ``framework_bridge.make_monitored_config``
    from ``NodePool`` / ``GpuDetector`` / ``list_devices`` — not re-probed here.
    """
    config.rocm_root = Path(rock_dir)
    print(f"ROCm path: {config.rocm_root}")

    os.environ["ROCM_PATH"] = str(config.rocm_root)
    if ld_path.get("LD_LIBRARY_PATH"):
        os.environ["LD_LIBRARY_PATH"] = ld_path["LD_LIBRARY_PATH"]
    elif (config.rocm_root / "lib").is_dir():
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        rocm_lib = str(config.rocm_root / "lib")
        os.environ["LD_LIBRARY_PATH"] = f"{rocm_lib}:{existing}" if existing else rocm_lib

    for bin_dir in (config.rocm_root / "bin", config.rocm_root / "llvm" / "bin"):
        if bin_dir.is_dir():
            os.environ["PATH"] = f"{bin_dir}:{os.environ.get('PATH', '')}"

    hipcc = config.rocm_root / "bin" / "hipcc"
    if hipcc.is_file() and os.access(hipcc, os.X_OK):
        config.clangxx = str(hipcc)
    else:
        config.clangxx = shutil.which("hipcc") or ""

    config.rocm_lib = config.rocm_root / "lib"
    config.rocm_version = rocm_version_from_path(config.rocm_root)

    try:
        for root, _dirs, _ in os.walk(config.rocm_root):
            if root.endswith("/amdgcn/bitcode"):
                os.environ["HIP_DEVICE_LIB_PATH"] = root
                break
    except Exception:
        pass

    if not config.gpu_device_id:
        config.gpu_device_id = detect_gpu_device_id()
    if not config.gpu_short_name:
        config.gpu_short_name = GPU_DEVICE_MAP.get(config.gpu_device_id, "")
    if not config.gpu_conf_dir:
        config.gpu_conf_dir = match_rvs_gpu_dir(
            config.gpu_short_name,
            config.gpu_model,
            config.rocm_root,
            config.build_dir,
        )

    print(
        f"Detected GPU: {config.gpu_model} [device={config.gpu_device_id or 'unknown'}, "
        f"arch={config.gpu_arch or 'unknown'}, mapped={config.gpu_short_name or 'none'}, "
        f"count={config.num_gpus}]"
    )
