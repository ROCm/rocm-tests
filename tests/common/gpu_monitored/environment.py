# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ROCm and GPU environment detection for the pytest port."""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import sys

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_pci_map import GPU_DEVICE_MAP, ConfDirUnresolvedError, detect_device_key


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


def detect_gpu_device_key(cmake_executor=None) -> str:
    """Return the ``<device>_<revision>`` key the RVS config mapping is indexed by.

    Delegates to the detector the RVS module tests use, so both suites resolve a
    host to the same key -- including the power-variant split for boards that
    share a device id. Detection follows *cmake_executor* to the node under
    test, which is what makes remote-node runs pick the config of the GPU that
    will actually run the workload rather than the one in the machine running
    pytest.

    An undetectable GPU yields ``""``, which the mapping reports as UNSUPPORTED
    rather than failing the run.
    """
    try:
        return detect_device_key(cmake_executor=cmake_executor)
    except ConfDirUnresolvedError as e:
        print(f"  WARNING: {e}; RVS tests will report UNSUPPORTED", file=sys.stderr)
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
        config.gpu_device_id = detect_gpu_device_key()
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
