# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixture for RVS.

Checks for a pre-installed RVS binary first. If not found, clones
ROCmValidationSuite from GitHub using the framework's external_build
utility and builds from source via CMake.

Source: https://github.com/ROCm/ROCmValidationSuite
Binary: rvs
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import subprocess

import pytest

logger = logging.getLogger(__name__)

_RVS_REPO_URL = "https://github.com/ROCm/ROCmValidationSuite.git"
_RVS_REF = os.environ.get("ROCM_TEST_RVS_REF", "master")

"""PCI device ID + revision -> short name (used for RVS config directory lookup)"""
_GPU_DEVICE_MAP = {
    "66a1_00": "MI50", "66a1_06": "MI50",
    "738c_01": "MI100", "738c_cc": "MI100",
    "740f_02": "MI210", "7410_02": "MI210",
    "740c_01": "MI250", "7408_00": "MI250",
    "74a0_00": "MI300A", "74b4_00": "MI300A",
    "74a1_00": "MI300X", "74b5_00": "MI300X",
    "74a9_00": "MI300X-HF", "74bd_00": "MI300X-HF",
    "74a2_00": "MI308X", "74b6_00": "MI308X",
    "74a8_00": "MI308X-HF", "74bc_00": "MI308X-HF",
    "74a5_00": "MI325X", "74b9_00": "MI325X",
    "75a0_00": "MI350X", "75b0_00": "MI350X",
    "75a3_00": "MI355X", "75b3_00": "MI355X",
    "73a3_00": "nv21", "73ae_00": "nv21",
    "7448_00": "nv31", "7448_ec": "nv31",
    "744c_c0": "nv31", "744c_c8": "nv31", "744c_cc": "nv31",
    "744c_ce": "nv31", "744c_cf": "nv31", "744c_e0": "nv31",
    "744c_ec": "nv31", "744c_e8": "nv31", "744c_ee": "nv31",
    "745e_cc": "nv31", "7449_00": "nv31", "744a_00": "nv31",
    "7460_00": "nv32", "7461_00": "nv32", "7470_00": "nv32",
    "747e_c8": "nv32", "747e_c9": "nv32", "747e_ff": "nv32",
    "747e_d8": "nv32", "747e_d9": "nv32", "747e_db": "nv32",
    "748f_30": "gfx1200", "748f_31": "gfx1200", "748f_32": "RX9060",
    "748f_f0": "gfx1200", "748f_f1": "gfx1200", "748f_f2": "gfx1200",
    "748f_f3": "RX9060", "7590_c0": "gfx1200", "7590_c7": "RX9060",
    "746f_30": "RX9070GRE", "746f_31": "gfx1201", "746f_32": "RX9070",
    "746f_f0": "RX9070GRE", "746f_f1": "RX9070GRE", "746f_f2": "RX9070GRE",
    "746f_f3": "RX9070", "746f_f4": "RX9070", "746f_f5": "gfx1201",
    "746f_f6": "RX9070GRE", "7550_c0": "gfx1201", "7550_c3": "RX9070GRE",
    "7551_c0": "gfx1201",
}


def _is_rvs_installed(rock_dir: str, cmake_executor=None) -> bool:
    """Check if RVS is pre-installed with binary and config files."""
    rock_dir_path = pathlib.Path(rock_dir)
    rvs = rock_dir_path / "bin" / "rvs"
    conf = rock_dir_path / "share" / "rocm-validation-suite" / "conf"

    if cmake_executor is None:
        return rvs.is_file() and os.access(rvs, os.X_OK) and conf.is_dir()

    result = cmake_executor.run(f"test -x {rvs} && test -d {conf}")
    return result.ok


def _file_exists(path: pathlib.Path, cmake_executor=None) -> bool:
    """Check if a file exists locally or on remote node."""
    if cmake_executor is None:
        return path.is_file()
    return cmake_executor.run(f"test -f {path}").ok


def _detect_gpu_conf_dir(cmake_executor=None) -> str:
    """Detect GPU PCI device ID and map to RVS config directory name."""
    cmd = "lspci -n -d 1002: | grep -E '0300|1200' | head -1"
    if cmake_executor is None:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        line = result.stdout.strip()
    else:
        result = cmake_executor.run(cmd)
        line = (result.stdout or "").strip()

    if not line:
        logger.warning("No AMD GPU detected via lspci")
        return ""

    match = re.search(r"1002:([0-9a-f]{4})", line, re.IGNORECASE)
    if not match:
        logger.warning("Could not parse device ID from lspci line: %s", line)
        return ""
    device_id = match.group(1).lower()

    rev_match = re.search(r"\(rev\s+([0-9a-f]+)\)", line, re.IGNORECASE)
    rev = rev_match.group(1).lower() if rev_match else "00"

    key = f"{device_id}_{rev}"
    gpu_name = _GPU_DEVICE_MAP.get(key, "")

    if gpu_name:
        logger.info("Detected GPU: device_id=%s, rev=%s, key=%s -> %s", device_id, rev, key, gpu_name)
    else:
        logger.warning(
            "GPU detected (device_id=%s, rev=%s, key=%s) but no mapping found in GPU_DEVICE_MAP",
            device_id, rev, key,
        )

    return gpu_name


@pytest.fixture(scope="session")
def rvs_source(external_build, compiler_build_dir: str, cmake_executor) -> str:
    """Clone ROCmValidationSuite with submodules once per session; return source path."""
    dest = pathlib.Path(compiler_build_dir) / "rvs" / "ROCmValidationSuite"
    src_dir = external_build.clone_repo(_RVS_REPO_URL, dest, ref=_RVS_REF)
    external_build.assert_license_present(src_dir)
    if cmake_executor is not None:
        cmake_executor.run(
            f"cd {src_dir} && git submodule update --init --recursive",
            timeout=120.0,
        )
    else:
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=str(src_dir), check=True, timeout=120,
        )
    return str(src_dir)


@pytest.fixture(scope="session")
def rvs_binary(
    rock_dir: str,
    rvs_source: str,
    compiler_build_dir: str,
    cmake_build_dir,
    cmake_executor,
    built_binary,
):
    """Locate or build the RVS binary.

    Priority:
      1. Pre-installed at {rock_dir}/bin/rvs
      2. Previously built at {rvs_source}/install/
      3. Build from source using framework cmake_build_dir + DESTDIR install
    """
    src_dir = pathlib.Path(rvs_source)
    install_dir = src_dir / "install"

    # 1. Check pre-installed
    preinstalled = pathlib.Path(rock_dir) / "bin" / "rvs"
    if _is_rvs_installed(rock_dir, cmake_executor):
        logger.info("Using pre-installed RVS: %s", preinstalled)
        return str(preinstalled)

    # 2. Check previously built
    if install_dir.exists():
        for candidate in install_dir.rglob("bin/rvs"):
            if candidate.is_file():
                logger.info("Using previously built RVS: %s", candidate)
                return str(candidate)

    # 3. Build from source via framework cmake_build_dir
    logger.info("Building RVS from source: %s", src_dir)

    build_dir = cmake_build_dir(
        src=str(src_dir),
        subdir="rvs",
        extra_cmake_args=[
            f"-DROCM_PATH={rock_dir}",
            f"-DCMAKE_PREFIX_PATH={rock_dir}",
            f"-DHIPCC_PATH={rock_dir}",
        ],
        compiler_mode="optional_auto",
        label="rvs",
        artifact="rvs",
    )

    # 4. Install step using DESTDIR (RVS uses absolute install paths)
    logger.info("Running cmake --install for RVS with DESTDIR=%s", install_dir)
    install_dir.mkdir(parents=True, exist_ok=True)

    if cmake_executor is not None:
        result = cmake_executor.run(
            f"DESTDIR={install_dir} cmake --install {build_dir}",
            timeout=120.0,
        )
        if not result.ok:
            pytest.fail(f"RVS cmake install failed:\n{(result.stderr or '')[:3000]}")
    else:
        env = os.environ.copy()
        env["DESTDIR"] = str(install_dir)
        proc = subprocess.run(
            ["cmake", "--install", str(build_dir)],
            env=env, capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            pytest.fail(f"RVS cmake install failed:\n{proc.stderr[:3000]}")

    # 5. Locate installed binary
    rvs_bin = None
    for candidate in install_dir.rglob("bin/rvs"):
        if candidate.is_file():
            rvs_bin = candidate
            break

    if rvs_bin is None:
        pytest.fail(
            f"RVS binary not found under {install_dir} after install. "
            f"Contents: {list(install_dir.rglob('*'))[:20]}"
        )

    logger.info("RVS binary installed at: %s", rvs_bin)
    return built_binary(str(rvs_bin), "rvs")


@pytest.fixture(scope="session")
def gpu_conf_dir(cmake_executor) -> str:
    """Auto-detect GPU and return the matching RVS config directory name."""
    return _detect_gpu_conf_dir(cmake_executor)


@pytest.fixture(scope="session")
def rvs_find_conf(rock_dir: str, rvs_source: str, cmake_executor, rvs_binary: str):
    """Return a factory that locates RVS config files with GPU-specific lookup."""
    rock_dir_path = pathlib.Path(rock_dir)
    install_base = pathlib.Path(rvs_source) / "install"
    install_conf = None
    if install_base.exists():
        for candidate in install_base.rglob("share/rocm-validation-suite/conf"):
            if candidate.is_dir():
                install_conf = candidate
                break
    source_conf = pathlib.Path(rvs_source) / "rvs" / "conf"

    def _find_conf(config_name: str, *, gpu_only: bool = False, gpu_conf_dir: str = "") -> str:
        search_roots = []
        installed_conf = rock_dir_path / "share" / "rocm-validation-suite" / "conf"

        if cmake_executor is None:
            for p in (installed_conf, install_conf, source_conf):
                if p is not None and p.is_dir():
                    search_roots.append(p)
        else:
            for p in (installed_conf, install_conf, source_conf):
                if p is None:
                    continue
                result = cmake_executor.run(f"test -d {p}")
                if result.ok:
                    search_roots.append(p)

        # GPU-specific lookup
        for root in search_roots:
            if gpu_conf_dir:
                candidate = root / gpu_conf_dir / config_name
                if _file_exists(candidate, cmake_executor):
                    logger.info("Resolved config %s -> %s (GPU-specific: %s)", config_name, candidate, gpu_conf_dir)
                    return str(candidate)

        if gpu_only:
            pytest.skip(
                f"GPU-specific config {config_name} not found under {gpu_conf_dir} "
                f"in {[str(r) for r in search_roots]}"
            )

        # Generic fallback
        for root in search_roots:
            candidate = root / config_name
            if _file_exists(candidate, cmake_executor):
                logger.info("Resolved config %s -> %s (generic fallback)", config_name, candidate)
                return str(candidate)

        pytest.skip(f"RVS config {config_name} not found in {[str(r) for r in search_roots]}")

    return _find_conf
