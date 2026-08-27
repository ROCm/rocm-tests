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

import pytest

from framework.executors.local_executor import run_cmd_get_stdout_stderr
from tests.common.gpu_pci_map import detect_gpu_conf_dir_from_lspci

logger = logging.getLogger(__name__)

_RVS_REPO_URL = "https://github.com/ROCm/ROCmValidationSuite.git"
_RVS_REF = os.environ.get("ROCM_TEST_RVS_REF", "master")


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
    return detect_gpu_conf_dir_from_lspci(cmake_executor=cmake_executor)


def _collect_conf_roots(
    *paths: pathlib.Path | None,
    cmake_executor=None,
) -> list[pathlib.Path]:
    """Return the subset of candidate paths that exist as directories."""
    roots: list[pathlib.Path] = []
    for p in paths:
        if p is None:
            continue
        if cmake_executor is None:
            if p.is_dir():
                roots.append(p)
        elif cmake_executor.run(f"test -d {p}").ok:
            roots.append(p)
    return roots


def _resolve_conf_file(
    config_name: str,
    search_roots: list[pathlib.Path],
    cmake_executor=None,
    gpu_conf_dir: str = "",
) -> str | None:
    """Search for a config file across roots, trying GPU-specific path first."""
    if gpu_conf_dir:
        for root in search_roots:
            candidate = root / gpu_conf_dir / config_name
            if _file_exists(candidate, cmake_executor):
                logger.info("Resolved config %s -> %s (GPU-specific: %s)", config_name, candidate, gpu_conf_dir)
                return str(candidate)

    for root in search_roots:
        candidate = root / config_name
        if _file_exists(candidate, cmake_executor):
            logger.info("Resolved config %s -> %s (generic fallback)", config_name, candidate)
            return str(candidate)

    return None


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
        rc, _stdout, stderr = run_cmd_get_stdout_stderr(
            "git",
            "submodule",
            "update",
            "--init",
            "--recursive",
            cwd=str(src_dir),
            timeout=120,
        )
        if rc != 0:
            pytest.fail(f"git submodule update failed:\n{stderr}")
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
        rc, _stdout, stderr = run_cmd_get_stdout_stderr(
            "cmake",
            "--install",
            str(build_dir),
            env={"DESTDIR": str(install_dir)},
            timeout=120,
        )
        if rc != 0:
            pytest.fail(f"RVS cmake install failed:\n{stderr[:3000]}")

    # 5. Locate installed binary
    rvs_bin = None
    for candidate in install_dir.rglob("bin/rvs"):
        if candidate.is_file():
            rvs_bin = candidate
            break

    if rvs_bin is None:
        pytest.fail(
            f"RVS binary not found under {install_dir} after install. " f"Contents: {list(install_dir.rglob('*'))[:20]}"
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
        installed_conf = rock_dir_path / "share" / "rocm-validation-suite" / "conf"
        search_roots = _collect_conf_roots(
            installed_conf,
            install_conf,
            source_conf,
            cmake_executor=cmake_executor,
        )

        resolved = _resolve_conf_file(config_name, search_roots, cmake_executor, gpu_conf_dir)
        if resolved:
            return resolved

        if gpu_only:
            pytest.skip(
                f"GPU-specific config {config_name} not found under {gpu_conf_dir} "
                f"in {[str(r) for r in search_roots]}"
            )

        pytest.skip(f"RVS config {config_name} not found in {[str(r) for r in search_roots]}")

    return _find_conf


def _rvs_install_base(rvs_source: str | None) -> pathlib.Path | None:
    """Return the RVS ``install`` tree when the source checkout has one."""
    if not rvs_source:
        return None
    base = pathlib.Path(rvs_source) / "install"
    return base if base.exists() else None


def _export_rvs_conf_root(rvs_source: str, rock_dir: str, install_base: pathlib.Path | None) -> None:
    """Point workloads at the first readable RVS ``conf`` tree."""
    install_conf = None
    if install_base is not None:
        install_conf = next(
            (c for c in install_base.rglob("share/rocm-validation-suite/conf") if c.is_dir()),
            None,
        )
    for root in (
        install_conf,
        pathlib.Path(rock_dir) / "share" / "rocm-validation-suite" / "conf",
        pathlib.Path(rvs_source) / "rvs" / "conf",
    ):
        if root is not None and root.is_dir():
            os.environ["ROCM_TEST_RVS_CONF_ROOT"] = str(root)
            return


def _export_transferbench_bin(
    rvs_binary: str,
    rock_dir: str,
    compiler_build_dir: str | None,
    install_base: pathlib.Path | None,
) -> None:
    """Publish a prebuilt TransferBench found near ROCm, RVS or the build tree."""
    candidates = [
        pathlib.Path(rock_dir) / "bin" / "TransferBench",
        pathlib.Path(rvs_binary).parent / "TransferBench",
    ]
    if compiler_build_dir:
        build_root = pathlib.Path(compiler_build_dir) / "transferbench"
        candidates += [build_root / "TransferBench", build_root / "build" / "TransferBench"]
    if install_base is not None:
        candidates.extend(install_base.rglob("bin/TransferBench"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            os.environ["ROCM_TEST_TRANSFERBENCH_BIN"] = str(candidate)
            return


def export_rvs_env_paths(
    rvs_binary: str | None,
    rvs_source: str | None,
    rock_dir: str,
    *,
    transferbench_binary: str | None = None,
    compiler_build_dir: str | None = None,
) -> None:
    """Publish RVS / TransferBench paths for gpu_monitored workload modules."""
    if rvs_binary:
        os.environ["ROCM_TEST_RVS_BIN"] = rvs_binary

    install_base = _rvs_install_base(rvs_source)
    if rvs_source:
        _export_rvs_conf_root(rvs_source, rock_dir, install_base)

    if transferbench_binary:
        os.environ["ROCM_TEST_TRANSFERBENCH_BIN"] = transferbench_binary
        return

    if rvs_binary:
        _export_transferbench_bin(rvs_binary, rock_dir, compiler_build_dir, install_base)


@pytest.fixture(scope="session")
def transferbench_binary(
    rock_dir: str,
    compiler_build_dir: str,
    cmake_build_dir,
    cmake_executor,
    built_binary,
    request,
) -> str:
    """Locate or build TransferBench from the RVS external submodule."""
    for candidate in (
        pathlib.Path(rock_dir) / "bin" / "TransferBench",
        pathlib.Path(compiler_build_dir) / "transferbench" / "TransferBench",
        pathlib.Path(compiler_build_dir) / "transferbench" / "build" / "TransferBench",
    ):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            logger.info("Using TransferBench: %s", candidate)
            return str(candidate)

    # Only clone/build RVS submodules when TransferBench is not already present.
    rvs_source = request.getfixturevalue("rvs_source")
    tb_src = pathlib.Path(rvs_source) / "external" / "TransferBench"
    if not (tb_src / "CMakeLists.txt").is_file():
        pytest.fail(f"TransferBench source not found at {tb_src}. Ensure RVS submodules are initialized.")

    logger.info("Building TransferBench from source: %s", tb_src)
    build_dir = cmake_build_dir(
        src=str(tb_src),
        subdir="transferbench",
        extra_cmake_args=[
            f"-DROCM_PATH={rock_dir}",
            f"-DCMAKE_PREFIX_PATH={rock_dir}",
        ],
        compiler_mode="optional_auto",
        label="transferbench",
        artifact="TransferBench",
    )
    tb_bin = pathlib.Path(build_dir) / "TransferBench"
    if not tb_bin.is_file() and cmake_executor is None:
        # cmake_build_dir returns the build directory; binary lives there.
        pass
    return built_binary(str(tb_bin), "TransferBench")
