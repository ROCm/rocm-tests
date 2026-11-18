# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KFD (kfdtest) build helpers shared by KFD test suites.

Sparse-clones the libhsakmt project from the public rocm-systems monorepo,
validates that the required thunk headers are present, and CMake-builds the
``kfdtest`` GTest binary.

Source: https://github.com/ROCm/rocm-systems (projects/rocr-runtime/libhsakmt)
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shlex
import shutil
import subprocess

from framework.builder.binary_builder import find_rocm_clangpp

logger = logging.getLogger(__name__)

_KFDTEST_MONOREPO_URL = "https://github.com/ROCm/rocm-systems.git"
# Whole libhsakmt project is checked out — kfdtest's CMakeLists.txt resolves
# the thunk headers via ../../include, so the sibling include/ tree is required.
_LIBHSAKMT_SUBPATH = "projects/rocr-runtime/libhsakmt"
_KFDTEST_BUILD_SUBDIR = "tests/kfdtest"
_DEFAULT_KFDTEST_REF = "develop"

_THUNK_HEADERS = ("hsakmt/hsakmt.h", "hsakmt/linux/kfd_ioctl.h")
# pkg-config modules kfdtest's CMakeLists.txt requires at configure time.
_PROVISIONING_PKGS = ("libdrm", "libnuma")

# Build prerequisites matching the original test's provisioning step.
_KFD_BUILD_PREREQS: dict[str, list[str]] = {
    "apt-get": ["libdrm-dev", "libnuma-dev", "pkg-config"],
    "dnf": ["libdrm-devel", "numactl-devel", "pkgconfig"],
    "zypper": ["libdrm-devel", "libnuma-devel", "pkg-config"],
}


def _sudo() -> list[str]:
    return [] if getattr(os, "geteuid", lambda: 1)() == 0 else ["sudo"]


def install_kfd_build_prereqs(executor=None) -> None:
    """Install KFD/kfdtest build prerequisites using the first available package manager.

    Installs libdrm, libnuma, and pkg-config dev headers required by
    kfdtest's CMakeLists.txt, matching the original test's provisioning step.
    """
    for pm, pkgs in _KFD_BUILD_PREREQS.items():
        if shutil.which(pm) is None:
            continue
        logger.info("KFD prereqs: installing %s via %s", pkgs, pm)
        if pm == "apt-get":
            cmd = [*_sudo(), "apt-get", "install", "-y", *pkgs]
        elif pm == "dnf":
            cmd = [*_sudo(), "dnf", "install", "-y", *pkgs]
        else:
            cmd = [*_sudo(), "zypper", "--non-interactive", "--no-gpg-checks", "install", *pkgs]

        if executor is not None:
            result = executor.run(" ".join(shlex.quote(c) for c in cmd), timeout=300.0)
            if not result.ok:
                logger.warning("KFD prereq install failed (exit=%d): %s", result.exit_code, result.stderr[:500])
        else:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                logger.warning("KFD prereq install failed (exit=%d): %s", proc.returncode, proc.stderr[:500])
        return
    logger.warning(
        "KFD prereqs: no supported package manager found. Install libdrm-dev libnuma-dev pkg-config manually."
    )


def find_kfdtest_binary(rock_dir: str, executor=None) -> str:
    """Locate a pre-installed kfdtest binary under the ROCm prefix.

    Mirrors the original test which first tries to install the kfdtest
    package before building from source.

    Returns:
        Absolute path to the kfdtest binary, or "" when not found.
    """
    candidates = [
        pathlib.Path(rock_dir) / "bin" / "kfdtest",
        pathlib.Path(rock_dir) / "kfdtest" / "bin" / "kfdtest",
    ]
    for base in sorted(pathlib.Path(rock_dir).glob("extras*")):
        candidates.append(base / "bin" / "kfdtest")

    for candidate in candidates:
        if executor is not None:
            if executor.run(f"test -x '{candidate}'", timeout=15.0).ok:
                return str(candidate)
        else:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return ""


def build_kfdtest_binary(
    rock_dir: str,
    external_build,
    cmake_executor,
    framework_config,
    cmake_build_dir_fn,
    built_binary_fn,
    gpu_arch: str | None,
    compiler_build_dir: str,
) -> str:
    """Sparse-clone libhsakmt and CMake-build kfdtest; return the binary path.

    Checks out the whole ``libhsakmt`` project (not just ``tests/kfdtest``) so
    that kfdtest's CMakeLists.txt can resolve the sibling ``include/`` tree it
    needs for the HSAkmt headers.

    Args:
        rock_dir:             ROCm install root.
        external_build:       Framework external-build helper.
        cmake_executor:       Remote cmake executor; ``None`` means local.
        framework_config:     Framework config (provides ``build_timeout_secs``).
        cmake_build_dir_fn:   ``cmake_build_dir`` fixture callable.
        built_binary_fn:      ``built_binary`` fixture callable.
        gpu_arch:             Optional GPU arch string (used for logging only).
        compiler_build_dir:   Output directory for compiled artefacts.

    Returns:
        Absolute path to the built ``kfdtest`` binary.

    Raises:
        RuntimeError: When a required pkg-config build dependency is missing.
    """
    rocm_path = os.path.realpath(rock_dir)
    ref = os.environ.get("KFDTEST_REF", _DEFAULT_KFDTEST_REF)
    build_timeout = float(framework_config.therock.build_timeout_secs)

    # Install build prerequisites (libdrm, libnuma, pkg-config) before cmake.
    install_kfd_build_prereqs(cmake_executor)

    libhsakmt_src = external_build.clone_repo(
        url=_KFDTEST_MONOREPO_URL,
        dest=pathlib.Path(compiler_build_dir) / "kfd" / f"rocm-systems-{_safe_ref_name(ref)}",
        ref=ref,
        timeout=build_timeout,
        sparse_subtree=_LIBHSAKMT_SUBPATH,
    )
    external_build.assert_license_present(libhsakmt_src)

    kfdtest_src = pathlib.Path(libhsakmt_src) / _KFDTEST_BUILD_SUBDIR
    assert_thunk_headers_present(pathlib.Path(libhsakmt_src) / "include", cmake_executor)

    try:
        build_dir = cmake_build_dir_fn(
            src=str(kfdtest_src),
            subdir=f"kfd/kfdtest-{_safe_ref_name(ref)}",
            compiler_mode="auto",
            extra_cmake_args=[
                *_kfd_compiler_args(rocm_path),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_EXE_LINKER_FLAGS=-ldl",
                f"-DROCM_DIR={rocm_path}",
                f"-DOPENCL_DIR={rocm_path}",
            ],
            label="kfdtest",
        )
    except (AssertionError, RuntimeError) as exc:
        missing = _missing_provisioning_pkg(str(exc))
        if missing is None:
            raise
        raise RuntimeError(
            f"kfdtest build dependency '{missing}' is not provisioned on the build node — "
            "cmake configure could not find its pkg-config module."
        ) from exc

    return str(built_binary_fn(os.path.join(str(build_dir), "kfdtest"), "kfdtest"))


def assert_thunk_headers_present(include_dir: pathlib.Path, executor) -> None:
    """Fail fast when the libhsakmt thunk headers are missing from the checkout.

    kfdtest compiles ``#include "hsakmt/hsakmt.h"`` against the sibling
    ``libhsakmt/include`` tree. A stale sparse checkout that omitted the
    sibling ``include/`` directory produces a late, opaque compiler error —
    this guard surfaces it early with actionable guidance.

    Args:
        include_dir: ``libhsakmt/include`` path in the checkout.
        executor:    Remote executor; ``None`` means local.

    Raises:
        RuntimeError: When any required header is missing.
    """
    missing: list[str] = []
    for header in _THUNK_HEADERS:
        header_path = include_dir / header
        if executor is not None:
            present = executor.run(f"test -f {header_path}", timeout=30.0).ok
        else:
            present = header_path.is_file()
        if not present:
            missing.append(str(header_path))

    if missing:
        raise RuntimeError(
            "kfdtest thunk headers are missing from the checkout:\n  "
            + "\n  ".join(missing)
            + f"\nThe sibling include/ tree under {include_dir.parent} was not checked out "
            "alongside tests/kfdtest. Remove the cached checkout and re-run so the whole "
            "libhsakmt project (including include/) is fetched."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_ref_name(ref: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _kfd_compiler_args(rocm_path: str) -> list[str]:
    clangpp = find_rocm_clangpp(rocm_path)
    if clangpp is None:
        raise RuntimeError(
            f"ROCm clang++ not found under {rocm_path} — KFD build requires a "
            "ROCm-provided C/C++ compiler. Verify ROCK_DIR / --rock-dir points at "
            "a complete ROCm install."
        )
    clangpp_str = str(clangpp)
    clang_str = clangpp_str[:-2] if clangpp_str.endswith("++") else clangpp_str
    return [
        f"-DCMAKE_C_COMPILER={clang_str}",
        f"-DCMAKE_CXX_COMPILER={clangpp_str}",
    ]


def _missing_provisioning_pkg(output: str) -> str | None:
    lowered = output.lower()
    if "required packages were not found" not in lowered and "not found" not in lowered:
        return None
    for pkg in _PROVISIONING_PKGS:
        if re.search(rf"package '{re.escape(pkg)}'.*not found", lowered) or f"- {pkg}" in lowered:
            return pkg
    return None
