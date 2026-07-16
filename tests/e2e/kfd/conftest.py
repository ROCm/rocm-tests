# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- Build fixture for tests/e2e/kfd/ (KFD / kfdtest suite).

``kfdtest`` is the GTest suite for the Kernel Fusion Driver (KFD) — the AMD GPU
kernel driver / thunk layer (libhsakmt) at the base of the ROCm stack. Upstream
it lives in the ROCm/rocm-systems monorepo under
``projects/rocr-runtime/libhsakmt/tests/kfdtest``.

    1. Sparse-clone the ``libhsakmt`` project subtree from ROCm/rocm-systems.
       The whole project (not just ``tests/kfdtest``) is checked out because
       kfdtest's CMakeLists.txt resolves the thunk headers via the relative path
       ``${PROJECT_SOURCE_DIR}/../../include`` -> ``libhsakmt/include`` (which
       carries ``hsakmt/hsakmt.h`` and ``hsakmt/linux/kfd_ioctl.h``). A narrower
       checkout of only ``tests/kfdtest`` omits that sibling ``include`` tree and
       the build fails with ``'hsakmt/hsakmt.h' file not found``.
    2. Configure + build it with CMake against the resolved TheRock/ROCm install,
       replicating the original test's CMake arguments:
         -DCMAKE_C_COMPILER   = <rocm>/.../clang (amdclang)
         -DCMAKE_CXX_COMPILER = <rocm>/.../clang++ (amdclang++)
         -DCMAKE_PREFIX_PATH  = <rocm>            (added automatically by cmake_build)
         -DROCM_PATH          = <rocm>            (added automatically by cmake_build)
         -DCMAKE_EXE_LINKER_FLAGS = -ldl
         -DROCM_DIR           = <rocm>
         -DOPENCL_DIR         = <rocm>

Provisioning assumptions (see the module docstring in test_kfd.py for the full
list): the target node must have a loaded ``amdgpu`` kernel module and the KFD
character device (``/dev/kfd``). OS build/runtime dependencies (libdrm, libnuma)
are expected to be provisioned on the fleet via ``--pre-install`` rather than
installed from inside the test (the original test installed these itself, which
is host-provisioning and out of scope for a framework test).
"""

from __future__ import annotations

import logging
import os
import pathlib
import re

import pytest

from framework.builder.binary_builder import cmake_build, find_rocm_clangpp

logger = logging.getLogger(__name__)

# kfdtest lives inside the ROCm/rocm-systems monorepo. Canonical source:
# https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime/libhsakmt/tests/kfdtest
_KFDTEST_MONOREPO_URL = "https://github.com/ROCm/rocm-systems.git"
# Sparse-checkout the whole libhsakmt project (not just tests/kfdtest): kfdtest's
# CMakeLists.txt pulls the thunk headers from the sibling include/ dir via the
# relative path ../../include, so that tree must be present alongside the tests.
_LIBHSAKMT_SUBPATH = "projects/rocr-runtime/libhsakmt"
# kfdtest itself lives under tests/kfdtest within the libhsakmt project.
_KFDTEST_BUILD_SUBDIR = "tests/kfdtest"
# Track the live rocm-systems source by default so CI follows the latest KFD
# thunk headers; pin a known-good ref with KFDTEST_REF when needed.
_DEFAULT_KFDTEST_REF = "develop"
_KFDTEST_REF = os.environ.get("KFDTEST_REF", _DEFAULT_KFDTEST_REF)


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


# pkg-config deps kfdtest's CMakeLists.txt requires (pkg_check_modules). When the
# corresponding -dev package is not provisioned on the build node, cmake configure
# aborts before any compilation. These are host-provisioning deps (see module
# docstring), so a missing one is an un-provisioned environment, not a test failure.
_PROVISIONING_PKGS = ("libdrm", "libnuma")


def _missing_provisioning_pkg(output: str) -> str | None:
    """Return the name of a missing pkg-config provisioning dep in *output*, if any.

    Detects the pkg_check_modules "required packages were not found" signature so a
    missing host dep can be turned into an actionable skip rather than a hard ERROR.
    """
    lowered = output.lower()
    if "required packages were not found" not in lowered and "not found" not in lowered:
        return None
    for pkg in _PROVISIONING_PKGS:
        if re.search(rf"package '{re.escape(pkg)}'.*not found", lowered) or f"- {pkg}" in lowered:
            return pkg
    return None


def _kfd_compiler_args(rocm_path: str) -> list[str]:
    """Return ``-DCMAKE_{C,CXX}_COMPILER`` args pointing at ROCm's clang/clang++.

    The original test hard-coded ``<rocm>/llvm/bin/amdclang`` and
    ``amdclang++``. ``find_rocm_clangpp`` performs the same TheRock/ROCm probe
    used by every other CMake-based conftest; the C compiler is derived from the
    resolved C++ compiler by stripping the trailing ``++`` (clang++ -> clang,
    amdclang++ -> amdclang).
    """
    clangpp = find_rocm_clangpp(rocm_path)
    if clangpp is None:
        pytest.skip(
            f"ROCm clang++ not found under {rocm_path} — cannot build kfdtest. "
            "Verify ROCK_DIR / --rock-dir points at a complete ROCm install."
        )
    clangpp_str = str(clangpp)
    clang_str = clangpp_str[:-2] if clangpp_str.endswith("++") else clangpp_str
    return [
        f"-DCMAKE_C_COMPILER={clang_str}",
        f"-DCMAKE_CXX_COMPILER={clangpp_str}",
    ]


# Thunk headers kfdtest #includes as "hsakmt/hsakmt.h" and "hsakmt/linux/kfd_ioctl.h";
# both live under libhsakmt/include/ in the checkout. Their absence means the sibling
# include/ tree was not checked out alongside tests/kfdtest.
_THUNK_HEADERS = ("hsakmt/hsakmt.h", "hsakmt/linux/kfd_ioctl.h")


def _assert_thunk_headers_present(include_dir: pathlib.Path, cmake_executor) -> None:
    """Fail fast (with guidance) if the libhsakmt thunk headers are missing on the build node.

    kfdtest compiles ``#include "hsakmt/hsakmt.h"`` against the sibling
    ``libhsakmt/include`` tree. When only ``tests/kfdtest`` was checked out (e.g. a
    stale sparse checkout from an earlier run), that tree is absent and every
    translation unit fails with ``'hsakmt/hsakmt.h' file not found``. Turn that
    late, noisy compiler error into an actionable message here.
    """
    missing: list[str] = []
    for header in _THUNK_HEADERS:
        header_path = include_dir / header
        if cmake_executor is not None:
            present = cmake_executor.run(f"test -f {header_path}", timeout=30.0).ok
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


@pytest.fixture(scope="session")
def kfdtest_binary(
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    gpu_arch: str | None,
    built_binary,
) -> str:
    """Sparse-clone and CMake-build ``kfdtest``; return the absolute binary path.

    Session-scoped so the clone + build happens once per session regardless of
    how many test functions request it.
    """
    rocm_path = os.path.realpath(rock_dir)

    # clone_repo(sparse_subtree=...) returns the subtree path directly. We check
    # out the whole libhsakmt project so kfdtest's sibling include/ tree is present
    # (its CMakeLists.txt references ../../include for hsakmt/hsakmt.h).
    libhsakmt_src = external_build.clone_repo(
        url=_KFDTEST_MONOREPO_URL,
        dest=pathlib.Path(compiler_build_dir) / "kfd" / f"rocm-systems-{_safe_ref_name(_KFDTEST_REF)}",
        ref=_KFDTEST_REF,
        timeout=float(framework_config.therock.build_timeout_secs),
        sparse_subtree=_LIBHSAKMT_SUBPATH,
    )
    external_build.assert_license_present(libhsakmt_src)  # open-source provenance guard

    kfdtest_src = pathlib.Path(libhsakmt_src) / _KFDTEST_BUILD_SUBDIR
    build_dir = pathlib.Path(kfdtest_src) / "build"
    _assert_thunk_headers_present(pathlib.Path(libhsakmt_src) / "include", cmake_executor)

    # Source is already on the build node (cloned above), so no sync_dirs are
    # needed; cmake_build handles local vs. remote (cmake_executor) transparently.
    try:
        actual_build_dir = cmake_build(
            src=str(kfdtest_src),
            build_dir=str(build_dir),
            rocm_path=rocm_path,
            gpu_arch=gpu_arch,
            compiler_args=_kfd_compiler_args(rocm_path),
            extra_cmake_args=[
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_EXE_LINKER_FLAGS=-ldl",
                f"-DROCM_DIR={rocm_path}",
                f"-DOPENCL_DIR={rocm_path}",
            ],
            label="kfdtest",
            remote_executor=cmake_executor,
        )
    except (AssertionError, RuntimeError) as exc:
        # A missing pkg-config provisioning dep (libdrm/libnuma) is an
        # un-provisioned build host, not a broken build — skip with guidance
        # rather than erroring. Any other failure is real breakage; re-raise.
        missing = _missing_provisioning_pkg(str(exc))
        if missing is None:
            raise
        pytest.skip(
            f"kfdtest build dependency '{missing}' is not provisioned on the build node — "
            f"cmake configure could not find its pkg-config module. Provision it via "
            f"'--pre-install pkg={missing}-dev' (see conftest.py module docstring)."
        )
    return built_binary(os.path.join(str(actual_build_dir), "kfdtest"), "kfdtest")
