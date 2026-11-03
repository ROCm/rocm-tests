# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS installation and runtime-readiness helpers.

Shared across all RVS module test suites. Locates a pre-installed RVS binary,
builds from source when none is present, and exposes the top-level
``ensure_rvs_runtime`` entry point that every suite calls from its session fixture.

Source: https://github.com/ROCm/ROCmValidationSuite
"""

from __future__ import annotations

import logging
import os
import pathlib
import shlex

import pytest

from framework.executors.local_executor import run_cmd_get_stdout_stderr

logger = logging.getLogger(__name__)

_RVS_REPO_URL = "https://github.com/ROCm/ROCmValidationSuite.git"
_RVS_REF = os.environ.get("ROCM_TEST_RVS_REF", "master")


def find_rvs_binary(rock_dir: str, executor=None) -> str:
    """Locate a pre-installed ``rvs`` under the ROCm prefix and ``extras-N`` dirs.

    Both layouts are searched: packages may install into an ``extras-N``
    subdirectory rather than the ROCm prefix, so looking only in ``bin/rvs``
    would miss a packaged RVS. The config tree must sit beside the binary —
    an rvs without its conf directory cannot run any action.

    Args:
        rock_dir: ROCm install root (``--rock-dir`` / ``ROCK_DIR``).
        executor:  Remote executor; ``None`` means local.

    Returns:
        Absolute path to the ``rvs`` binary, or ``""`` when not found.
    """
    if executor is None:
        for base in (pathlib.Path(rock_dir), *sorted(pathlib.Path(rock_dir).glob("extras*"))):
            rvs = base / "bin" / "rvs"
            conf = base / "share" / "rocm-validation-suite" / "conf"
            if rvs.is_file() and os.access(rvs, os.X_OK) and conf.is_dir():
                return str(rvs)
        return ""

    probe = (
        f"for base in {shlex.quote(rock_dir)} {shlex.quote(rock_dir)}/extras*; do "
        'test -x "$base/bin/rvs" && test -d "$base/share/rocm-validation-suite/conf" '
        '&& { echo "$base/bin/rvs"; break; }; done'
    )
    found = (executor.run(probe).stdout or "").strip()
    return found.splitlines()[0].strip() if found else ""


def build_rvs_from_source(
    rock_dir: str,
    external_build,
    compiler_build_dir: str,
    cmake_build_dir_fn,
    built_binary_fn,
    executor=None,
    version: str = _RVS_REF,
) -> str:
    """Clone ROCmValidationSuite and build it via CMake.

    Called only when no pre-installed binary is found. Clones the public
    ROCmValidationSuite repository, initialises submodules, and builds using
    the framework's ``cmake_build_dir`` fixture factory.

    Args:
        rock_dir:            ROCm install root.
        external_build:      Framework external-build helper.
        compiler_build_dir:  Output directory for compiled artefacts.
        cmake_build_dir_fn:  ``cmake_build_dir`` fixture callable.
        built_binary_fn:     ``built_binary`` fixture callable.
        executor:            Remote executor; ``None`` means local.
        version:             Git ref to check out (default ``ROCM_TEST_RVS_REF`` env var or ``master``).

    Returns:
        Absolute path to the built ``rvs`` binary.
    """
    dest = pathlib.Path(compiler_build_dir) / "rvs" / "ROCmValidationSuite"
    src_dir = external_build.clone_repo(_RVS_REPO_URL, dest, ref=version)
    external_build.assert_license_present(src_dir)

    if executor is not None:
        executor.run(
            f"cd {shlex.quote(str(src_dir))} && git submodule update --init --recursive",
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

    install_dir = pathlib.Path(str(src_dir)) / "install"

    # Check for a prior build before invoking cmake.
    prior = _find_first(install_dir, "bin/rvs", executor)
    if prior is not None:
        logger.info("RVS: using previously built binary: %s", prior)
        return str(prior)

    logger.info("RVS: building from source at %s", src_dir)
    build_dir = cmake_build_dir_fn(
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

    if executor is not None:
        executor.run(f"mkdir -p {shlex.quote(str(install_dir))}")
    else:
        install_dir.mkdir(parents=True, exist_ok=True)

    if executor is not None:
        result = executor.run(
            f"DESTDIR={shlex.quote(str(install_dir))} cmake --install {shlex.quote(str(build_dir))}",
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

    rvs_bin = _find_first(install_dir, "bin/rvs", executor)
    if rvs_bin is None:
        pytest.fail(f"RVS binary not found under {install_dir} after install.")

    logger.info("RVS binary installed at: %s", rvs_bin)
    return str(built_binary_fn(str(rvs_bin), "rvs"))


def ensure_rvs_runtime(
    rock_dir: str,
    external_build,
    compiler_build_dir: str,
    cmake_build_dir_fn,
    built_binary_fn,
    executor=None,
    version: str = _RVS_REF,
) -> str:
    """Return the path to a ready ``rvs`` binary, building from source if needed.

    Priority:
      1. Pre-installed at ``{rock_dir}/bin/rvs`` or ``{rock_dir}/extras*/bin/rvs``
      2. Previously built under the compiler build dir
      3. Clone and build from source

    Calls ``pytest.skip`` when neither a pre-installed nor buildable RVS is available.

    Args:
        rock_dir:            ROCm install root.
        external_build:      Framework external-build helper.
        compiler_build_dir:  Output directory for compiled artefacts.
        cmake_build_dir_fn:  ``cmake_build_dir`` fixture callable.
        built_binary_fn:     ``built_binary`` fixture callable.
        executor:            Remote executor; ``None`` means local.
        version:             Git ref for source build.

    Returns:
        Absolute path to the ``rvs`` binary.
    """
    preinstalled = find_rvs_binary(rock_dir, executor)
    if preinstalled:
        logger.info("RVS: using pre-installed binary: %s", preinstalled)
        return preinstalled

    logger.info("RVS: no pre-installed binary found; building from source")
    return build_rvs_from_source(
        rock_dir=rock_dir,
        external_build=external_build,
        compiler_build_dir=compiler_build_dir,
        cmake_build_dir_fn=cmake_build_dir_fn,
        built_binary_fn=built_binary_fn,
        executor=executor,
        version=version,
    )


# ---------------------------------------------------------------------------
# Internal helpers (not part of the public surface)
# ---------------------------------------------------------------------------


def _find_first(root: pathlib.Path, pattern: str, executor=None) -> pathlib.Path | None:
    """Return the first entry under *root* matching *pattern*, local or remote."""
    if executor is None:
        if not root.exists():
            return None
        return next(
            (c for c in root.rglob(pattern) if c.is_file()),
            None,
        )
    probe = f"find {shlex.quote(str(root))} -path {shlex.quote('*/' + pattern)} " "-type f 2>/dev/null | head -n 1"
    found = (executor.run(probe).stdout or "").strip()
    return pathlib.Path(found) if found else None
