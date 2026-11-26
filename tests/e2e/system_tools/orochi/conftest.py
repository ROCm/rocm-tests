# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone/build fixtures for tests/e2e/system_tools/orochi/.

Orochi (https://github.com/GPUOpen-LibrariesAndSDKs/Orochi, MIT licensed) loads
the HIP and CUDA driver/runtime APIs dynamically, letting an application pick
HIP or CUDA at runtime instead of compiling separate builds for each. This
module clones and builds Orochi with its bundled premake5 + MSBuild toolchain,
then restages its ``amd_comgr``/``hiprtc`` DLLs from the ROCm install under
test (``rock_dir``) so Orochi's own unit tests validate that build rather than
whatever it picked up while compiling.

Windows-only: Orochi's build here uses premake5 + MSBuild and requires the
Visual Studio 2022 MSBuild component.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess

import pytest

from framework.executors.background_process import _blocking_stream_run

logger = logging.getLogger(__name__)

_OROCHI_URL = "https://github.com/GPUOpen-LibrariesAndSDKs/Orochi.git"
_OROCHI_REF = os.environ.get("ROCM_TEST_OROCHI_REF", "main")


def _find_vs2022_msbuild() -> pathlib.Path:
    """Find the MSBuild executable belonging to an installed Visual Studio 2022."""
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")
    vswhere = pathlib.Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        located = shutil.which("vswhere")
        if located:
            vswhere = pathlib.Path(located)
    if not vswhere.is_file():
        pytest.fail(f"Orochi requires Visual Studio 2022, but vswhere.exe was not found at {vswhere}")

    result = subprocess.run(
        [
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-version",
            "[17.0,18.0)",
            "-requires",
            "Microsoft.Component.MSBuild",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"Failed to query Visual Studio 2022 with vswhere:\n{result.stderr}")

    installation_path = result.stdout.strip()
    if not installation_path:
        pytest.fail("Orochi requires Visual Studio 2022 with the MSBuild component installed")

    msbuild = pathlib.Path(installation_path) / "MSBuild" / "Current" / "Bin" / "MSBuild.exe"
    if not msbuild.is_file():
        pytest.fail(f"Visual Studio 2022 was found, but MSBuild is missing at {msbuild}")
    return msbuild


def _run_build_step(cmd: list[str], *, cwd: pathlib.Path, timeout: float, label: str) -> None:
    """Run one Orochi build step with the framework's Windows-safe stream runner."""
    result = _blocking_stream_run(
        command=subprocess.list2cmdline(cmd),
        env=os.environ.copy(),
        cwd=str(cwd),
        timeout=timeout,
        stream_stdout=True,
        stream_stderr=True,
        log_path=None,
    )
    if not result.ok:
        raise RuntimeError(
            f"Orochi {label} failed (exit={result.exit_code}):\n"
            f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
        )


def _stage_rocm_runtime_dlls(bin_dir: pathlib.Path, rock_dir: str) -> None:
    """Replace Orochi's own amd_comgr/hiprtc DLLs with the ones from rock_dir.

    Orochi loads these libraries dynamically at runtime, so its unit tests only
    validate the ROCm build under test once these files are staged here instead
    of whatever Orochi's own build produced or bundled.
    """
    rocm_bin = pathlib.Path(rock_dir) / "bin"
    amd_comgr_dlls = sorted(dll for dll in rocm_bin.glob("amd_comgr*.dll") if "amd_comgr_" not in dll.name)
    hiprtc_dlls = sorted(rocm_bin.glob("hiprtc*.dll"))
    if not amd_comgr_dlls:
        pytest.fail(f"Orochi: no amd_comgr DLLs (excluding amd_comgr_*.dll) found under {rocm_bin}")
    if not hiprtc_dlls:
        pytest.fail(f"Orochi: no hiprtc DLLs found under {rocm_bin}")

    bin_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("amd_comgr*.dll", "hiprtc*.dll"):
        for stale in bin_dir.glob(pattern):
            stale.unlink()

    for dll in amd_comgr_dlls + hiprtc_dlls:
        shutil.copy2(dll, bin_dir / dll.name)
        logger.info("Orochi: staged %s -> %s", dll.name, bin_dir)


@pytest.fixture(scope="session")
def orochi_scripts_dir(
    request: pytest.FixtureRequest,
    compiler_build_dir: str,
    framework_config,
    external_build,
    rock_dir: str,
    cmake_executor,
) -> str:
    """Clone, build, and prepare Orochi; return its ``scripts`` directory.

    Returns:
        Path to the cloned repo's ``scripts`` directory, which holds the
        ``unittest.bat`` runner the test invokes.
    """
    if request.config.getoption("--no-gpu", default=False):
        pytest.skip("Orochi unit tests require a real GPU; they cannot run in --no-gpu mode.")
    if cmake_executor is not None:
        pytest.skip("Orochi requires a local Windows build; remote build execution is not supported.")
    if not rock_dir:
        pytest.fail("Orochi requires --rock-dir / ROCK_DIR to stage the ROCm runtime DLLs.")

    build_timeout = float(framework_config.therock.build_timeout_secs)
    dest = pathlib.Path(compiler_build_dir) / "orochi" / f"orochi-{_OROCHI_REF}"
    msbuild = _find_vs2022_msbuild()

    with external_build.build_lock("system-tools-orochi", timeout=build_timeout):
        repo_dir = pathlib.Path(
            external_build.clone_repo(url=_OROCHI_URL, dest=dest, ref=_OROCHI_REF, timeout=build_timeout)
        ).resolve()
        external_build.assert_license_present(repo_dir)

        premake_exe = repo_dir / "tools" / "premake5" / "win" / "premake5.exe"
        assert premake_exe.is_file(), f"Orochi: bundled premake5 not found at {premake_exe}"
        _run_build_step(
            [str(premake_exe), "vs2022"],
            cwd=repo_dir,
            timeout=120.0,
            label="premake5 solution generation",
        )

        sln_path = repo_dir / "YamatanoOrochi.sln"
        assert sln_path.is_file(), f"Orochi: premake5 did not generate {sln_path}"
        _run_build_step(
            [str(msbuild), sln_path.name, "/p:Configuration=Release", "/p:Platform=x64"],
            cwd=repo_dir,
            timeout=build_timeout,
            label="MSBuild",
        )

        bin_dir = repo_dir / "dist" / "bin" / "Release"
        assert bin_dir.is_dir(), f"Orochi: expected build output directory missing: {bin_dir}"
        _stage_rocm_runtime_dlls(bin_dir, rock_dir)

        return str(repo_dir / "scripts")
