# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone/build fixtures for tests/e2e/hpc/kokkos/.

Kokkos (https://github.com/kokkos/kokkos) is a large third-party performance-
portability HPC library. It is a full CMake project, so it uses the framework's
remote-transparent external-build primitives rather than the single-file
``compile_binary``/``hipcc`` path:

    * ``external_build.clone_repo``             -- idempotent git clone (local/remote)
    * ``external_build.assert_license_present`` -- provenance guard
    * a bespoke ``cmake configure/build/install`` runner (streaming Popen locally,
      ``cmake_executor`` on a remote build node) with the ROCm build environment
      injected via an ``env VAR=... cmd`` prefix rather than by mutating
      ``os.environ``.

The feature under test is HIP Relocatable Device Code (RDC, the ``-fgpu-rdc``
compiler flag). The build enables ``Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON``
and runs a VERBOSE build so the emitted device-compile command lines can be
inspected: the presence of ``-fgpu-rdc`` in that output is the authoritative
proof that RDC compilation actually happened. The Kokkos ctest suite is then run
by ``target_executor`` on the GPU node in ``test_gpu_rdc_kokkos.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import pathlib
import re

import pytest

from framework.builder.binary_builder import find_rocm_clangpp, resolve_parallel_jobs
from framework.executors.background_process import _blocking_stream_run
from tests.e2e.hpc.kokkos._workload import KOKKOS_REF, KOKKOS_URL, kokkos_arch_flag

logger = logging.getLogger(__name__)

# Written by a successful cmake configure — used for build idempotency (QUDA-style).
_BUILD_SENTINEL = "CTestTestfile.cmake"
# Durable record of the -fgpu-rdc scan result, so a cached (skipped) build can
# report the RDC verdict without re-running the multi-minute VERBOSE build.
_RDC_FLAG_FILE = "rdc_flag.txt"
# The compiler flag whose presence in the VERBOSE build output proves RDC.
_RDC_FLAG = "-fgpu-rdc"


@dataclass(frozen=True)
class KokkosRdcBuild:
    """Result of the session-scoped Kokkos HIP RDC build.

    Attributes:
        build_dir:        CMake build directory; the ctest suite runs here.
        install_dir:      CMake install prefix (``cmake --install`` target).
        rdc_flag_present: True when ``-fgpu-rdc`` was found in the VERBOSE build
                          output — the authoritative RDC compilation gate.
        build_log_path:   Path to the persisted VERBOSE build log for debugging.
    """

    build_dir: str
    install_dir: str
    rdc_flag_present: bool
    build_log_path: str


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _path_exists(path: str, cmake_executor) -> bool:
    """Return True when *path* is a file, transparently for local/remote nodes."""
    if cmake_executor is not None:
        return cmake_executor.run(f"test -f {path}", timeout=30.0).ok
    return os.path.isfile(path)


def _read_text(path: str, cmake_executor) -> str:
    """Read a small text file, transparently for local/remote nodes."""
    if cmake_executor is not None:
        return cmake_executor.run(f"cat {path} 2>/dev/null", timeout=30.0).stdout or ""
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return ""


def _write_text(path: str, value: str, cmake_executor) -> None:
    """Write a small text file, transparently for local/remote nodes."""
    if cmake_executor is not None:
        cmake_executor.run(f"printf %s {value} > {path}", timeout=30.0)
        return
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(value)


def _resolve_compilers(rock_dir: str, cmake_executor) -> str:
    """Return the ROCm ``hipcc`` path used as the Kokkos CXX/HIP compiler.

    Locally we confirm the TheRock/ROCm layout via ``find_rocm_clangpp`` (only to
    validate the install is complete); the Kokkos build itself is driven through
    ``hipcc`` exactly as the original spec does. On a remote build node the ROCm
    tree lives on the far host, so we use the flattened ``<rock_dir>/bin/hipcc``.
    """
    if cmake_executor is None and find_rocm_clangpp(rock_dir) is None:
        pytest.fail(
            f"ROCm clang toolchain not found under {rock_dir} — cannot build Kokkos. "
            "Verify ROCK_DIR / --rock-dir points to a complete ROCm install."
        )
    return f"{rock_dir}/bin/hipcc"


def _run_build_step(cmd: str, *, cmake_executor, timeout: float, label: str, log_path: str) -> str:
    """Run a single cmake build step, streaming output live, and return it.

    Kokkos' VERBOSE configure/build each run for minutes and emit very large
    output, so a buffered ``subprocess.run(capture_output=True)`` is unusable.
    Remote steps stream through the SSH executor's ``stream=True`` path; local
    steps use the framework's shared streaming Popen runner
    (``_blocking_stream_run``), which forwards stdout+stderr to the console in
    real time *and* appends them to *log_path*. Returns the combined
    stdout+stderr so the caller can scan it (e.g. for ``-fgpu-rdc``).

    Live console output still requires ``pytest -s`` (pytest captures fd output
    by default); *log_path* is written either way — ``tail -f`` it to watch a run
    started without ``-s``.
    """
    logger.info("Kokkos %s -> streaming to %s", label, log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"Kokkos {label} failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return (result.stdout or "") + (result.stderr or "")
    result = _blocking_stream_run(
        command=cmd,
        env=os.environ.copy(),
        cwd=None,
        timeout=timeout,
        stream_stdout=True,
        stream_stderr=True,
        log_path=log_path,
    )
    if not result.ok:
        raise RuntimeError(
            f"Kokkos {label} failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )
    return (result.stdout or "") + (result.stderr or "")


@pytest.fixture(scope="session")
def kokkos_rdc_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    node_pool,
    require_gpu_arch_for,
) -> KokkosRdcBuild:
    """Clone, configure (RDC on), VERBOSE-build, and install Kokkos once per session.

    Returns a :class:`KokkosRdcBuild` carrying the ctest build dir and the
    ``-fgpu-rdc`` verdict scanned from the VERBOSE build output. ROCm paths come
    from ``rock_dir``; the GPU arch is resolved from ``--gpu-arch`` (falling back
    to the node pool's first GPU) and mapped to ``-DKokkos_ARCH_AMD_GFX<NNN>=ON``.
    """
    # --- resolve GPU arch (explicit flag > node-pool detection > hard fail) ----
    # An RDC device build must target a concrete GFX arch, so a missing arch is a
    # CI misconfiguration, not an optional resource.
    resolved_gpu_arch = gpu_arch
    if not resolved_gpu_arch and node_pool:
        try:
            resolved_gpu_arch = next(iter(node_pool.gpus)).arch
        except (StopIteration, AttributeError):
            resolved_gpu_arch = None
    if not resolved_gpu_arch:
        require_gpu_arch_for("kokkos")  # raises pytest.fail — no auto-detect for RDC.

    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)

    # --- clone (idempotent) + provenance guard --------------------------------
    ref_label = _safe_ref_name(KOKKOS_REF)
    dest = pathlib.Path(compiler_build_dir) / "kokkos" / f"kokkos-{ref_label}"
    kokkos_dir = str(external_build.clone_repo(url=KOKKOS_URL, dest=dest, ref=KOKKOS_REF, timeout=build_timeout))
    external_build.assert_license_present(kokkos_dir)

    # Absolute path on the build node (the ctest run later cd's via --test-dir).
    if cmake_executor is not None and hasattr(cmake_executor, "workspace_path_for"):
        kokkos_dir = str(cmake_executor.workspace_path_for(kokkos_dir))
    else:
        kokkos_dir = os.path.abspath(kokkos_dir)

    # Namespace the build/install trees by GPU arch: an RDC build bakes a single
    # GFX code object in, so a gfx942 build cannot run on gfx950. Arch-specific
    # dirs keep the idempotency check below correct — switching --gpu-arch forces
    # a fresh build instead of silently reusing a mismatched one.
    arch_label = resolved_gpu_arch or "auto"
    build_dir = f"{kokkos_dir}/kokkos_build-{arch_label}"
    install_dir = f"{kokkos_dir}/kokkos_install-{arch_label}"
    build_log = f"{build_dir}/kokkos-build-verbose.log"
    rdc_flag_path = f"{build_dir}/{_RDC_FLAG_FILE}"

    # --- idempotency: reuse a configured build tree, re-reading the RDC verdict -
    if _path_exists(f"{build_dir}/{_BUILD_SENTINEL}", cmake_executor):
        cached = _read_text(rdc_flag_path, cmake_executor).strip()
        logger.info("Kokkos: existing build tree at %s — skipping configure/build/install", build_dir)
        return KokkosRdcBuild(
            build_dir=build_dir,
            install_dir=install_dir,
            rdc_flag_present=cached == "1",
            build_log_path=build_log,
        )

    # --- build environment (never via os.environ) -----------------------------
    hipcc = _resolve_compilers(rock_dir, cmake_executor)
    env_prefix = (
        f"ROCM_PATH={rocm_path} "
        f"HIP_PATH={rocm_path} "
        f"PATH={rocm_path}/bin:$PATH "
        f"LD_LIBRARY_PATH={rocm_path}/lib:$LD_LIBRARY_PATH"
    )

    # CPU host arch: a single ``-DKokkos_ARCH_NATIVE=ON`` (compile for the build
    # host's own CPU) is used instead of branching Intel-vs-AMD (ZEN2/NATIVE).
    # The CPU arch only tunes host-side Serial-backend SIMD codegen and has zero
    # bearing on whether the HIP device compile emits -fgpu-rdc, so the simpler,
    # vendor-agnostic single flag is preferred (porting rule 13).
    arch_flag = kokkos_arch_flag(resolved_gpu_arch)  # -DKokkos_ARCH_AMD_GFX<NNN>=ON
    configure_flags = (
        "-DKokkos_ARCH_NATIVE=ON "
        f"-DCMAKE_CXX_COMPILER={hipcc} "
        "-DKokkos_ENABLE_HIP=ON "
        "-DKokkos_ENABLE_SERIAL=ON "
        "-DKokkos_ENABLE_TESTS=ON "
        "-DCMAKE_CXX_STANDARD=20 "
        "-DKokkos_ENABLE_BENCHMARKS=on "
        f"-DCMAKE_INSTALL_PREFIX={install_dir} "
        "-DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=ON "
        f"{arch_flag}"
    )

    jobs = resolve_parallel_jobs(remote_executor=cmake_executor)
    log_dir = os.path.join(framework_config.framework.artifact_dir, "kokkos")
    os.makedirs(log_dir, exist_ok=True)

    configure_cmd = f"env {env_prefix} cmake -S {kokkos_dir} -B {build_dir} {configure_flags}"
    # ``--verbose`` forwards VERBOSE=1 to the underlying make so the full device
    # compile lines (including -fgpu-rdc) are emitted for the RDC scan below.
    build_cmd = f"env {env_prefix} cmake --build {build_dir} --parallel {jobs} --verbose"
    install_cmd = f"env {env_prefix} cmake --install {build_dir}"

    logger.info("Kokkos: configuring (gpu_arch=%s, rocm=%s)", resolved_gpu_arch, rocm_path)
    _run_build_step(
        configure_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="cmake configure",
        log_path=os.path.join(log_dir, "build-configure.log"),
    )
    logger.info("Kokkos: VERBOSE build with %s parallel jobs", jobs)
    build_output = _run_build_step(
        build_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="cmake build (VERBOSE)",
        log_path=build_log,
    )

    # Scan the captured VERBOSE build output for the RDC flag. This is the core
    # RDC evidence; persist the verdict so a later cached build can report it.
    rdc_flag_present = _RDC_FLAG in build_output
    _write_text(rdc_flag_path, "1" if rdc_flag_present else "0", cmake_executor)
    logger.info("Kokkos: %s %s in VERBOSE build output", _RDC_FLAG, "found" if rdc_flag_present else "NOT found")

    logger.info("Kokkos: installing to %s", install_dir)
    _run_build_step(
        install_cmd,
        cmake_executor=cmake_executor,
        timeout=build_timeout,
        label="cmake install",
        log_path=os.path.join(log_dir, "build-install.log"),
    )

    return KokkosRdcBuild(
        build_dir=build_dir,
        install_dir=install_dir,
        rdc_flag_present=rdc_flag_present,
        build_log_path=build_log,
    )
