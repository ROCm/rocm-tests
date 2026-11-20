# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Clone/build fixtures for tests/e2e/hpc/lammps_kokkos/.

LAMMPS (https://github.com/lammps/lammps) is a large third-party molecular
dynamics code.  The Kokkos/HIP configuration built here links against an MPI
runtime, hipFFT and rocThrust, so it cannot use the single-file
``compile_binary``/``hipcc`` path.  This module therefore uses the framework's
remote-transparent external-build primitives:

    * ``external_build.clone_repo``             -- idempotent git clone (local/remote)
    * ``external_build.assert_license_present``  -- provenance guard
    * ``external_build.detect_mpi_runtime`` /
      ``external_build.provision_openmpi_runtime`` -- MPI discovery / bootstrap
    * a bespoke cmake configure/build runner (streaming subprocess locally, or the
      ``cmake_executor`` on the remote build node) with the MPI + ROCm build
      environment injected via an ``env VAR=... cmd`` prefix rather than by
      mutating ``os.environ``.

The resulting ``lmp`` binary is executed by ``target_executor`` on the GPU node
in ``tests/e2e/hpc/lammps_kokkos/test_lammps_kokkos.py``.
"""

from __future__ import annotations

import logging
import os
import pathlib
import re
import shlex

import pytest

from framework.builder.binary_builder import MpiRuntime, build_artifact_exists, resolve_parallel_jobs
from framework.executors.background_process import _blocking_stream_run
from tests.common.gpu_pci_map import GPU_DEVICE_MAP, detect_device_revision

logger = logging.getLogger(__name__)

# Upstream LAMMPS molecular dynamics code.
_LAMMPS_URL = "https://github.com/lammps/lammps"
# Pinned so scores stay comparable across runs. Override with LAMMPS_REF to
# track a different tag/branch/commit.
_DEFAULT_LAMMPS_REF = "patch_27Jun2024"
_LAMMPS_REF = os.environ.get("LAMMPS_REF", _DEFAULT_LAMMPS_REF)

# cmake writes the LAMMPS executable here on success -- used for idempotency.
_BUILD_SENTINEL = "lmp"

# The MPI stack is always built rather than taken from the node: a system MPI
# varies by distro image (4.1 on some, 5.0 on others) and the transport it picks
# moves the multi-GPU numbers, so scores would not be comparable across
# machines. Both components are pinned for the same reason.
# Override with ROCM_TEST_LAMMPS_OPENMPI_VERSION / ROCM_TEST_LAMMPS_UCX_VERSION.
_OPENMPI_VERSION = "5.0.10"
_UCX_VERSION = "v1.18.0"
_UCX_URL = "https://github.com/openucx/ucx"

# UCX carries the ROCm transport that makes mpirun move data between ranks over
# the GPU fabric; without it OpenMPI falls back to staging through host memory,
# which changes the multi-GPU scores.
# The ROCm transport module specifically, not just any UCX artifact: UCX's
# configure accepts a --with-rocm prefix that holds no ROCm and silently drops
# the transport, which would quietly move every multi-GPU score onto the host
# memory path instead of failing.
_UCX_SENTINEL = "install/lib/ucx/libuct_rocm.so"
_MPI_SENTINEL = "install/bin/mpirun"

# SLES ships a gcc too old to build UCX/OpenMPI, so the newer series is selected
# there, matching what the distro provides as gcc-13.
_SLES_PROBE = "grep -qiE '^ID=.*sles|^ID_LIKE=.*suse' /etc/os-release"

# Kokkos needs the hwloc headers, not just the CLI tool, and Debian splits those
# into libhwloc-dev while the RPM distros ship them in hwloc-devel. The headers
# land in a multiarch subdirectory on Debian, hence the glob. The SLES arm needs
# --no-gpg-checks because the ROCm CI images carry repos with untrusted keys.
_HWLOC_PROBE = "ls /usr/include/hwloc.h /usr/include/*/hwloc.h 2>/dev/null | grep -q ."
_HWLOC_INSTALL = (
    'SUDO=""; if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO="sudo -n"; fi; '
    "if command -v apt-get >/dev/null 2>&1; then "
    "  $SUDO apt-get install -y libhwloc-dev; "
    "elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then "
    "  DNF=$(command -v dnf || command -v yum); $SUDO $DNF install -y hwloc-devel; "
    "elif command -v zypper >/dev/null 2>&1; then "
    "  $SUDO zypper --non-interactive install --no-gpg-checks hwloc-devel; "
    "else "
    '  echo "no supported package manager (apt-get/dnf/yum/zypper)" >&2; exit 1; '
    "fi"
)
# Bounds a wedged package mirror rather than budgeting for the few MB fetched.
_HWLOC_TIMEOUT = 600.0

# LAMMPS bundles fmt 10.2.0, whose FMT_STRING macro wraps its argument in a
# consteval lambda. Clang 17 and newer implement P2564, which promotes that
# lambda to an immediate function and so rejects every FMT_STRING use from a
# runtime context -- the whole src/ tree fails to compile under -std=c++20.
# fmt declares the macro behind #ifndef specifically so it can be overridden,
# and defining it empty restores the pre-C++20 constexpr path. Drop this once
# the pinned ref ships fmt 11 or newer.
_FMT_CONSTEVAL_FLAG = "-DFMT_CONSTEVAL="

# APU parts whose CPU and GPU share one physical memory pool. Kokkos only
# migrates pages into that pool when XNACK is on, so these need HSA_XNACK=1.
# The discrete parts report the same gfx942 arch as MI300A, so the board has to
# be identified by PCI device id rather than by gpu_arch.
_XNACK_PARTS = frozenset({"MI300A"})


def _safe_ref_name(ref: str) -> str:
    """Return a filesystem-safe label for a git ref."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", ref).strip("_") or "default"


def _run_probe(cmd: str, cmake_executor) -> bool:
    """Run a shell probe on the build node, transparently for local/remote."""
    if cmake_executor is not None:
        return cmake_executor.run(cmd, timeout=60.0).ok
    return _blocking_stream_run(
        command=cmd,
        env=os.environ.copy(),
        cwd=None,
        timeout=60.0,
        stream_stdout=False,
        stream_stderr=False,
        log_path=os.devnull,
    ).ok


def _abs_path(path: str, cmake_executor) -> str:
    """Return an absolute path for *path*, transparently for local/remote nodes.

    Remote paths are resolved on the node itself rather than against this
    host's filesystem, and relative to the same working directory the clone
    ran from, so the two agree on what a bare ``output/...`` path means.
    """
    if cmake_executor is not None:
        result = cmake_executor.run(f"readlink -f {shlex.quote(path)}", timeout=30.0)
        resolved = result.stdout.strip()
        if not resolved:
            raise RuntimeError(f"LAMMPS: could not resolve {path!r} on the build node")
        return resolved
    return str(pathlib.Path(path).resolve())


def _host_compilers(cmake_executor) -> list[str]:
    """Return ``CC=``/``CXX=`` configure arguments for the build node."""
    if _run_probe(_SLES_PROBE, cmake_executor):
        return ["CC=gcc-13", "CXX=g++-13"]
    return ["CC=gcc", "CXX=g++"]


def _ensure_hwloc(cmake_executor) -> bool:
    """Return True when hwloc headers are available, installing them if needed.

    Kokkos only uses hwloc for CPU-side thread and rank pinning, so a node
    without it still produces a valid run -- the caller turns the cmake option
    off rather than failing, and the downgrade is logged because it does affect
    the numbers slightly.
    """
    if _run_probe(_HWLOC_PROBE, cmake_executor):
        return True
    if cmake_executor is not None:
        cmake_executor.run(_HWLOC_INSTALL, timeout=_HWLOC_TIMEOUT)
    else:
        _blocking_stream_run(
            command=_HWLOC_INSTALL,
            env=os.environ.copy(),
            cwd=None,
            timeout=_HWLOC_TIMEOUT,
            stream_stdout=False,
            stream_stderr=False,
            log_path=os.devnull,
        )
    return _run_probe(_HWLOC_PROBE, cmake_executor)


def _run_build_step(cmd: str, *, cmake_executor, timeout: float, label: str, log_path: str) -> None:
    """Run a single cmake build step, streaming output live and to *log_path*.

    A full Kokkos/HIP LAMMPS compile runs for tens of minutes, so a buffered
    ``subprocess.run(capture_output=True)`` (which shows nothing until it exits)
    is unusable here.  Remote steps stream through the SSH executor's own
    ``stream=True`` path; local steps use the framework's shared streaming Popen
    runner, which forwards stdout+stderr to the console in real time *and*
    appends them to *log_path*.

    Live console output still requires ``pytest -s``; the *log_path* file is
    written either way -- ``tail -f`` it to watch a run started without ``-s``.

    ``cmake_build`` is not usable here: its local path buffers via
    ``subprocess.run(capture_output=True)`` and injects only ``ROCM_PATH``, not
    the MPI toolchain this build needs. The framework's equivalent streaming
    runner is private (``binary_builder._run_stream_step``), so the hpc suites
    (quda, rochpl, and this one) each carry a local copy.
    """
    logger.info("LAMMPS %s -> streaming to %s", label, log_path)
    if cmake_executor is not None:
        result = cmake_executor.run(cmd, timeout=timeout, stream=True)
        if not result.ok:
            raise RuntimeError(
                f"LAMMPS {label} failed on remote (exit={result.exit_code}):\n"
                f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
            )
        return
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
            f"LAMMPS {label} failed locally (exit={result.exit_code}). Full log: {log_path}\n"
            f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
        )


@pytest.fixture
def lammps_xnack_env(target_executor) -> str:
    """Return ``"HSA_XNACK=1 "`` when the run node is an APU, else ``""``.

    Detection follows ``target_executor`` to the node that will run the
    workload, so a remote-node run keys off the GPU under test rather than the
    one in the machine driving pytest. An undetectable GPU yields ``""``, which
    leaves XNACK at the driver default.
    """
    key = detect_device_revision(cmake_executor=target_executor)
    part = GPU_DEVICE_MAP.get(key, "")
    if part not in _XNACK_PARTS:
        return ""
    logger.info("LAMMPS: %s detected (device %s) -- enabling HSA_XNACK", part, key)
    return "HSA_XNACK=1 "


def _openmpi_version() -> str:
    """Return the OpenMPI version to build, honouring the env overrides."""
    return os.environ.get("ROCM_TEST_LAMMPS_OPENMPI_VERSION") or os.environ.get("OPENMPI_VERSION") or _OPENMPI_VERSION


def _mpi_runtime_env(prefix: str) -> dict[str, str]:
    """Return the include/library env for an OpenMPI install at *prefix*."""
    include_dir = f"{prefix}/include"
    return {
        "MPI_HOME": prefix,
        "C_INCLUDE_PATH": include_dir,
        "CPLUS_INCLUDE_PATH": include_dir,
        "LD_LIBRARY_PATH": f"{prefix}/lib:{prefix}/lib64",
    }


@pytest.fixture(scope="session")
def lammps_ucx(rock_dir, compiler_build_dir, framework_config, external_build, cmake_executor) -> str:
    """Build UCX with the ROCm transport enabled; return its install prefix.

    OpenMPI is configured against this, which is what lets ranks exchange GPU
    buffers over the device fabric instead of bouncing through host memory.
    """
    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)
    version = os.environ.get("ROCM_TEST_LAMMPS_UCX_VERSION") or _UCX_VERSION
    log_dir = os.path.join(framework_config.framework.artifact_dir, "lammps_kokkos")
    os.makedirs(log_dir, exist_ok=True)

    dest = pathlib.Path(compiler_build_dir) / "lammps_kokkos" / f"ucx-{_safe_ref_name(version)}"
    with external_build.build_lock(f"hpc-lammps-kokkos-ucx-{_safe_ref_name(version)}", timeout=build_timeout):
        source_dir = _abs_path(
            str(external_build.clone_repo(url=_UCX_URL, dest=dest, ref=version, timeout=build_timeout)),
            cmake_executor,
        )
        external_build.assert_license_present(source_dir)
        build_dir = f"{source_dir}/build"
        prefix = f"{build_dir}/install"

        logger.info("LAMMPS: building UCX %s (rocm=%s)", version, rocm_path)
        external_build.autotools_make(
            source_dir,
            build_dir,
            bootstrap_script="./autogen.sh",
            # UCX keeps its tuned release configure under contrib/, not at the root.
            configure_script="../contrib/configure-opt",
            configure_args=[
                f"--prefix={prefix}",
                f"--with-rocm={rocm_path}",
                "--without-knem",
                "--without-cuda",
                *_host_compilers(cmake_executor),
            ],
            env_prefix=f"ROCM_PATH={rocm_path} LD_LIBRARY_PATH={rocm_path}/lib:{rocm_path}/lib64:$LD_LIBRARY_PATH",
            make_install=True,
            sentinel=_UCX_SENTINEL,
            log_dir=log_dir,
            use_lock=False,
            timeout=build_timeout,
        )
        if not build_artifact_exists(pathlib.Path(build_dir), _UCX_SENTINEL, cmake_executor):
            raise RuntimeError(
                f"LAMMPS: UCX {version} built without the ROCm transport "
                f"({build_dir}/{_UCX_SENTINEL} missing). Check that {rocm_path} is a "
                "full ROCm install; scores from a host-memory fallback are not comparable."
            )
    return prefix


@pytest.fixture(scope="session")
def lammps_mpi_runtime(
    lammps_ucx: str,
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
):
    """Return an ``MpiRuntime`` (launcher + build/run env) for the LAMMPS suite.

    OpenMPI is built against the ROCm-aware UCX from ``lammps_ucx`` rather than
    taken from the node: the system MPI differs across distro images and its
    choice of transport shifts the multi-GPU scores, which would make results
    incomparable between machines.  LAMMPS is compiled against this same
    toolchain, so the build and the run cannot drift apart, and the install is
    cached behind a sentinel, so only the first run on a node pays for it.
    """
    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)
    version = _openmpi_version()
    log_dir = os.path.join(framework_config.framework.artifact_dir, "lammps_kokkos")
    os.makedirs(log_dir, exist_ok=True)

    root = pathlib.Path(compiler_build_dir) / "lammps_kokkos" / f"openmpi-{version}"
    with external_build.build_lock(f"hpc-lammps-kokkos-openmpi-{version}", timeout=build_timeout):
        root_abs = _abs_path(str(root), cmake_executor)
        source_dir = f"{root_abs}/openmpi-{version}"
        build_dir = f"{source_dir}/build"
        prefix = f"{build_dir}/install"

        if not build_artifact_exists(pathlib.Path(build_dir), _MPI_SENTINEL, cmake_executor):
            # OpenMPI ships release tarballs rather than build-ready git tags, so
            # the source is fetched the same way upstream publishes it.
            major_minor = ".".join(version.split(".")[:2])
            url = f"https://download.open-mpi.org/release/open-mpi/v{major_minor}/openmpi-{version}.tar.gz"
            logger.info("LAMMPS: fetching OpenMPI %s", version)
            _run_build_step(
                f"mkdir -p {shlex.quote(root_abs)} && cd {shlex.quote(root_abs)} && "
                f"curl -fsSL -o openmpi.tar.gz {shlex.quote(url)} && "
                "tar -xf openmpi.tar.gz && rm -f openmpi.tar.gz",
                cmake_executor=cmake_executor,
                timeout=build_timeout,
                label="openmpi fetch",
                log_path=os.path.join(log_dir, "build-openmpi-fetch.log"),
            )

        logger.info("LAMMPS: building OpenMPI %s against UCX at %s", version, lammps_ucx)
        external_build.autotools_make(
            source_dir,
            build_dir,
            configure_script="../configure",
            configure_args=[
                f"--prefix={prefix}",
                f"--with-ucx={lammps_ucx}",
                f"--with-rocm={rocm_path}",
                # btl-uct duplicates what the UCX PML already provides and fails
                # to build against newer UCX releases.
                "--enable-mca-no-build=btl-uct",
                *_host_compilers(cmake_executor),
            ],
            env_prefix=f"ROCM_PATH={rocm_path} LD_LIBRARY_PATH={rocm_path}/lib:{rocm_path}/lib64:$LD_LIBRARY_PATH",
            make_install=True,
            sentinel=_MPI_SENTINEL,
            log_dir=log_dir,
            use_lock=False,
            timeout=build_timeout,
        )

    launcher = f"{prefix}/bin/mpirun"
    return MpiRuntime(launcher=launcher, env=_mpi_runtime_env(prefix))


@pytest.fixture(scope="session")
def lammps_build(
    rock_dir: str,
    gpu_arch: str | None,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
    lammps_mpi_runtime,
    require_gpu_arch_for,
    built_binary,
) -> tuple[str, str]:
    """Clone and build LAMMPS with Kokkos/HIP; return ``(source_dir, lmp_path)``.

    The source dir is returned alongside the binary because each benchmark runs
    from its own directory inside the checkout -- the input decks reference their
    potential files by relative path.

    The build tree is namespaced by GPU arch: ``--offload-arch`` bakes a
    single-arch code object in, so a gfx942 build cannot run on a gfx950 GPU.
    Everything the run matrix varies (box size, steps, rank count) is a runtime
    argument to ``lmp``, so one build per arch serves all 16 variants.

    Because the test is parametrized, several xdist workers may request this
    session-scoped build concurrently for the same arch. ``external_build.build_lock``
    serializes them: the first worker builds; the rest block, then reuse the
    binary via the sentinel check.
    """
    require_gpu_arch_for("lammps_kokkos")  # --offload-arch must be explicit -- no auto-detect here.
    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)

    ref_label = _safe_ref_name(_LAMMPS_REF)
    arch_label = gpu_arch or "auto"
    # lmp links against the MPI it was built with, so the MPI version belongs in
    # the build key alongside the arch. Without it a tree built earlier against a
    # different MPI would satisfy the sentinel and then be launched under this
    # mpirun, which is an ABI mismatch rather than a slow path.
    build_label = f"{arch_label}-ompi{_safe_ref_name(_openmpi_version())}"
    dest = pathlib.Path(compiler_build_dir) / "lammps_kokkos" / f"lammps-{ref_label}"

    with external_build.build_lock(f"hpc-lammps-kokkos-{ref_label}-{build_label}", timeout=build_timeout):
        lammps_dir = external_build.clone_repo(url=_LAMMPS_URL, dest=dest, ref=_LAMMPS_REF, timeout=build_timeout)
        # clone_repo hands back a repo-relative path locally, but the benchmark
        # cds into the deck's own directory before launching mpirun -- anything
        # still relative at that point stops resolving.
        lammps_dir = _abs_path(str(lammps_dir), cmake_executor)
        external_build.assert_license_present(lammps_dir)

        build_dir = f"{lammps_dir}/build-{build_label}"
        lmp_binary = f"{build_dir}/{_BUILD_SENTINEL}"

        if build_artifact_exists(pathlib.Path(build_dir), _BUILD_SENTINEL, cmake_executor):
            logger.info("LAMMPS: existing build at %s -- skipping configure/build", build_dir)
            return str(lammps_dir), lmp_binary

        mpi_home = lammps_mpi_runtime.env.get("MPI_HOME", "")
        mpi_bin = os.path.dirname(lammps_mpi_runtime.launcher)
        mpi_lib = lammps_mpi_runtime.env.get("LD_LIBRARY_PATH", "")

        env_prefix = (
            f"MPI_HOME={mpi_home} "
            f"ROCM_PATH={rocm_path} "
            f"PATH={mpi_bin}:{rocm_path}/bin:$PATH "
            f"LD_LIBRARY_PATH={mpi_lib}:{rocm_path}/lib:$LD_LIBRARY_PATH"
        )

        hwloc = "on" if _ensure_hwloc(cmake_executor) else "off"
        if hwloc == "off":
            logger.warning(
                "LAMMPS: hwloc headers unavailable and could not be installed; building with "
                "Kokkos_ENABLE_HWLOC=off. Ranks lose CPU pinning, which moves the reported score."
            )

        jobs = resolve_parallel_jobs(remote_executor=cmake_executor)
        log_dir = os.path.join(framework_config.framework.artifact_dir, "lammps_kokkos")
        os.makedirs(log_dir, exist_ok=True)

        # REAXFF and MANYBODY supply the reaxff/tersoff pair styles; MOLECULE,
        # KSPACE, RIGID and ML-SNAP are kept on so the build also covers the
        # wider set of decks shipped in bench/. CMAKE_PREFIX_PATH carries ROCm so
        # the HIPFFT backend's cmake package is found without a separate hint.
        configure_flags = (
            "-DPKG_KOKKOS=on "
            "-DPKG_REAXFF=on "
            "-DPKG_MANYBODY=on "
            "-DPKG_ML-SNAP=on "
            "-DPKG_MOLECULE=on "
            "-DPKG_KSPACE=on "
            "-DPKG_RIGID=on "
            "-DBUILD_MPI=on "
            "-DMPI_CXX_SKIP_MPICXX=on "
            "-DFFT_KOKKOS=HIPFFT "
            f"-DMPI_C_COMPILER={mpi_bin}/mpicc "
            f"-DMPI_CXX_COMPILER={mpi_bin}/mpicxx "
            f"-DCMAKE_PREFIX_PATH='{mpi_home};{rocm_path}' "
            "-DCMAKE_BUILD_TYPE=Release "
            "-DKokkos_ENABLE_HIP=on "
            "-DKokkos_ENABLE_SERIAL=on "
            f"-DKokkos_ENABLE_HWLOC={hwloc} "
            "-DKokkos_ARCH_NATIVE=ON "
            "-DKokkos_ENABLE_HIP_MULTIPLE_KERNEL_INSTANTIATIONS=ON "
            # FindOpenMP accepts hipcc's -fopenmp=libomp but leaves
            # OpenMP_CXX_LIB_NAMES empty, so the OpenMP::OpenMP_CXX target
            # carries the compile flag with no library attached and lmp fails
            # to link on omp_*/__kmpc_* symbols. Point it at the libomp that
            # ships with ROCm's LLVM.
            "-DOpenMP_CXX_LIB_NAMES=omp "
            f"-DOpenMP_omp_LIBRARY={rocm_path}/lib/llvm/lib/libomp.so "
            "-DCMAKE_CXX_STANDARD=20 "
            f"-DCMAKE_CXX_COMPILER={rocm_path}/bin/hipcc "
            "-DLAMMPS_SIZES=smallbig "
            f"-DCMAKE_CXX_FLAGS='--offload-arch={gpu_arch} {_FMT_CONSTEVAL_FLAG}'"
        )

        # LAMMPS keeps its CMakeLists under cmake/, not at the repo root.
        configure_cmd = f"env {env_prefix} cmake -S {lammps_dir}/cmake -B {build_dir} {configure_flags}"
        build_cmd = f"env {env_prefix} cmake --build {build_dir} --parallel {jobs}"

        logger.info("LAMMPS: configuring (arch=%s, mpi_home=%s, hwloc=%s)", arch_label, mpi_home or "<unset>", hwloc)
        _run_build_step(
            configure_cmd,
            cmake_executor=cmake_executor,
            timeout=build_timeout,
            label="cmake configure",
            log_path=os.path.join(log_dir, "build-configure.log"),
        )
        logger.info("LAMMPS: building with %s parallel jobs", jobs)
        _run_build_step(
            build_cmd,
            cmake_executor=cmake_executor,
            timeout=build_timeout,
            label="cmake build",
            log_path=os.path.join(log_dir, "build-compile.log"),
        )

        return str(lammps_dir), built_binary(lmp_binary, "lammps_kokkos")
