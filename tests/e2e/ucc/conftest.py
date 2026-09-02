# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- UCX / OpenMPI / UCC provisioning for tests/e2e/ucc/.

``ucc_test_mpi`` needs a GPU-aware MPI stack that no ROCm package ships and that
the framework does not otherwise provide: ``detect_mpi_runtime`` finds host MPI
without UCX, and ``provision_openmpi_runtime`` builds a CPU-only OpenMPI.  This
module therefore reproduces the three autotools builds performed by ROCmTest's
``ucc_scatter_gather.sh``:

    1. UCX      configured ``--with-rocm``
    2. OpenMPI  configured ``--with-ucx`` against that UCX
    3. UCC      configured against UCX, OpenMPI, ROCm and RCCL

RCCL comes from the ROCm install rather than from source, matching the original:
its step 2 is answered "n" under ``-q``, which points ``--with-rccl`` at
``$_ROCM_DIR``.

Each step is skipped when its install sentinel already exists, so the (long)
first build is paid once per workspace.  Set ``ROCM_TEST_UCX_PREFIX``,
``ROCM_TEST_OMPI_PREFIX`` and ``ROCM_TEST_UCC_PREFIX`` to reuse an existing
stack, or ``ROCM_TEST_UCC_AUTO_BUILD=0`` to skip the suite instead of building.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
import pathlib

import pytest

from framework.builder.binary_builder import resolve_parallel_jobs
from framework.executors.background_process import _blocking_stream_run

logger = logging.getLogger(__name__)

_AUTO_BUILD = os.environ.get("ROCM_TEST_UCC_AUTO_BUILD", "1").strip().lower() not in ("0", "false", "no")

# --- source repositories (ucc_scatter_gather.sh steps 1, 3 and 4) -----------
UCX_REPO = os.environ.get("ROCM_TEST_UCX_REPO", "https://github.com/ROCmSoftwarePlatform/ucx")
UCX_REF = os.environ.get("ROCM_TEST_UCX_REF", "develop")

OMPI_REPO = os.environ.get("ROCM_TEST_OMPI_REPO", "https://github.com/open-mpi/ompi")
OMPI_REF = os.environ.get("ROCM_TEST_OMPI_REF", "v5.0.x")

UCC_REPO = os.environ.get("ROCM_TEST_UCC_REPO", "https://github.com/ROCmSoftwarePlatform/ucc")
UCC_REF = os.environ.get("ROCM_TEST_UCC_REF", "develop")


@dataclass(frozen=True)
class UccStack:
    """Resolved install prefixes for the UCC MPI workload."""

    ucx_prefix: str
    ompi_prefix: str
    ucc_prefix: str
    rocm_prefix: str

    @property
    def mpirun(self) -> str:
        """Path to the UCX-enabled ``mpirun``."""
        return f"{self.ompi_prefix}/bin/mpirun"

    @property
    def ucc_test_mpi(self) -> str:
        """Path to the ``ucc_test_mpi`` collective driver."""
        return f"{self.ucc_prefix}/bin/ucc_test_mpi"

    @property
    def ld_library_path(self) -> str:
        """Loader path covering OpenMPI, UCC, UCX and ROCm (RCCL included)."""
        return ":".join(
            (
                f"{self.ompi_prefix}/lib",
                f"{self.ucc_prefix}/lib",
                f"{self.ucx_prefix}/lib",
                f"{self.rocm_prefix}/lib",
            )
        )


@dataclass(frozen=True)
class _Builder:
    """Execution context shared by the three provisioning steps."""

    executor: object | None
    timeout: float
    jobs: int
    log_dir: str
    rocm_prefix: str

    def exists(self, path: str) -> bool:
        """Return True when *path* is an existing file, locally or remotely."""
        if self.executor is not None:
            return self.executor.run(f"test -f {path}", timeout=30.0).ok  # type: ignore[attr-defined]
        return os.path.isfile(path)

    def run(self, command: str, *, cwd: str, label: str) -> None:
        """Run one build step, streaming output live and into a per-step log."""
        log_path = os.path.join(self.log_dir, f"{label}.log")
        logger.info("UCC stack: %s -> streaming to %s", label, log_path)
        # ROCm's bin must lead PATH so UCX/UCC configure find hipcc rather than a
        # stray system copy; the environment is injected per command instead of
        # mutating os.environ, which would leak into unrelated fixtures.
        env_prefix = f"PATH={self.rocm_prefix}/bin:$PATH ROCM_PATH={self.rocm_prefix}"
        wrapped = f"cd {cwd} && env {env_prefix} bash -c {_quote(command)}"

        if self.executor is not None:
            result = self.executor.run(wrapped, timeout=self.timeout, stream=True)  # type: ignore[attr-defined]
            if not result.ok:
                raise RuntimeError(
                    f"UCC stack step '{label}' failed on remote (exit={result.exit_code}):\n"
                    f"stdout: {result.stdout[-4000:]}\nstderr: {result.stderr[-2000:]}"
                )
            return

        result = _blocking_stream_run(
            command=wrapped,
            env=os.environ.copy(),
            cwd=None,
            timeout=self.timeout,
            stream_stdout=True,
            stream_stderr=True,
            log_path=log_path,
        )
        if not result.ok:
            raise RuntimeError(
                f"UCC stack step '{label}' failed (exit={result.exit_code}). Full log: {log_path}\n"
                f"stdout tail: {result.stdout[-4000:]}\nstderr tail: {result.stderr[-2000:]}"
            )


def _quote(command: str) -> str:
    """Single-quote *command* for use as a ``bash -c`` argument."""
    return "'" + command.replace("'", "'\\''") + "'"


def _build_ucx(builder: _Builder, src: str) -> str:
    """Build UCX ``--with-rocm``; return its install prefix."""
    prefix = f"{src}/install"
    if builder.exists(f"{prefix}/lib/libucp.so"):
        logger.info("UCC stack: reusing UCX at %s", prefix)
        return prefix
    builder.run(
        "./autogen.sh"
        " && mkdir -p build"
        f" && cd build && ../configure --prefix={prefix} --libdir={prefix}/lib"
        f" --with-rocm={builder.rocm_prefix} --without-knem --without-gdrcopy"
        f" && make -j{builder.jobs} && make install",
        cwd=src,
        label="ucx",
    )
    return prefix


def _build_ompi(builder: _Builder, src: str, ucx_prefix: str) -> str:
    """Build OpenMPI against *ucx_prefix*; return its install prefix."""
    prefix = f"{src}/install"
    if builder.exists(f"{prefix}/bin/mpirun"):
        logger.info("UCC stack: reusing OpenMPI at %s", prefix)
        return prefix
    # OpenMPI's git tree carries prrte/openpmix/hwloc as submodules; autogen.pl
    # aborts without them, and clone_repo does not recurse.
    builder.run(
        "git submodule update --init --recursive"
        " && ./autogen.pl"
        f" && ./configure --prefix={prefix} --libdir={prefix}/lib --with-ucx={ucx_prefix}"
        " --disable-sphinx --disable-mpi-fortran --disable-oshmem --with-hcoll=no"
        f" && make -j{builder.jobs} && make install",
        cwd=src,
        label="openmpi",
    )
    return prefix


def _build_ucc(builder: _Builder, src: str, ucx_prefix: str, ompi_prefix: str) -> str:
    """Build UCC against UCX, OpenMPI, ROCm and RCCL; return its install prefix."""
    prefix = f"{src}/install"
    if builder.exists(f"{prefix}/bin/ucc_test_mpi"):
        logger.info("UCC stack: reusing UCC at %s", prefix)
        return prefix
    builder.run(
        "./autogen.sh"
        f" && ./configure --prefix={prefix} --libdir={prefix}/lib --with-ucx={ucx_prefix}"
        f" --with-rocm={builder.rocm_prefix} --with-rccl={builder.rocm_prefix}"
        f" --with-mpi={ompi_prefix} --enable-gtest"
        f" && make -j{builder.jobs} && make install",
        cwd=src,
        label="ucc",
    )
    return prefix


def _prebuilt_stack(rocm_prefix: str) -> UccStack | None:
    """Return a stack from ``ROCM_TEST_*_PREFIX`` overrides when all are set."""
    ucx = os.environ.get("ROCM_TEST_UCX_PREFIX", "").strip()
    ompi = os.environ.get("ROCM_TEST_OMPI_PREFIX", "").strip()
    ucc = os.environ.get("ROCM_TEST_UCC_PREFIX", "").strip()
    if not (ucx and ompi and ucc):
        return None
    logger.info("UCC stack: using prebuilt prefixes ucx=%s ompi=%s ucc=%s", ucx, ompi, ucc)
    return UccStack(ucx_prefix=ucx, ompi_prefix=ompi, ucc_prefix=ucc, rocm_prefix=rocm_prefix)


def _require_rccl(rock_dir: str, cmake_executor) -> None:
    """Skip when RCCL is missing, since ``--with-rccl`` could not be satisfied.

    UCC's ``config/m4/rccl.m4`` probes ``rccl/rccl.h`` before ``rccl.h``, so both
    layouts are accepted here; ROCm ships the former.
    """
    headers = (f"{rock_dir}/include/rccl/rccl.h", f"{rock_dir}/include/rccl.h")
    if cmake_executor is not None:
        header_test = " || ".join(f"test -f {header}" for header in headers)
        probe = cmake_executor.run(f'bash -c "({header_test}) && ls {rock_dir}/lib/librccl.so*"', timeout=15.0)
        if not probe.ok:
            pytest.skip(f"RCCL not found under {rock_dir} on the build node — UCC cannot be configured")
        return
    has_header = any(os.path.isfile(header) for header in headers)
    has_lib = any(pathlib.Path(rock_dir, "lib").glob("librccl.so*"))
    if not (has_header and has_lib):
        pytest.skip(f"RCCL not found under {rock_dir} — UCC cannot be configured")


def _clone_sources(external_build, root: pathlib.Path, timeout: float) -> dict[str, str]:
    """Clone UCX, OpenMPI and UCC; return their absolute source paths by name."""
    repos = {
        "ucx": (UCX_REPO, UCX_REF),
        "ompi": (OMPI_REPO, OMPI_REF),
        "ucc": (UCC_REPO, UCC_REF),
    }
    sources: dict[str, str] = {}
    for name, (url, ref) in repos.items():
        src = external_build.clone_repo(url=url, dest=root / name, ref=ref, timeout=timeout)
        external_build.assert_license_present(src)
        # Local clones come back workspace-relative, but autotools rejects a
        # relative --prefix ("expected an absolute directory name").
        src_path = pathlib.Path(src)
        sources[name] = str(src_path if src_path.is_absolute() else src_path.resolve())
    return sources


@pytest.fixture(scope="session")
def ucc_stack(
    rock_dir: str,
    compiler_build_dir: str,
    framework_config,
    external_build,
    cmake_executor,
) -> UccStack:
    """Provision UCX, OpenMPI and UCC once per session; return their prefixes."""
    rocm_prefix = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir

    prebuilt = _prebuilt_stack(rocm_prefix)
    if prebuilt is not None:
        return prebuilt

    if not _AUTO_BUILD:
        pytest.skip(
            "UCC stack not provisioned and ROCM_TEST_UCC_AUTO_BUILD=0 — set the "
            "ROCM_TEST_{UCX,OMPI,UCC}_PREFIX overrides to use an existing install"
        )

    _require_rccl(rock_dir, cmake_executor)

    log_dir = os.path.join(framework_config.framework.artifact_dir, "ucc")
    os.makedirs(log_dir, exist_ok=True)
    builder = _Builder(
        executor=cmake_executor,
        timeout=float(framework_config.therock.build_timeout_secs),
        jobs=resolve_parallel_jobs(remote_executor=cmake_executor),
        log_dir=log_dir,
        rocm_prefix=rocm_prefix,
    )

    root = pathlib.Path(compiler_build_dir) / "ucc"
    sources = _clone_sources(external_build, root, builder.timeout)

    ucx_prefix = _build_ucx(builder, sources["ucx"])
    ompi_prefix = _build_ompi(builder, sources["ompi"], ucx_prefix)
    ucc_prefix = _build_ucc(builder, sources["ucc"], ucx_prefix, ompi_prefix)

    return UccStack(
        ucx_prefix=ucx_prefix,
        ompi_prefix=ompi_prefix,
        ucc_prefix=ucc_prefix,
        rocm_prefix=rocm_prefix,
    )
