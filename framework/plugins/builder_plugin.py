# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
builder_plugin.py -- HIP/C++ binary compilation fixtures.

Provides: rock_dir, compiler_build_dir, compile_binary, ld_path, gpu_arch, arch_lib_path.

rock_dir resolution order: --rock-dir CLI → ROCK_DIR env → rocm-test.toml → "" (empty).
compile_binary wraps BinaryBuilder — session-scoped, incremental, xdist-safe via file lock.
For .hip sources or CMake builds use cmake_build() from framework.builder.binary_builder instead.

CLI options added: --rock-dir, --compiler-build-dir.
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

from framework.builder.binary_builder import (
    BinaryBuilder,
    assert_binary_exists,
    assert_license_present,
    build_artifact_exists,
    build_cache_action,
    clone_repo,
    cmake_build,
    detect_mpi_runtime,
    find_rocm_clangpp,
    make_build,
    provision_openmpi_runtime,
    require_gpu_arch,
    source_tree_fingerprint,
    wipe_build_dir,
    write_build_fingerprint,
)
from framework.common.workspace_layout import local_external_clone_dest

logger = logging.getLogger(__name__)

_FRAMEWORK_CMAKE_DIR = pathlib.Path(__file__).resolve().parents[1] / "builder" / "cmake"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ROCm compiler CLI options."""
    group = parser.getgroup("rocm-compiler", "ROCm Compiler options")
    group.addoption(
        "--rock-dir",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Path to TheRock/ROCm install dir (contains bin/hipcc, lib/). "
            "Also read from ROCK_DIR or ROCM_TEST_THEROCK_ROCK_DIR env vars."
        ),
    )
    group.addoption(
        "--compiler-build-dir",
        action="store",
        default=None,
        metavar="PATH",
        help=(
            "Output directory for compiled test binaries. "
            "Defaults to output/test-binaries/ (or rocm-test.toml [therock] build_dir)."
        ),
    )


@pytest.fixture(scope="session")
def rock_dir(request: pytest.FixtureRequest, framework_config) -> str:
    """Resolve the TheRock/ROCm installation path.

    Resolution order (first non-empty wins):
        1. ``--rock-dir`` CLI flag
        2. ``ROCK_DIR`` environment variable
        3. ``ROCM_TEST_THEROCK_ROCK_DIR`` environment variable
        4. ``framework_config.therock.rock_dir`` (rocm-test.toml or defaults)

    Returns:
        Absolute, realpath-resolved path string.

    Raises:
        pytest.fail.Exception: When no path is available from any source — this is
            a CI environment defect, not a resource shortage.
    """
    path: str | None = (
        request.config.getoption("--rock-dir", default=None)
        or os.environ.get("ROCK_DIR")
        or os.environ.get("ROCM_TEST_THEROCK_ROCK_DIR")
        or framework_config.therock.rock_dir
        or None
    )
    if not path:
        pytest.fail(
            "rock_dir not configured — pass --rock-dir=<path>, "
            "set ROCK_DIR=<path>, or set [therock] rock_dir in rocm-test.toml"
        )
    resolved = os.path.realpath(path)
    logger.info("rock_dir resolved to: %s", resolved)
    return resolved


@pytest.fixture(scope="session")
def compiler_build_dir(request: pytest.FixtureRequest, framework_config) -> str:
    """Return the root directory for compiled test binaries, creating it if needed.

    Resolution order (first non-empty wins):
        1. ``--compiler-build-dir`` CLI flag
        2. ``framework_config.therock.build_dir`` (rocm-test.toml or default)

    The default is ``output/test-binaries/``.  Component-specific sub-directories
    (e.g. ``output/test-binaries/compiler/``, ``output/test-binaries/multi_gpu/``)
    are created on demand by ``compile_binary(subdir=...)``.

    Returns:
        Path string (relative or absolute, as configured).
    """
    path: str = request.config.getoption("--compiler-build-dir", default=None) or framework_config.therock.build_dir
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("compiler_build_dir: %s", path)
    return path


@pytest.fixture(scope="session")
def ld_path(rock_dir: str) -> dict:  # pylint: disable=redefined-outer-name
    """Build an LD_LIBRARY_PATH env dict for binaries linked against TheRock libs.

    Prepends ``{rock_dir}/lib`` to the existing ``LD_LIBRARY_PATH`` so that
    the TheRock-built ``libamdhip64.so`` and related libraries are found at
    runtime without requiring a system-level install.

    Args:
        rock_dir: Resolved path to the TheRock/ROCm installation.

    Returns:
        Dict ``{"LD_LIBRARY_PATH": "<rock_dir>/lib:<existing_path>"}``
        suitable for merging into a GPU execution env dict.
    """
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    lib_path = f"{rock_dir}/lib:{existing}" if existing else f"{rock_dir}/lib"
    return {"LD_LIBRARY_PATH": lib_path}


@pytest.fixture(scope="session")
def built_binary(cmake_executor):
    """Return a path checker for binaries produced by local or remote builds."""

    def _binary(path: str | os.PathLike, name: str) -> str:
        return assert_binary_exists(str(path), cmake_executor, name)

    return _binary


@pytest.fixture(scope="session")
def require_gpu_arch_for(gpu_arch: str | None):
    """Return a small guard for fixtures that must pass an explicit GPU arch."""

    def _require(label: str) -> None:
        require_gpu_arch(gpu_arch, label)

    return _require


@pytest.fixture(scope="session")
def arch_lib_path(gpu_arch: str | None):
    """Return a callable that resolves an arch-specific library sub-directory path.

    Appends ``/<gpu_arch>`` to the base path when ``--gpu-arch`` is set so that
    ROCm library runtimes (e.g. hipBLASLt/Tensile) locate their arch-specific files.
    Returns ``str(base)`` unchanged when ``gpu_arch`` is ``None``.

    Args:
        gpu_arch: Architecture string from the ``gpu_arch`` fixture, or ``None``.

    Returns:
        Callable ``(base: str | pathlib.Path) -> str``.
    """
    import pathlib as _pathlib

    def _resolve(base) -> str:
        p = _pathlib.Path(base)
        return str(p / gpu_arch) if gpu_arch else str(p)

    return _resolve


@pytest.fixture(scope="session")
def compile_binary(
    rock_dir: str, compiler_build_dir: str, framework_config, gpu_arch: str | None, cmake_executor
):  # pylint: disable=redefined-outer-name
    """Return a compile factory pre-bound to rock_dir and compiler_build_dir.

    The returned callable wraps ``BinaryBuilder`` so that e2e tests can compile
    a HIP/C++ binary without importing ``BinaryBuilder`` or wiring up path
    resolution directly.  See ``BinaryBuilder.compile()`` for parameter details.

    Example::

        @pytest.fixture(scope="session")
        def allreduce_binary(compile_binary):
            return compile_binary(
                src="tests/e2e/multi_gpu/src/allreduce.cpp",
                output_name="allreduce",
                subdir="multi_gpu",
            )
    """

    def _compile(
        src: str,
        output_name: str,
        *,
        include_dirs: list[str] | None = None,
        extra_flags: list[str] | None = None,
        std: str = "c++17",
        opt: str = "-O2",
        arch: str | None = None,
        subdir: str | None = None,
    ) -> str:
        resolved_arch = arch if arch is not None else gpu_arch
        out_dir = os.path.join(compiler_build_dir, subdir) if subdir else compiler_build_dir
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        log_path = os.path.join(out_dir, output_name + ".build.log")
        return BinaryBuilder().compile(
            rocm_dir=rock_dir,
            src=src,
            output=os.path.join(out_dir, output_name),
            std=std,
            opt=opt,
            arch=resolved_arch,
            include_dirs=include_dirs,
            extra_flags=extra_flags,
            timeout=framework_config.therock.build_timeout_secs,
            inactivity_timeout=framework_config.therock.build_inactivity_timeout_secs,
            log_path=log_path,
            remote_executor=cmake_executor,
        )

    return _compile


@pytest.fixture(scope="session")
def cmake_build_dir(
    rock_dir: str, compiler_build_dir: str, framework_config, gpu_arch: str | None, cmake_executor
):  # pylint: disable=redefined-outer-name
    """Return a CMake build factory bound to the session build config.

    ``compiler_mode`` controls only compiler injection: ``auto`` requires ROCm
    clang++, ``cxx_hip`` also sets ``CMAKE_HIP_COMPILER``, ``optional_*`` keeps
    the old "use clang++ if present, otherwise let CMake fall back" behavior, and
    ``none`` leaves compiler discovery entirely to the upstream CMake project.
    """
    rocm_path = os.path.realpath(rock_dir)
    session_gpu_arch = gpu_arch

    def _build(
        src: str,
        subdir: str,
        *,
        extra_cmake_args: list[str] | None = None,
        gpu_arch_var: str = "GPU_ARCH",
        gpu_arch: str | None = None,
        sync_dirs: list[str] | None = None,
        label: str | None = None,
        artifact: str | None = None,
        compiler_mode: str = "auto",
        target: str | None = None,
    ) -> str:
        """Configure/build a CMake project and return the node-local build dir."""
        _label = label or subdir
        resolved_gpu_arch = gpu_arch if gpu_arch is not None else session_gpu_arch
        # Remote in-tree builds (sync_dirs provided) are staged under the SSH
        # executor's managed SFTP workspace. External builds (no sync_dirs) clone
        # and build under the managed work workspace. In both cases this fixture
        # computes the final remote path up front so fingerprint checks, cmake,
        # and returned binary paths all point at the same location.
        if cmake_executor is not None:
            build_dir = pathlib.Path(
                cmake_executor.workspace_path_for(pathlib.Path(compiler_build_dir) / subdir / "build")
            )
        else:
            build_dir = pathlib.Path(compiler_build_dir) / subdir / "build"
        local_source_dirs: list[str | os.PathLike] = [src, *list(sync_dirs or [])]
        source_fp = source_tree_fingerprint(local_source_dirs)

        # Fingerprint-based cache check (only when artifact sentinel is provided).
        if artifact is not None and build_artifact_exists(build_dir, artifact, cmake_executor):
            full_inputs = [rocm_path, resolved_gpu_arch or "", src, subdir, source_fp, *sorted(extra_cmake_args or [])]
            full_fp = _build_fp(full_inputs)
            structural_fp = _build_fp([rocm_path, resolved_gpu_arch or "", src, subdir])
            action = build_cache_action(build_dir, structural_fp, full_fp, cmake_executor)
            if action == "skip":
                logger.info("%s: fingerprint matches — skipping cmake", _label)
                return str(build_dir)
            if action == "wipe":
                logger.info("%s: structural fingerprint changed — wiping build dir", _label)
                wipe_build_dir(build_dir, cmake_executor)
            else:
                logger.info("%s: full fingerprint changed — incremental rebuild", _label)

        # Preserve each suite's original compiler contract while centralizing
        # where the decision is made. Some legacy ports require ROCm clang++;
        # others intentionally let their CMakeLists fall back to hipcc.
        compiler_args: list[str] | None = None
        if compiler_mode in ("auto", "cxx_hip", "optional_auto", "optional_cxx_hip"):
            clangpp = find_rocm_clangpp(rocm_path)
            optional_compiler = compiler_mode.startswith("optional_")
            if cmake_executor is None and clangpp is None and not optional_compiler:
                pytest.fail(
                    f"ROCm clang++ not found under {rocm_path} — required for {_label}. "
                    "Verify ROCK_DIR / --rock-dir points to a complete ROCm install."
                )
            if clangpp is not None or not optional_compiler:
                compiler_path = clangpp or (pathlib.Path(rocm_path) / "lib" / "llvm" / "bin" / "clang++")
                compiler_args = [f"-DCMAKE_CXX_COMPILER={compiler_path}"]
                if compiler_mode in ("cxx_hip", "optional_cxx_hip"):
                    compiler_args.append(f"-DCMAKE_HIP_COMPILER={compiler_path}")

        module_dir = (
            cmake_executor.remote_path_for(str(_FRAMEWORK_CMAKE_DIR))
            if cmake_executor is not None
            else str(_FRAMEWORK_CMAKE_DIR)
        )
        effective_extra_args = list(extra_cmake_args or [])
        effective_extra_args.append(f"-DROCM_TEST_CMAKE_MODULE_DIR={module_dir}")
        effective_sync_dirs = [os.path.abspath(d) for d in (sync_dirs or [])]
        if cmake_executor is not None:
            effective_sync_dirs.append(str(_FRAMEWORK_CMAKE_DIR))

        actual_build_dir = cmake_build(
            src=str(pathlib.Path(src).resolve()) if cmake_executor is None else src,
            build_dir=str(build_dir),
            rocm_path=rocm_path,
            gpu_arch=resolved_gpu_arch,
            gpu_arch_var=gpu_arch_var,
            compiler_args=compiler_args,
            extra_cmake_args=effective_extra_args,
            label=_label,
            remote_executor=cmake_executor,
            sync_dirs=effective_sync_dirs if cmake_executor is not None else sync_dirs,
            target=target,
        )

        if artifact is not None:
            full_inputs = [rocm_path, resolved_gpu_arch or "", src, subdir, source_fp, *sorted(extra_cmake_args or [])]
            full_fp = _build_fp(full_inputs)
            structural_fp = _build_fp([rocm_path, resolved_gpu_arch or "", src, subdir])
            write_build_fingerprint(actual_build_dir, full_fp, cmake_executor, structural=structural_fp)

        return str(actual_build_dir)

    return _build


def _build_fp(inputs: list[str]) -> str:
    """Internal shortcut: compute a build fingerprint from a list of strings."""
    from framework.builder.binary_builder import compute_fingerprint  # local import to avoid circular

    return compute_fingerprint(inputs)


@pytest.fixture(scope="session")
def external_build(compiler_build_dir: str, framework_config, cmake_executor):
    """Return remote-aware helpers for external clone/make-style test suites."""
    build_timeout = float(framework_config.therock.build_timeout_secs)

    class _ExternalBuild:
        def clone_repo(
            self,
            url: str,
            dest,
            ref: str | None = None,
            *,
            sparse_subtree: str | None = None,
            timeout: float = 1800.0,
        ) -> pathlib.Path:
            """Clone once into the build workspace, locally or on the remote node."""
            clone_dest = dest if cmake_executor is not None else local_external_clone_dest(dest, compiler_build_dir)
            return clone_repo(
                url=url,
                dest=clone_dest,
                ref=ref,
                timeout=timeout,
                sparse_subtree=sparse_subtree,
                remote_executor=cmake_executor,
            )

        def make_build(
            self,
            repo_dir,
            make_args: list[str] | None = None,
            *,
            env: dict[str, str] | None = None,
            timeout: float | None = None,
        ) -> None:
            """Run ``make`` in a cloned tree using the active build executor."""
            make_build(
                repo_dir=repo_dir,
                make_args=make_args,
                env=env,
                timeout=timeout if timeout is not None else build_timeout,
                remote_executor=cmake_executor,
            )

        def assert_license_present(self, path) -> None:
            """Verify a cloned third-party tree carries a recognizable license."""
            assert_license_present(path, remote_executor=cmake_executor)

        def detect_mpi_runtime(self):
            """Discover MPI on the same node that will build external sources."""
            return detect_mpi_runtime(remote_executor=cmake_executor)

        def provision_openmpi_runtime(self, version: str):
            """Build a private OpenMPI when the target node has no MPI runtime."""
            return provision_openmpi_runtime(
                compiler_build_dir,
                version=version,
                remote_executor=cmake_executor,
            )

    return _ExternalBuild()
