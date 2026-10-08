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
import shlex

import pytest

from framework.executors.local_executor import run_cmd_get_stdout_stderr
from tests.common.gpu_pci_map import (
    ConfDirUnresolvedError,
    conf_filename_for,
    detect_device_key,
    detect_gpu_conf_dir,
)
from tests.common.rvs_config_map import conf_dir_for_module, covers_module, device_name, is_known_device

logger = logging.getLogger(__name__)

_RVS_REPO_URL = "https://github.com/ROCm/ROCmValidationSuite.git"
_RVS_REF = os.environ.get("ROCM_TEST_RVS_REF", "master")

# Build prerequisites ROCm does not ship: yaml-cpp is probed by the RVS
# CMakeLists, libnuma by its bundled TransferBench, which aborts later in the
# same configure. Package names and the EPEL requirement come from the RVS
# README; pciutils is absent because RVS builds its own static libpci. SLES
# carries yaml-cpp-devel only in the subscription-gated Development module.
_PREREQ_SCRIPT = (
    "if [ -f /usr/include/yaml-cpp/yaml.h ] && [ -f /usr/include/numa.h ]; then exit 0; fi; "
    'SUDO=""; if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then SUDO=sudo; fi; '
    "if command -v apt-get >/dev/null 2>&1; then "
    "  $SUDO apt-get update && $SUDO apt-get install -y libyaml-cpp-dev libnuma-dev; "
    "elif command -v dnf >/dev/null 2>&1 || command -v yum >/dev/null 2>&1; then "
    "  DNF=$(command -v dnf || command -v yum); "
    "  $SUDO $DNF install -y epel-release || true; "
    "  $SUDO $DNF install -y yaml-cpp-devel yaml-cpp-static numactl-devel; "
    "elif command -v zypper >/dev/null 2>&1; then "
    "  $SUDO zypper --non-interactive install yaml-cpp-devel libnuma-devel; "
    "else "
    '  echo "no supported package manager (apt-get/dnf/yum/zypper)" >&2; exit 1; '
    "fi"
)

# Bounds a wedged package mirror rather than budgeting for the few MB fetched.
_PREREQ_TIMEOUT = 600.0


def _find_preinstalled_rvs(rock_dir: str, cmake_executor=None) -> str:
    """Return the path of a pre-installed ``rvs``, or ``""`` if there is none.

    Both layouts are searched because the packages install into an ``extras-N``
    subdirectory rather than the ROCm prefix itself, so looking only in
    ``bin/rvs`` would miss a packaged RVS and fall through to a source build.
    That matters beyond the wasted build: a build owned by the test user cannot
    be loaded by ``rvs`` running under sudo, which the packaged root-owned copy
    handles fine.

    The config tree has to sit beside the binary, since an rvs without its conf
    directory can run no action at all.
    """
    if cmake_executor is None:
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
    # The loop exits non-zero when nothing matched, so the output is the signal
    # rather than the status.
    found = (cmake_executor.run(probe).stdout or "").strip()
    return found.splitlines()[0].strip() if found else ""


def _file_exists(path: pathlib.Path, cmake_executor=None) -> bool:
    """Check if a file exists locally or on remote node."""
    if cmake_executor is None:
        return path.is_file()
    return cmake_executor.run(f"test -f {shlex.quote(str(path))}").ok


def _dir_exists(path: pathlib.Path, cmake_executor=None) -> bool:
    """Check if a directory exists locally or on remote node."""
    if cmake_executor is None:
        return path.is_dir()
    return cmake_executor.run(f"test -d {shlex.quote(str(path))}").ok


def _is_executable(path: pathlib.Path, cmake_executor=None) -> bool:
    """Check if a path is an executable file locally or on remote node."""
    if cmake_executor is None:
        return path.is_file() and os.access(path, os.X_OK)
    return cmake_executor.run(f"test -x {shlex.quote(str(path))}").ok


def _ensure_build_prereqs(cmake_executor=None) -> None:
    """Best-effort install of yaml-cpp and libnuma before building RVS.

    Logged rather than raised on failure: RVS also accepts yaml-cpp from
    ``CMAKE_PREFIX_PATH``, so a node this cannot help may still configure, and
    the one that cannot gets a clearer error from CMake than from here.
    """
    if cmake_executor is not None:
        result = cmake_executor.run(f"bash -lc {shlex.quote(_PREREQ_SCRIPT)}", timeout=_PREREQ_TIMEOUT)
        ok, output = result.ok, (result.stdout or "") + (result.stderr or "")
    else:
        returncode, stdout, stderr = run_cmd_get_stdout_stderr("bash", "-lc", _PREREQ_SCRIPT, timeout=_PREREQ_TIMEOUT)
        ok, output = returncode == 0, (stdout or "") + (stderr or "")
    if not ok:
        logger.warning("Could not install RVS build prerequisites (yaml-cpp, libnuma):\n%s", output[-1000:])


def _find_first(
    root: pathlib.Path,
    pattern: str,
    cmake_executor=None,
    *,
    want_dir: bool = False,
) -> pathlib.Path | None:
    """Return the first entry under *root* whose path ends with *pattern*.

    Stands in for ``Path.rglob`` so a build or install tree that lives on the
    node under test is searched there. Resolved locally the same path names
    either nothing or a directory belonging to another user, which reads as
    "not built" or raises outright.
    """
    if cmake_executor is None:
        if not root.exists():
            return None
        return next(
            (c for c in root.rglob(pattern) if (c.is_dir() if want_dir else c.is_file())),
            None,
        )
    probe = (
        f"find {shlex.quote(str(root))} -path {shlex.quote('*/' + pattern)} "
        f"-type {'d' if want_dir else 'f'} 2>/dev/null | head -n 1"
    )
    found = (cmake_executor.run(probe).stdout or "").strip()
    return pathlib.Path(found) if found else None


def _detect_device_key(cmake_executor=None) -> str:
    """Detect the GPU key the qualification matrix is indexed by.

    A plain function rather than a fixture: ``rvs_find_conf`` is re-exported by
    sibling suites whose conftests cannot resolve fixtures defined here, so its
    fixture signature has to stay self-contained.
    """
    try:
        key = detect_device_key(cmake_executor=cmake_executor)
    except ConfDirUnresolvedError as exc:
        pytest.fail(str(exc))

    # Reported once here rather than per module, so a GPU the matrix does not
    # cover does not read as a string of "module not qualified" skips.
    if not is_known_device(key):
        pytest.fail(f"GPU {key} has no row in the RVS config mapping, so no module is qualified on it")
    return key


def _conf_dir_from_matrix(config_name: str, device_key: str) -> str:
    """Return the conf directory the qualification matrix gives for this module.

    Skips when the matrix has no column for the module at all, and when it has
    one that is empty for this GPU. Both mean the same thing for the run: no
    config here was ever qualified, so there is nothing meaningful to execute.
    """
    module = config_name[: -len(".conf")] if config_name.endswith(".conf") else config_name
    if not covers_module(module):
        pytest.skip(f"RVS {module} is not a module in the RVS config mapping")

    conf_dir = conf_dir_for_module(device_key, module)
    if conf_dir is None:
        pytest.skip(f"RVS {module} is not qualified on {device_name(device_key)} ({device_key})")
    return conf_dir


def _list_dir_entries(path: pathlib.Path, cmake_executor=None) -> list[str]:
    """Return the entry names directly under *path*, or [] if it cannot be listed."""
    if cmake_executor is None:
        try:
            return [entry.name for entry in path.iterdir()]
        except OSError:
            return []
    result = cmake_executor.run(f"ls -1 {shlex.quote(str(path))}")
    return result.stdout.splitlines() if result.ok else []


def _resolve_ignoring_case(root: pathlib.Path, parts: tuple[str, ...], cmake_executor=None) -> pathlib.Path | None:
    """Walk *parts* under *root*, matching each component without regard to case."""
    current = root
    for part in parts:
        entries = _list_dir_entries(current, cmake_executor)
        match = next((entry for entry in entries if entry.lower() == part.lower()), None)
        if match is None:
            return None
        current = current / match
    return current


def _locate_under_roots(search_roots: list[pathlib.Path], parts: tuple[str, ...], cmake_executor=None) -> str | None:
    """Return the first root holding ``parts``, preferring an exact-case match.

    Every root is tried exactly before any is tried case-insensitively, so a
    correctly cased file never loses to a differently cased one in an earlier
    root.
    """
    for root in search_roots:
        candidate = root.joinpath(*parts)
        if _file_exists(candidate, cmake_executor):
            return str(candidate)

    for root in search_roots:
        candidate = _resolve_ignoring_case(root, parts, cmake_executor)
        if candidate is not None:
            logger.debug("Resolved %s case-insensitively to %s", root.joinpath(*parts), candidate)
            return str(candidate)

    return None


def _detect_gpu_conf_dir(cmake_executor=None) -> str:
    """Detect GPU PCI device ID and map to RVS config directory name."""
    try:
        return detect_gpu_conf_dir(cmake_executor=cmake_executor)
    except ConfDirUnresolvedError as exc:
        # Carrying on with no directory would resolve the generic config, i.e.
        # settings this GPU was never qualified with; stop instead.
        pytest.fail(str(exc))


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
        elif cmake_executor.run(f"test -d {shlex.quote(str(p))}").ok:
            roots.append(p)
    return roots


def _resolve_conf_file(
    config_name: str,
    search_roots: list[pathlib.Path],
    cmake_executor=None,
    gpu_conf_dir: str = "",
) -> str | None:
    """Search the roots for the config the qualification matrix named.

    *gpu_conf_dir* is the matrix's answer, so an empty one means the top-level
    conf is the qualified config rather than a fallback, and a named directory
    missing the file is reported rather than resolved further up the tree.
    """
    if gpu_conf_dir:
        # A few directories ship a module's conf under a different name.
        gpu_config_name = conf_filename_for(gpu_conf_dir, config_name)
        resolved = _locate_under_roots(search_roots, (gpu_conf_dir, gpu_config_name), cmake_executor)
        if resolved:
            logger.info("Resolved config %s -> %s (GPU-specific: %s)", config_name, resolved, gpu_conf_dir)
            return resolved
        # The top-level conf holds settings this GPU was never qualified with,
        # so a missing mapped config is an install fault, not a path to fall
        # back to.
        return None

    resolved = _locate_under_roots(search_roots, (config_name,), cmake_executor)
    if resolved:
        logger.info("Resolved config %s -> %s (matrix: top-level conf)", config_name, resolved)
        return resolved

    return None


@pytest.fixture(scope="session")
def rvs_source(external_build, compiler_build_dir: str, cmake_executor) -> str:
    """Clone ROCmValidationSuite with submodules once per session; return source path."""
    dest = pathlib.Path(compiler_build_dir) / "rvs" / "ROCmValidationSuite"
    src_dir = external_build.clone_repo(_RVS_REPO_URL, dest, ref=_RVS_REF)
    external_build.assert_license_present(src_dir)
    if cmake_executor is not None:
        cmake_executor.run(
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
      1. Pre-installed at {rock_dir}/bin/rvs or {rock_dir}/extras*/bin/rvs
      2. Previously built at {rvs_source}/install/
      3. Build from source using framework cmake_build_dir + DESTDIR install
    """
    src_dir = pathlib.Path(rvs_source)
    install_dir = src_dir / "install"

    # 1. Check pre-installed
    if preinstalled := _find_preinstalled_rvs(rock_dir, cmake_executor):
        logger.info("Using pre-installed RVS: %s", preinstalled)
        return preinstalled

    # 2. Check previously built
    if previous := _find_first(install_dir, "bin/rvs", cmake_executor):
        logger.info("Using previously built RVS: %s", previous)
        return str(previous)

    # 3. Build from source via framework cmake_build_dir
    logger.info("Building RVS from source: %s", src_dir)
    _ensure_build_prereqs(cmake_executor)

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
    if cmake_executor is not None:
        cmake_executor.run(f"mkdir -p {shlex.quote(str(install_dir))}")
    else:
        install_dir.mkdir(parents=True, exist_ok=True)

    if cmake_executor is not None:
        result = cmake_executor.run(
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

    # 5. Locate installed binary
    rvs_bin = _find_first(install_dir, "bin/rvs", cmake_executor)
    if rvs_bin is None:
        pytest.fail(
            f"RVS binary not found under {install_dir} after install. "
            f"Contents: {_list_dir_entries(install_dir, cmake_executor)[:20]}"
        )

    logger.info("RVS binary installed at: %s", rvs_bin)
    return built_binary(str(rvs_bin), "rvs")


@pytest.fixture(scope="session")
def gpu_conf_dir(cmake_executor) -> str:
    """Auto-detect GPU and return the matching RVS config directory name."""
    return _detect_gpu_conf_dir(cmake_executor)


@pytest.fixture(scope="session")
def rvs_env(rvs_binary: str, rock_dir: str, ld_path: dict) -> str:
    """Return ``VAR=value`` assignments for running RVS via ``env``.

    ``libomp.so`` lives under ``llvm/lib`` and the binary may sit in an
    ``extras-N`` tree carrying its own ``lib``; without both on
    ``LD_LIBRARY_PATH`` RVS fails to load instead of reporting a verdict. Passing
    them explicitly also survives ``sudo``, which resets the environment for the
    modules that need root.
    """
    rvs_base = pathlib.Path(rvs_binary).resolve().parent.parent
    libs = [f"{rvs_base}/lib", f"{rock_dir}/lib", f"{rock_dir}/llvm/lib"]
    if existing := ld_path.get("LD_LIBRARY_PATH"):
        libs.append(existing)
    assignments = {
        "ROCM_PATH": rock_dir,
        "RVS_PATH": str(rvs_base),
        "LD_LIBRARY_PATH": ":".join(libs),
    }
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in assignments.items())


@pytest.fixture(scope="session")
def rvs_find_conf(rock_dir: str, rvs_source: str, cmake_executor, rvs_binary: str):
    """Return a factory that locates RVS config files with GPU-specific lookup."""
    device_key = _detect_device_key(cmake_executor)
    rock_dir_path = pathlib.Path(rock_dir)
    install_conf = _find_first(
        pathlib.Path(rvs_source) / "install",
        "share/rocm-validation-suite/conf",
        cmake_executor,
        want_dir=True,
    )
    source_conf = pathlib.Path(rvs_source) / "rvs" / "conf"

    def _find_conf(config_name: str, *, gpu_conf_dir: str = "") -> str:
        """Locate the config *config_name* is qualified with on this GPU.

        *gpu_conf_dir* is still accepted so suites that pass a pre-detected
        directory keep working, but the qualification matrix decides; they can
        drop the argument once they no longer need to support older checkouts.
        """
        del gpu_conf_dir
        conf_dir = _conf_dir_from_matrix(config_name, device_key)
        installed_conf = rock_dir_path / "share" / "rocm-validation-suite" / "conf"
        search_roots = _collect_conf_roots(
            _binary_conf_root(rvs_binary),
            installed_conf,
            install_conf,
            source_conf,
            cmake_executor=cmake_executor,
        )

        resolved = _resolve_conf_file(config_name, search_roots, cmake_executor, conf_dir)
        if resolved:
            return resolved

        where = f"{conf_dir}/" if conf_dir else "the top-level conf"
        pytest.fail(
            f"RVS {config_name} is qualified on {device_name(device_key)} under {where}, "
            f"but that config is missing from {[str(r) for r in search_roots]}"
        )

    return _find_conf


def _binary_conf_root(rvs_binary: str | None) -> pathlib.Path | None:
    """Return the ``conf`` tree that sits beside ``rvs_binary``.

    Both a packaged and a locally built rvs live at ``<base>/bin/rvs`` with their
    configs under the matching ``<base>/share``, so deriving the root from the
    resolved binary keeps the configs in step with the build that reads them
    instead of pairing one version's binary with another's config.
    """
    if not rvs_binary:
        return None
    base = pathlib.Path(rvs_binary).resolve().parent.parent
    return base / "share" / "rocm-validation-suite" / "conf"


def _rvs_install_base(rvs_source: str | None, cmake_executor=None) -> pathlib.Path | None:
    """Return the RVS ``install`` tree when the source checkout has one."""
    if not rvs_source:
        return None
    base = pathlib.Path(rvs_source) / "install"
    return base if _dir_exists(base, cmake_executor) else None


def _export_rvs_conf_root(
    rvs_source: str,
    rock_dir: str,
    install_base: pathlib.Path | None,
    rvs_binary: str | None = None,
    cmake_executor=None,
) -> None:
    """Point workloads at the first readable RVS ``conf`` tree."""
    install_conf = None
    if install_base is not None:
        install_conf = _find_first(
            install_base,
            "share/rocm-validation-suite/conf",
            cmake_executor,
            want_dir=True,
        )
    for root in (
        _binary_conf_root(rvs_binary),
        install_conf,
        pathlib.Path(rock_dir) / "share" / "rocm-validation-suite" / "conf",
        pathlib.Path(rvs_source) / "rvs" / "conf",
    ):
        if root is not None and _dir_exists(root, cmake_executor):
            os.environ["ROCM_TEST_RVS_CONF_ROOT"] = str(root)
            return


def _export_transferbench_bin(
    rvs_binary: str,
    rock_dir: str,
    compiler_build_dir: str | None,
    install_base: pathlib.Path | None,
    cmake_executor=None,
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
        installed = _find_first(install_base, "bin/TransferBench", cmake_executor)
        if installed is not None:
            candidates.append(installed)
    for candidate in candidates:
        if _is_executable(candidate, cmake_executor):
            os.environ["ROCM_TEST_TRANSFERBENCH_BIN"] = str(candidate)
            return


def export_rvs_env_paths(
    rvs_binary: str | None,
    rvs_source: str | None,
    rock_dir: str,
    *,
    transferbench_binary: str | None = None,
    compiler_build_dir: str | None = None,
    cmake_executor=None,
) -> None:
    """Publish resolved RVS / TransferBench paths into the environment.

    Not used by the tests in this directory, which take the paths from the
    fixtures directly. It exists for other suites that drive the same binaries
    and need the locations this conftest already resolved.
    """
    if rvs_binary:
        os.environ["ROCM_TEST_RVS_BIN"] = rvs_binary

    install_base = _rvs_install_base(rvs_source, cmake_executor)
    if rvs_source:
        _export_rvs_conf_root(rvs_source, rock_dir, install_base, rvs_binary, cmake_executor)

    if transferbench_binary:
        os.environ["ROCM_TEST_TRANSFERBENCH_BIN"] = transferbench_binary
        return

    if rvs_binary:
        _export_transferbench_bin(rvs_binary, rock_dir, compiler_build_dir, install_base, cmake_executor)


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
        if _is_executable(candidate, cmake_executor):
            logger.info("Using TransferBench: %s", candidate)
            return str(candidate)

    # Only clone/build RVS submodules when TransferBench is not already present.
    rvs_source = request.getfixturevalue("rvs_source")
    tb_src = pathlib.Path(rvs_source) / "external" / "TransferBench"
    if not _file_exists(tb_src / "CMakeLists.txt", cmake_executor):
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
