# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS invocation and config-resolution helpers.

Shared across all RVS module test suites. Provides ``run_rvs`` for executing a
single RVS config action, and ``resolve_conf_dir`` for looking up the
per-device/module configuration directory from the qualification matrix CSV.
"""

from __future__ import annotations

import logging
import pathlib
import shlex

import pytest

from tests.common.gpu_pci_map import ConfDirUnresolvedError, detect_device_key
from tests.common.rvs_config_map import conf_dir_for_module, covers_module, device_name, is_known_device

logger = logging.getLogger(__name__)

RVS_DEBUG_LEVEL = 3


def run_rvs(executor, rvs_env: str, binary: str, conf_path: str, timeout: float, label: str) -> tuple[str, int]:
    """Execute one RVS config and return the combined output and exit code.

    Args:
        executor:   Any executor with a ``.run()`` method.
        rvs_env:    Shell ``VAR=value`` assignments prefixed with ``env``.
        binary:     Absolute path to the ``rvs`` binary.
        conf_path:  Path to the ``.conf`` file (already shell-safe).
        timeout:    Seconds before the run is abandoned.
        label:      Short human-readable label for log messages.

    Returns:
        ``(combined_stdout_stderr, exit_code)``
    """
    cmd = f"env {rvs_env} {binary} -c {conf_path} -d {RVS_DEBUG_LEVEL}"
    logger.info("Running %s: %s", label, cmd)
    result = executor.run(cmd, timeout=timeout)
    return (result.stdout or "") + (result.stderr or ""), result.exit_code


def resolve_conf_dir(device_key: str, module: str) -> str:
    """Return the conf subdirectory for *module* on the detected GPU.

    Consults the RVS qualification matrix (``rvs_config_mapping.csv``) to find
    which config directory the given module uses for this GPU. Returns an empty
    string when the module uses the top-level conf directory. Calls
    ``pytest.skip`` when the module is not qualified on this GPU.

    Args:
        device_key: ``<did>_<rid>`` key from :func:`detect_rvs_device_key`.
        module:     RVS module name (e.g. ``"iet"``, ``"mem"``).

    Returns:
        Config subdirectory name, or ``""`` for the top-level conf.
    """
    if not covers_module(module):
        pytest.skip(f"RVS {module} is not a column in the RVS config mapping")

    conf_dir = conf_dir_for_module(device_key, module)
    if conf_dir is None:
        pytest.skip(f"RVS {module} is not qualified on {device_name(device_key)} ({device_key})")
    return conf_dir


def detect_rvs_device_key(executor=None) -> str:
    """Detect the GPU device key used to index the RVS qualification matrix.

    Calls ``pytest.fail`` when the GPU cannot be detected or is not in the
    qualification matrix at all (a string of per-module skips would be
    misleading — a single fail is clearer).

    Args:
        executor: Remote executor; ``None`` means local.

    Returns:
        ``<did>_<rid>`` key string.
    """
    try:
        key = detect_device_key(cmake_executor=executor)
    except ConfDirUnresolvedError as exc:
        pytest.fail(str(exc))

    if not is_known_device(key):
        pytest.fail(f"GPU {key} has no row in the RVS config mapping; no module is qualified on it")
    return key


def binary_conf_root(rvs_binary: str) -> pathlib.Path | None:
    """Return the ``conf`` tree that sits beside *rvs_binary*.

    Packaged and locally built RVS both live at ``<base>/bin/rvs`` with their
    configs under the matching ``<base>/share``, so the root is derived from the
    resolved binary path rather than hard-coded.

    Args:
        rvs_binary: Absolute path to the ``rvs`` binary.

    Returns:
        Path to the ``conf`` directory, or ``None`` when *rvs_binary* is empty.
    """
    if not rvs_binary:
        return None
    base = pathlib.Path(rvs_binary).resolve().parent.parent
    return base / "share" / "rocm-validation-suite" / "conf"


def build_rvs_env(rvs_binary: str, rock_dir: str, ld_library_path: str) -> str:
    """Return ``VAR=value`` env assignments for running ``rvs`` via ``env``.

    ``libomp.so`` lives under ``llvm/lib`` and the binary may sit in an
    ``extras-N`` tree with its own ``lib``; both must be on ``LD_LIBRARY_PATH``
    or RVS fails to load its shared dependencies.

    Args:
        rvs_binary:       Absolute path to the ``rvs`` binary.
        rock_dir:         ROCm install root.
        ld_library_path:  Existing ``LD_LIBRARY_PATH`` value (may be empty).

    Returns:
        Shell ``env`` prefix string, e.g. ``"ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=..."``.
    """
    rvs_base = pathlib.Path(rvs_binary).resolve().parent.parent
    libs = [str(rvs_base / "lib"), f"{rock_dir}/lib", f"{rock_dir}/llvm/lib"]
    if ld_library_path:
        libs.append(ld_library_path)
    assignments = {
        "ROCM_PATH": rock_dir,
        "RVS_PATH": str(rvs_base),
        "LD_LIBRARY_PATH": ":".join(libs),
    }
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in assignments.items())
