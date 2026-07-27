# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared, env-configurable parameters and command builders for the JAX UT port.

Environment overrides (framework config-cascade idiom):
    ROCM_TEST_JAX_DIR   JAX source checkout on the execution node
                        (default ``/workspace/jax``).  This is a container
                        assumption; each test skips gracefully when the
                        directory is absent on the target node.
    JAX_NUM_GPUS        GPUs to request for the ``jax_ut`` suite (default 1).
                        ``1`` selects single-GPU mode (``hw.gpu``); ``>1``
                        selects multi-GPU mode (``hw.multi_gpu``) and adds the
                        multi-GPU sub-suite to the command list.
    JAX_VERSION         Skip node JAX-version probing; use this value instead.
    JAX_ROCM_VERSION    Skip node ROCm-version probing; use this value instead.
    JAX_UT_TIMEOUT      ``jax_ut`` run cap in seconds (default 14400 = 4h).
    JAX_RNN_UT_TIMEOUT  ``jax_rnn_ut`` run cap in seconds (default 1800).
    JAX_FP8_UT_TIMEOUT  ``jax_fp8_ut`` run cap in seconds (default 900).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re

from packaging.version import Version

# JAX source checkout on the execution node (container assumption).
JAX_DIR = os.environ.get("ROCM_TEST_JAX_DIR", "/workspace/jax")

# GPUs to request for jax_ut; 1 (default) = single-GPU mode, >1 = multi-GPU mode.
NUM_GPUS = int(os.environ.get("JAX_NUM_GPUS", "1"))
IS_SINGLE_GPU = NUM_GPUS <= 1

# JAX >= 0.9.1 uses the consolidated ``ci/run_pytest_rocm.sh`` entrypoint.
NEW_PATH_MIN_VERSION = Version("0.9.1")
# ROCm 7.0 relocated the legacy per-GPU run scripts under a plugin folder.
ROCM_7 = Version("7.0")

# Optional overrides to bypass on-node version probing.
JAX_VERSION_OVERRIDE = os.environ.get("JAX_VERSION", "").strip()
ROCM_VERSION_OVERRIDE = os.environ.get("JAX_ROCM_VERSION", "").strip()

# Per-suite run caps (seconds). jax_ut runs the full JAX UT suite (hours).
JAX_UT_TIMEOUT = float(os.environ.get("JAX_UT_TIMEOUT", "14400"))
JAX_RNN_UT_TIMEOUT = float(os.environ.get("JAX_RNN_UT_TIMEOUT", "1800"))
JAX_FP8_UT_TIMEOUT = float(os.environ.get("JAX_FP8_UT_TIMEOUT", "900"))

# Cap for lightweight setup steps (pip installs, mkdir) in the new-path flow.
SETUP_TIMEOUT = float(os.environ.get("JAX_UT_SETUP_TIMEOUT", "1800"))

# ``jax`` line in ``pip list`` output, e.g. "jax                      0.4.30".
_JAX_VERSION_RE = re.compile(r"^jax\s+v?(\d+\.\d+\.\d+)", re.MULTILINE)
# First MAJOR.MINOR[.PATCH] token in a version string.
_VERSION_RE = re.compile(r"(\d+)\.(\d+)(?:\.\d+)?")


@dataclass(frozen=True)
class JaxCommand:
    """One shell step in a ``jax_ut`` run.

    Attributes:
        label:    Short human-readable label used in diagnostics.
        command:  Shell command to run relative to ``JAX_DIR``.
        validate: ``True`` for a primary test command whose result determines the
                  suite outcome; ``False`` for setup steps (pip installs, mkdir).
    """

    label: str
    command: str
    validate: bool = True


def parse_jax_version(pip_list_output: str) -> Version | None:
    """Extract the installed JAX version from ``pip list`` output.

    Args:
        pip_list_output: stdout of ``pip list`` (or ``pip list | grep jax``).

    Returns:
        Parsed :class:`~packaging.version.Version` for the ``jax`` package, or
        ``None`` when no ``jax`` entry is found.  ``jaxlib`` / ``jax-cuda`` lines
        are intentionally not matched.
    """
    match = _JAX_VERSION_RE.search(pip_list_output or "")
    return Version(match.group(1)) if match else None


def parse_rocm_version(text: str) -> Version | None:
    """Extract a MAJOR.MINOR ROCm version from a version string.

    Accepts output such as ``hipconfig --version`` (``6.4.43483-...``) or the
    contents of a ROCm ``.info/version`` file.

    Args:
        text: Raw version text from the execution node.

    Returns:
        ``Version("MAJOR.MINOR")``, or ``None`` when no version token is found.
    """
    match = _VERSION_RE.search(text or "")
    if not match:
        return None
    return Version(f"{match.group(1)}.{match.group(2)}")


def jax_ut_commands(
    jax_version: Version,
    rocm_version: Version | None,
    *,
    single_gpu: bool,
    python: str = "python3",
) -> list[JaxCommand]:
    """Build the ordered ``jax_ut`` command list for the detected environment.

    * **JAX >= 0.9.1** -> the consolidated ``ci/run_pytest_rocm.sh`` entrypoint,
      which itself selects single- vs multi-GPU sub-suites from the visible GPU
      count.  A few pip prerequisites are installed first.
    * **JAX < 0.9.1** -> the legacy ``build/rocm/run_single_gpu.py`` /
      ``run_multi_gpu`` scripts, with the ROCm-7 plugin-folder relocation:
      ROCm < 7.0 uses a bash multi-GPU script; ROCm == 7.0 keeps the scripts at
      the tree root (plugin folder ``"."``); ROCm > 7.0 (or unknown) nests them
      under ``jax_rocm_plugin/``.  When multi-GPU is requested the multi-GPU
      script runs first, then the single-GPU script
    Args:
        jax_version:  Installed JAX version (from :func:`parse_jax_version`).
        rocm_version: Installed ROCm MAJOR.MINOR version (from
                      :func:`parse_rocm_version`); ``None`` when undetectable, in
                      which case the newest (>7.0) legacy layout is assumed.
        single_gpu:   ``True`` for single-GPU mode; ``False`` also runs the
                      multi-GPU sub-suite (legacy path only — the new path
                      auto-detects visible GPUs).
        python:       Python interpreter to invoke on the execution node.

    Returns:
        Ordered list of :class:`JaxCommand` steps.
    """
    if jax_version >= NEW_PATH_MIN_VERSION:
        # The consolidated script auto-detects visible GPUs and runs the
        # single- (and, when >1 GPU is visible, multi-) GPU sub-suites itself.
        return [
            JaxCommand(
                "install-test-reqs",
                f"{python} -m pip install -r build/test-requirements.txt",
                validate=False,
            ),
            JaxCommand(
                "install-pytest-extras",
                f"{python} -m pip install pytest-html pytest-csv uv pytest-json-report",
                validate=False,
            ),
            JaxCommand("reset-dist", "rm -rf dist && mkdir -p dist", validate=False),
            JaxCommand(
                "jax_ut",
                'PYTEST_ADDOPTS="-vv -s --tb=long" JAXCI_PYTHON="$(command -v python)" ./ci/run_pytest_rocm.sh',
            ),
        ]

    # Legacy path (JAX < 0.9.1).
    cmds: list[JaxCommand] = []
    if rocm_version is not None and rocm_version < ROCM_7:
        if not single_gpu:
            cmds.append(
                JaxCommand(
                    "jax_ut_mgpu",
                    "chmod +x ./build/rocm/run_multi_gpu.sh && bash ./build/rocm/run_multi_gpu.sh -c",
                )
            )
        cmds.append(JaxCommand("jax_ut_sgpu", f"{python} ./build/rocm/run_single_gpu.py -c"))
        return cmds

    # ROCm == 7.0 keeps the scripts at the tree root ("."); ROCm > 7.0 (or an
    # undetectable version) nests them under jax_rocm_plugin/.
    plugin = "." if rocm_version == ROCM_7 else "jax_rocm_plugin"
    if not single_gpu:
        cmds.append(JaxCommand("jax_ut_mgpu", f"{python} {plugin}/build/rocm/run_multi_gpu.py -c"))
    cmds.append(JaxCommand("jax_ut_sgpu", f"{python} {plugin}/build/rocm/run_single_gpu.py -c"))
    return cmds
