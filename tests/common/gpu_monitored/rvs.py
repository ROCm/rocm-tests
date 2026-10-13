# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RVS (ROCm Validation Suite) detection helpers.

RVS is shipped separately from the ROCm tarball, but is expected to be
preinstalled (its ``rvs`` binary and config tree land under ``rocm_root``)
before the suite runs. This module only *locates* that preinstalled RVS --
it does NOT build RVS from source or install a package. The stack under
test is treated as immutable (see AGENTS.md container conventions); if RVS
is absent the ``rvs_*`` tests surface ``BUILD_FAILED`` so CI flags the
missing component instead of silently source-building a mismatched RVS.
"""

from __future__ import annotations

import os
from pathlib import Path

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.executor_bridge import directory_exists, executable_exists

from .shared_builder import SharedToolBuilder


def is_installed(rocm_root: Path, executor: object | None = None) -> bool:
    rvs = rocm_root / "bin" / "rvs"
    conf = rocm_root / "share" / "rocm-validation-suite" / "conf"
    return executable_exists(executor, rvs) and directory_exists(executor, conf)


def find_bin(config: Config) -> Path | None:
    executor = config.probe_executor
    override = os.environ.get("ROCM_TEST_RVS_BIN", "").strip()
    if override and executable_exists(executor, override):
        return Path(override)
    if is_installed(config.rocm_root, executor):
        return config.rocm_root / "bin" / "rvs"
    return None


# ---------------------------------------------------------------------------
# Config resolution (rvs_config_mapping.csv)
# ---------------------------------------------------------------------------
# Config lookup:
#   * key on the PCI ``<device_id>_<revision>`` (DID_RID);
#   * per-test cell in the CSV names the relative conf subdir:
#       ``./``        -> generic top-level ``conf/<test>.conf``
#       ``./MI300A/`` -> ``conf/MI300A/<test>.conf``
#       (empty)       -> test not applicable for this device -> UNSUPPORTED
#   * device not present in the CSV at all               -> UNSUPPORTED
#   * mapped config file missing on disk                 -> FAIL
# Path components are resolved case-insensitively.


def _install_check(config: Config) -> bool:
    """True when RVS is preinstalled under the ROCm root or exported by pytest.

    Takes the whole config rather than just the root so the probe can follow
    ``probe_executor`` to the node the workload will run on.
    """
    executor = config.probe_executor
    override = os.environ.get("ROCM_TEST_RVS_BIN", "").strip()
    if override and executable_exists(executor, override):
        return True
    return is_installed(config.rocm_root, executor)


def _missing_install(config: Config) -> bool:
    """``SharedToolBuilder`` build_fn. RVS is shipped separately from the
    ROCm tarball but is expected to be preinstalled, so there is nothing to
    build: when the install check fails we report the missing component and
    let the caller surface ``BUILD_FAILED``. We deliberately do NOT
    clone/cmake a source tree or install a package -- the stack under test
    is immutable.
    """
    print(
        f"  [build] rvs: not found under {config.rocm_root} "
        f"(expected {config.rocm_root}/bin/rvs and "
        f"{config.rocm_root}/share/rocm-validation-suite/conf, or "
        f"ROCM_TEST_RVS_BIN from the RVS pytest fixtures). RVS is "
        f"shipped separately from the ROCm tarball; preinstall it under "
        f"the ROCm root or let tests/e2e/system_tools/rvs/conftest.py build it."
    )
    return False


_builder = SharedToolBuilder(
    label="rvs",
    install_check=_install_check,
    build_fn=_missing_install,
)


def build(config: Config) -> bool:
    """Verify RVS is present in the ROCm installation (shared between
    ``rvs_iet_stress`` and ``rvs_tst``). Does NOT build from source."""
    return _builder.build(config)
