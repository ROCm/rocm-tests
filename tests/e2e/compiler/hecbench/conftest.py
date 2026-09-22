# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
conftest.py -- HeCBench area fixtures for tests/e2e/compiler/hecbench/.

Design notes
------------
- The clone is location-transparent: ``external_build.clone_repo`` runs locally
  when no ``--remote-node`` is set and over SSH otherwise.  Like the sibling
  ``hip_examples_repo`` fixture in ``tests/e2e/compiler/conftest.py``, the
  per-benchmark ``make`` + ``make run`` commands are dispatched through
  ``target_executor`` so compile and run happen on the same node/filesystem as
  the checkout for the common single-node case.
- HeCBench is a large monorepo; a full checkout is required because the weekly
  soak run exercises every benchmark.  The ref is overridable via
  ``ROCM_TEST_HECBENCH_REF`` for reproducibility
"""

from __future__ import annotations

import logging
import os
import pathlib

import pytest

logger = logging.getLogger(__name__)

_SUBDIR = "compiler/hecbench"

# HeCBench is cloned at runtime; keep the ref overridable for reproducibility.
_HECBENCH_URL = os.environ.get("ROCM_TEST_HECBENCH_URL", "https://github.com/zjin-lcf/HeCBench")
_HECBENCH_REF = os.environ.get("ROCM_TEST_HECBENCH_REF", "master")


@pytest.fixture(scope="session")
def hecbench_repo(external_build, compiler_build_dir: str) -> pathlib.Path:
    """Clone HeCBench once per session and return the checkout path.

    Args:
        external_build:    Remote-aware clone/build helper from ``builder_plugin``.
        compiler_build_dir: Session-scoped root for build artifacts.

    Returns:
        ``pathlib.Path`` to the HeCBench checkout root (contains ``src/``).
    """
    dest = pathlib.Path(compiler_build_dir) / _SUBDIR / "HeCBench"
    repo = external_build.clone_repo(_HECBENCH_URL, dest, ref=_HECBENCH_REF)
    external_build.assert_license_present(repo)  # provenance guard
    logger.info("HeCBench checkout ready at %s", repo)
    return repo
