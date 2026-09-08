# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apex area fixtures.

Provides the ``apex_repo`` session fixture, which clones Apex and exposes the
checkout inside the container via a bind mount.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.e2e.ml_frameworks.apex._constants import _CONTAINER_WORKSPACE, APEX_COMMIT, APEX_URL


@pytest.fixture(scope="session")
def apex_repo(external_build, compiler_build_dir: str) -> str:
    """Clone Apex once per session; return the checkout path inside the container.

    Clones at ref ``APEX_COMMIT`` (else the default branch) into the bind-mounted
    output tree, so the returned path resolves inside the container.
    """
    dest = pathlib.Path(compiler_build_dir) / "apex"
    repo = external_build.clone_repo(APEX_URL, dest, ref=APEX_COMMIT or None)
    external_build.assert_license_present(repo)  # provenance guard
    return f"{_CONTAINER_WORKSPACE}/external/{pathlib.Path(repo).name}"
