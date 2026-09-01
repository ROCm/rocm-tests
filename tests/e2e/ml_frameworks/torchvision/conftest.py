# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TorchVision area fixtures.

The ``torchvision_repo`` fixture clones torchvision on the host into the bind-mounted
output tree and returns its container-side path.  The clone URL and commit are taken
from ``TORCHVISION_URL`` / ``TORCHVISION_COMMIT`` env vars when set; otherwise the
upstream default branch is used.  The ops build runs inside the test body via
``target_executor``, which executes inside the container.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.e2e.ml_frameworks.torchvision._constants import (
    _CONTAINER_WORKSPACE,
    TORCHVISION_COMMIT,
    TORCHVISION_URL,
)

_DEFAULT_TORCHVISION_URL = "https://github.com/pytorch/vision"


@pytest.fixture
def torchvision_repo(external_build, compiler_build_dir: str) -> str:
    """Clone torchvision on the host; return its container-side path.

    The URL and commit are taken from ``TORCHVISION_URL`` / ``TORCHVISION_COMMIT``
    env vars when set, otherwise the upstream default branch HEAD is used.
    The ops build happens inside the test body via ``target_executor``.
    """
    url = TORCHVISION_URL or _DEFAULT_TORCHVISION_URL
    commit = TORCHVISION_COMMIT or None
    dest = pathlib.Path(compiler_build_dir) / "vision"
    repo = external_build.clone_repo(url, dest, ref=commit)
    external_build.assert_license_present(repo)  # provenance guard
    return f"{_CONTAINER_WORKSPACE}/external/{pathlib.Path(repo).name}"
