# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TorchVision area config and fixtures.

The ``torchvision_repo`` fixture reads the repo URL + commit from the PyTorch
image's ``related_commits`` manifest, clones that commit, and exposes the checkout
inside the container via a bind mount.
"""

from __future__ import annotations

import pathlib
import shlex

import pytest

from framework.common.workspace_layout import REMOTE_WORKSPACE_DIR

# Host output/ tree bind-mounted into the container at a fixed absolute target.
# Not $HOME: the host shell would expand it to the host user's home when building
# the docker command, which need not match the container's home.
_OUTPUT_HOST = pathlib.Path("output").resolve()
_CONTAINER_WORKSPACE = f"/mnt/{REMOTE_WORKSPACE_DIR}"
CONTAINER_MOUNT_FLAGS = f"-v {shlex.quote(str(_OUTPUT_HOST))}:{_CONTAINER_WORKSPACE}"

# GPU UT suite files, restricted to cuda-tagged cases; each runs as its own test.
TEST_FILES = (
    "test/test_functional_tensor.py",
    "test/test_transforms_tensor.py",
)
PYTEST_SELECTOR = "cuda"

# All GPUs on the node (hw.multi_gpu); target_executor owns the allocation.
GPU_COUNT_ARG = "all"

# Whole-suite wall-clock cap (seconds): first-run ops build + one UT run.
RUN_TIMEOUT = 14400.0

# Seconds for the (trivial) in-container related_commits lookup.
_RESOLVE_TIMEOUT = 120.0

# Read the torchvision repo URL (field 6) and commit (field 5) from the PyTorch
# image's related_commits manifest, matched to this OS. Locates the manifest in the
# PyTorch dir (well-known spots + torch install), then a bounded find fallback.
_RELATED_COMMITS_LOOKUP = r"""
tdir=$(python -c 'import os,torch;print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null)
for c in "$PYTORCH_DIR/related_commits" /opt/pytorch/related_commits "$tdir/related_commits" /related_commits; do
  if [ -f "$c" ]; then f="$c"; break; fi
done
if [ -z "$f" ]; then f=$(find / -maxdepth 6 -name related_commits -type f 2>/dev/null | head -1); fi
if [ -z "$f" ] || [ ! -f "$f" ]; then echo "__TV_RC_NOTFOUND__"; exit 0; fi
osid=$(. /etc/os-release 2>/dev/null; echo "$ID")
line=$(grep -i torchvision "$f" | grep -i "$osid" | head -1)
[ -z "$line" ] && line=$(grep -i torchvision "$f" | head -1)
[ -z "$line" ] && { echo "__TV_RC_NOENTRY__"; exit 0; }
echo "__TV_URL__:$(echo "$line" | cut -d '|' -f 6 | tr -d '[:space:]')"
echo "__TV_COMMIT__:$(echo "$line" | cut -d '|' -f 5 | tr -d '[:space:]')"
"""


def _resolve_url_commit(target_executor) -> tuple[str, str]:
    """Return (url, commit) from the container's related_commits; fail if absent."""
    result = target_executor.run(_RELATED_COMMITS_LOOKUP, timeout=_RESOLVE_TIMEOUT)
    out = f"{result.stdout}\n{result.stderr}"
    if "__TV_RC_NOTFOUND__" in out:
        pytest.fail("related_commits file is missing in the container -- use a PyTorch image that ships it.")
    if "__TV_RC_NOENTRY__" in out:
        pytest.fail(f"related_commits has no torchvision entry for this OS:\n{out[-2000:]}")
    url = commit = ""
    for token in (ln.strip() for ln in out.splitlines()):
        if token.startswith("__TV_URL__:"):
            url = token.split(":", 1)[1].strip()
        elif token.startswith("__TV_COMMIT__:"):
            commit = token.split(":", 1)[1].strip()
    if not url.startswith("http") or not commit:
        pytest.fail(f"Could not resolve torchvision url/commit from related_commits:\n{out[-2000:]}")
    return url, commit


@pytest.fixture
def torchvision_repo(external_build, compiler_build_dir: str, target_executor) -> str:
    """Clone torchvision (commit from the container manifest) and return its container path.

    Reads the commit from the container's related_commits, clones on the host into the
    bind-mounted output tree, and returns the checkout path inside the container.
    """
    url, commit = _resolve_url_commit(target_executor)
    dest = pathlib.Path(compiler_build_dir) / "vision"
    repo = external_build.clone_repo(url, dest, ref=commit)
    external_build.assert_license_present(repo)  # provenance guard
    return f"{_CONTAINER_WORKSPACE}/external/{pathlib.Path(repo).name}"
