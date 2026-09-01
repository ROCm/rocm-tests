# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""TorchVision area fixtures.

The ``torchvision_repo`` fixture resolves the repo URL + commit (from the container's
``related_commits`` manifest or env-var overrides), clones the checkout on the host
into the bind-mounted output tree, and returns the container-side path.  The ops
build runs once per test session inside the test body, where ``target_executor``
already runs commands inside the container.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.e2e.ml_frameworks.torchvision._constants import (
    _CONTAINER_WORKSPACE,
    _RESOLVE_TIMEOUT,
    TORCHVISION_COMMIT,
    TORCHVISION_URL,
)

# Shell snippet that reads the related_commits manifest from well-known paths
# inside the container.  Runs via target_executor (i.e. inside the container).
_RELATED_COMMITS_LOOKUP = r"""
for c in /workspace/pytorch/related_commits "$PYTORCH_DIR/related_commits" \
         /opt/pytorch/related_commits /related_commits; do
  [ -f "$c" ] && { f="$c"; break; }
done
if [ -z "$f" ]; then
  tdir=$(python -c 'import os,torch;print(os.path.dirname(os.path.dirname(torch.__file__)))' 2>/dev/null)
  [ -f "$tdir/related_commits" ] && f="$tdir/related_commits"
fi
if [ -z "$f" ]; then
  f=$(find / -maxdepth 6 -name related_commits -type f 2>/dev/null | head -1)
fi
if [ -z "$f" ] || [ ! -f "$f" ]; then echo "__TV_RC_NOTFOUND__"; exit 0; fi
osid=$(. /etc/os-release 2>/dev/null; echo "$ID")
line=$(grep -i torchvision "$f" | grep -i "$osid" | head -1)
[ -z "$line" ] && line=$(grep -i torchvision "$f" | head -1)
[ -z "$line" ] && { echo "__TV_RC_NOENTRY__"; exit 0; }
echo "__TV_URL__:$(echo "$line" | cut -d '|' -f 6 | tr -d '[:space:]')"
echo "__TV_COMMIT__:$(echo "$line" | cut -d '|' -f 5 | tr -d '[:space:]')"
"""


def resolve_url_commit(target_executor) -> tuple[str, str]:
    """Return (url, commit) for torchvision.

    Checks env-var overrides first (``TORCHVISION_URL`` / ``TORCHVISION_COMMIT``),
    then reads the container's ``related_commits`` manifest via target_executor.
    """
    if TORCHVISION_URL and TORCHVISION_COMMIT:
        return TORCHVISION_URL, TORCHVISION_COMMIT

    result = target_executor.run(_RELATED_COMMITS_LOOKUP, timeout=_RESOLVE_TIMEOUT)
    out = f"{result.stdout}\n{result.stderr}"
    if "__TV_RC_NOTFOUND__" in out:
        pytest.fail(
            "related_commits file not found in the container. "
            "Set TORCHVISION_URL and TORCHVISION_COMMIT env vars to bypass the lookup, "
            "or use a PyTorch image that ships the manifest "
            "(e.g. check /workspace/pytorch/related_commits inside the container)."
        )
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
def torchvision_repo(external_build, compiler_build_dir: str) -> str:
    """Clone torchvision on the host; return its container-side path.

    The URL and commit are taken from ``TORCHVISION_URL`` / ``TORCHVISION_COMMIT``
    env vars when set, or resolved later inside the container during the test.
    Cloning only needs the host; the ops build happens inside the test body.
    """
    url = TORCHVISION_URL or "https://github.com/pytorch/vision"
    commit = TORCHVISION_COMMIT or None
    dest = pathlib.Path(compiler_build_dir) / "vision"
    repo = external_build.clone_repo(url, dest, ref=commit)
    external_build.assert_license_present(repo)  # provenance guard
    return f"{_CONTAINER_WORKSPACE}/external/{pathlib.Path(repo).name}"
