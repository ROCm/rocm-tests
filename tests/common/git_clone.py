# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Validated ``git clone`` / ``git checkout`` shell snippets and value validators."""

from __future__ import annotations

import re
import shlex

import pytest

SAFE_REF_RE = re.compile(r"^[0-9A-Za-z._/-]+$")
SAFE_URL_RE = re.compile(r"^https?://[0-9A-Za-z._~:/?#@!$&'()*+,;=%-]+$")
COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def validate_commit(commit: str, *, subject: str = "commit id") -> str:
    """Return *commit* if it is a shell-safe hex sha, else fail the test."""
    if not SAFE_REF_RE.match(commit):
        pytest.fail(f"{subject} contains unsafe characters: {commit!r}")
    if not COMMIT_RE.match(commit):
        pytest.fail(f"{subject} is not a valid 7-40 char hex sha: {commit!r}")
    return commit


def validate_url(url: str, *, subject: str = "repo URL") -> str:
    """Return *url* if it is a safe http(s) git URL, else fail the test."""
    if not url.startswith("http"):
        pytest.fail(f"{subject} is not an http(s) URL: {url!r}")
    if not SAFE_URL_RE.match(url):
        pytest.fail(f"{subject} contains unsafe characters: {url!r}")
    return url


def validate_path(path: str, *, subject: str = "path") -> str:
    """Return *path* if it is a shell-safe path, else fail the test."""
    if not SAFE_REF_RE.match(path):
        pytest.fail(f"{subject} contains unsafe characters: {path!r}")
    return path


def clone_and_checkout_cmd(url: str, ref: str, dest: str, *, subject: str = "repo") -> str:
    """Return a shell snippet that clones *url* into *dest* and checks out *ref*.

    URL and ref are validated and ``shlex.quote``\\ d; *dest* is emitted verbatim so
    callers may pass a shell expression (e.g. ``'"$work/src"'``). Newline-joined for
    embedding in a larger ``set -e`` script run on the target.
    """
    url = validate_url(url, subject=f"{subject} repo URL")
    ref = validate_commit(ref, subject=f"{subject} commit id")
    short = ref[:7]
    q_url = shlex.quote(url)
    q_ref = shlex.quote(ref)
    return "\n".join(
        (
            f"git clone {q_url} {dest}",
            f"cd {dest}",
            f"git checkout {q_ref}",
            f'git log -1 --format="HEAD is now at %h" | grep -q "HEAD is now at {short}"',
        )
    )
