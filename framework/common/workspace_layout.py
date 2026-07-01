# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Central path policy for managed local and remote test workspaces."""

from __future__ import annotations

import pathlib
import posixpath

REMOTE_WORKSPACE_DIR = "run-rocm-tests"

_CATEGORY_ROOTS: dict[str, str] = {
    "work": "output",  # compatibility alias
    "output": "output",
    "logs": "output/logs",
    "generated": "output/generated",
    "external": "external",
    "sftp": "sftp",
}


def category_root(category: str) -> str:
    """Return the workspace root directory for a logical category."""
    return _CATEGORY_ROOTS.get(category, category)


def is_managed_remote_path(path: str, workspace_root: str) -> bool:
    """Return True when *path* is already under the managed remote workspace."""
    normalized = posixpath.normpath(pathlib.PurePath(path).as_posix())
    workspace = workspace_root.rstrip("/")
    return normalized == workspace or normalized.startswith(workspace + "/")


def sftp_stage_path(workspace_root: str, local_path: str) -> str:
    """Map a local coordinator path to the remote SFTP staging tree.

    Args:
        workspace_root: Absolute remote workspace root (e.g. ``~/run-rocm-tests``).
        local_path:     Local filesystem path to stage on the remote host.

    Returns:
        Absolute POSIX path under ``<workspace_root>/sftp/`` mirroring the local path.
    """
    resolved = pathlib.Path(local_path).resolve()
    try:
        rel = resolved.relative_to(pathlib.Path.cwd().resolve()).as_posix()
    except ValueError:
        rel = resolved.as_posix().lstrip("/").replace(":", "_")
    return posixpath.join(workspace_root, category_root("sftp"), rel)


def remote_workspace_path(workspace_root: str, path: str | pathlib.PurePath, category: str = "work") -> str:
    """Map a relative/local path into the configured remote workspace category.

    Args:
        workspace_root: Absolute remote workspace root (e.g. ``~/run-rocm-tests``).
        path:           A relative or local path to map into the remote workspace.
        category:       Logical category key (``"work"``, ``"output"``, ``"external"``, etc.).
                        Defaults to ``"work"`` which maps to the ``output/`` subdirectory.

    Returns:
        Normalised absolute POSIX path under ``<workspace_root>/<category_root>/``.
    """
    raw = pathlib.PurePath(path).as_posix()
    if is_managed_remote_path(raw, workspace_root):
        return posixpath.normpath(raw)

    root = category_root(category)
    rel = _relative_part(raw, root) if pathlib.PurePosixPath(raw).is_absolute() else raw
    return posixpath.join(workspace_root, root, _trim_category_prefix(rel, root))


def local_external_path(compiler_build_dir: str, *parts: str) -> pathlib.Path:
    """Return the local external-source cache path for a compiler build root."""
    return _local_output_root(compiler_build_dir) / "external" / pathlib.Path(*parts)


def local_external_clone_dest(dest: str | pathlib.Path, compiler_build_dir: str) -> pathlib.Path:
    """Map legacy clone destinations into the local ``output/external`` tree.

    Args:
        dest:                Requested clone destination path (absolute or relative).
        compiler_build_dir:  Session compiler build root (e.g. ``output/test-binaries/``).

    Returns:
        Resolved ``pathlib.Path`` under the canonical ``output/external/`` tree.
    """
    dest_path = pathlib.Path(dest)
    build_root = pathlib.Path(compiler_build_dir)
    compare_root = build_root.resolve() if dest_path.is_absolute() and not build_root.is_absolute() else build_root
    compare_dest = dest_path.resolve() if dest_path.is_absolute() else dest_path

    try:
        relative = compare_dest.relative_to(compare_root)
    except ValueError:
        if dest_path.is_absolute():
            return dest_path
        relative = dest_path

    return local_external_path(compiler_build_dir, _trim_category_prefix(relative.as_posix(), "external"))


def _local_output_root(compiler_build_dir: str) -> pathlib.Path:
    """Return the local output root, stepping up from ``test-binaries/`` when needed."""
    build_root = pathlib.Path(compiler_build_dir)
    return build_root.parent if build_root.name == "test-binaries" else build_root


def _relative_part(path: str, category: str) -> str:
    """Extract the category-relative portion of an absolute path string."""
    normalized = posixpath.normpath(path).replace(":", "_")
    markers = ("/output/test-binaries/", "/output/external/", "/external/") if category == "external" else ()
    for marker in markers:
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    if category == "output" and "/output/" in normalized:
        return "output/" + normalized.split("/output/", 1)[1]
    if category.startswith("output/"):
        suffix = category.removeprefix("output/")
        marker = f"/output/{suffix}/"
        if marker in normalized:
            return f"{suffix}/" + normalized.split(marker, 1)[1]
    return normalized.lstrip("/")


def _trim_category_prefix(path: str, category: str) -> str:
    """Strip the redundant category prefix from a relative path."""
    rel = posixpath.normpath(path).lstrip("./")
    if rel == ".":
        return ""

    prefixes = {
        "output": ("output/",),
        "external": ("output/test-binaries/", "output/external/", "external/"),
    }.get(category, ())
    if category.startswith("output/"):
        suffix = category.removeprefix("output/")
        prefixes = (f"output/{suffix}/", f"{suffix}/")

    for prefix in prefixes:
        if rel.startswith(prefix):
            return rel[len(prefix) :]
    return rel
