#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""installer.py -- Build and install CRIU + the AMD amdgpu CRIU plugin on a test node.

Provisions CRIU so the ``criu_runtime`` fixture finds it ready. The ``criu_runtime`` fixture
clones CRIU via ``external_build.clone_repo()`` and passes the checkout with ``--src-dir``; this
installer then only builds and installs. Run standalone (no ``--src-dir``) it clones the source
itself so it remains usable from fleet provisioning.

What it does:
    1. Install CRIU build prerequisites (apt / dnf / zypper auto-detected).
    2. Use the ``--src-dir`` checkout, or ``git clone`` CRIU at the requested tag (default v4.1).
    3. ``make -j`` && ``sudo make install-criu``  (installs the binary to /usr/local/sbin; no man pages).
    4. ``sudo make amdgpu_plugin``.
    5. ``sudo mkdir -p /usr/lib/criu`` && copy ``amdgpu_plugin.so`` there.
    6. ``sudo criu check``  (should print "Looks good").

Usage:
    python3 installer.py [CRIU_VERSION_TAG]
    python3 installer.py --src-dir /path/to/criu v4.1
    CRIU_VERSION=v4.1 python3 installer.py

Requires: sudo privileges, a C toolchain, network access (for clone/prereqs). Linux only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI arg or environment)
# ---------------------------------------------------------------------------

DEFAULT_VERSION = "v4.1"
CRIU_REPO = os.environ.get("CRIU_REPO", "https://github.com/checkpoint-restore/criu.git")
CRIU_SRC_DIR = os.environ.get("CRIU_SRC_DIR", os.path.join(os.path.expanduser("~"), "criu_src"))
CRIU_PLUGIN_DIR = os.environ.get("CRIU_PLUGIN_DIR", "/usr/lib/criu")

# /usr/local/sbin must be on PATH for the elevated `criu` process (sudo resets env).
CRIU_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/usr/bin:/bin"

# Prerequisite package sets per package manager.
_PREREQS = {
    "apt-get": [
        "git",
        "build-essential",
        "libprotobuf-dev",
        "libprotobuf-c-dev",
        "protobuf-c-compiler",
        "protobuf-compiler",
        "python3-protobuf",
        "libnl-3-dev",
        "libnet-dev",
        "libcap-dev",
        "libbsd-dev",
        "libgnutls28-dev",
        "uuid-dev",
        "pkg-config",
        "libdrm-dev",
    ],
    "dnf": [
        "git",
        "gcc",
        "make",
        "protobuf-devel",
        "protobuf-c-devel",
        "protobuf-c-compiler",
        "protobuf-compiler",
        "python3-protobuf",
        "libnl3-devel",
        "libnet-devel",
        "libcap-devel",
        "libbsd-devel",
        "gnutls-devel",
        "libuuid-devel",
        "pkgconfig",
        "libdrm-devel",
    ],
    "zypper": [
        "git",
        "gcc",
        "make",
        "protobuf-devel",
        "libprotobuf-c-devel",
        "protobuf-c",
        "libnl3-devel",
        "libnet-devel",
        "libcap-devel",
        "libbsd-devel",
        "libgnutls-devel",
        "libuuid-devel",
        "pkg-config",
        "libdrm-devel",
    ],
}


def _log(message: str) -> None:
    """Print a namespaced progress line."""
    print(f"\n[criu-installer] {message}", flush=True)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run *cmd*, streaming output; raise CalledProcessError on non-zero exit."""
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _sudo() -> list[str]:
    """Return the elevated-command prefix: empty when already root, else ``["sudo"]``.

    Elevated commands drop ``sudo`` when running as root (e.g. inside a container, where ``sudo``
    is frequently not installed). Computed lazily so importing this module on a non-Linux host
    (for collection/lint) does not call the Linux-only ``os.geteuid``.
    """
    return [] if getattr(os, "geteuid", lambda: 1)() == 0 else ["sudo"]


def install_prereqs() -> None:
    """Install CRIU build prerequisites with the first available package manager."""
    if shutil.which("apt-get"):
        _log("Installing prerequisites with apt-get (Ubuntu/Debian)")
        _run([*_sudo(), "apt-get", "update"])
        _run([*_sudo(), "apt-get", "install", "-y", *_PREREQS["apt-get"]])
    elif shutil.which("dnf"):
        _log("Installing prerequisites with dnf (RHEL/CentOS/Fedora)")
        _run([*_sudo(), "dnf", "install", "-y", *_PREREQS["dnf"]])
    elif shutil.which("zypper"):
        _log("Installing prerequisites with zypper (SLES/openSUSE)")
        _run([*_sudo(), "zypper", "--non-interactive", "install", *_PREREQS["zypper"]])
    else:
        _log("WARNING: no supported package manager (apt/dnf/zypper) found.")
        _log("Install CRIU build prerequisites manually, then re-run.")


def build_and_install(version: str, src_dir: str | None = None) -> None:
    """Build CRIU + the amdgpu plugin from *src_dir* and install them.

    When *src_dir* is given (the fixture clones via ``external_build.clone_repo``), it is built
    as-is. Otherwise CRIU *version* is cloned into ``CRIU_SRC_DIR`` first so the installer also
    works standalone.
    """
    if src_dir:
        criu_src = src_dir
        _log(f"Building CRIU from provided checkout {criu_src}")
    else:
        criu_src = CRIU_SRC_DIR
        _log(f"Cloning CRIU {version} into {criu_src}")
        if os.path.exists(criu_src):
            shutil.rmtree(criu_src)
        _run(["git", "clone", "-b", version, CRIU_REPO, criu_src])

    _log("Building CRIU (make -j)")
    _run(["make", f"-j{os.cpu_count() or 1}"], cwd=criu_src)

    # 'install-criu' installs just the criu binary (+ plugins) to /usr/local/sbin and skips the
    # 'install-man' target, which rebuilds man pages via the asciidoc Python module -- often absent
    # in container venvs (e.g. /opt/venv) and not needed to run CRIU.
    _log("Installing CRIU (sudo make install-criu -> /usr/local/sbin)")
    _run([*_sudo(), "make", "install-criu"], cwd=criu_src)

    _log("Building the amdgpu CRIU plugin (sudo make amdgpu_plugin)")
    _run([*_sudo(), "make", "amdgpu_plugin"], cwd=criu_src)

    _log(f"Installing amdgpu_plugin.so into {CRIU_PLUGIN_DIR}")
    _run([*_sudo(), "mkdir", "-p", CRIU_PLUGIN_DIR])

    plugin_so = _find_plugin_so(criu_src)
    if not plugin_so:
        _log("ERROR: amdgpu_plugin.so was not produced by the build.")
        sys.exit(1)
    _run([*_sudo(), "cp", plugin_so, os.path.join(CRIU_PLUGIN_DIR, "amdgpu_plugin.so")])


def _find_plugin_so(criu_src: str) -> str | None:
    """Return the built amdgpu_plugin.so path under *criu_src*/plugins/amdgpu, or None."""
    amdgpu_dir = os.path.join(criu_src, "plugins", "amdgpu")
    for root, _dirs, files in os.walk(amdgpu_dir):
        if "amdgpu_plugin.so" in files:
            return os.path.join(root, "amdgpu_plugin.so")
    return None


def verify() -> None:
    """Run ``sudo criu check`` (with /usr/local/sbin on PATH); exit 1 on failure."""
    _log("Verifying installation with 'sudo criu check'")
    result = subprocess.run(
        [*_sudo(), "env", f"PATH={CRIU_PATH}", "criu", "check"],
        check=False,
    )
    if result.returncode == 0:
        _log("SUCCESS: 'criu check' reports the environment looks good.")
        _log(f"amdgpu plugin: {os.path.join(CRIU_PLUGIN_DIR, 'amdgpu_plugin.so')}")
    else:
        _log("WARNING: 'criu check' reported problems. Review the output above.")
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    """Install prerequisites, build/install CRIU + amdgpu plugin, and verify."""
    parser = argparse.ArgumentParser(description="Build and install CRIU + the AMD amdgpu CRIU plugin.")
    parser.add_argument(
        "version",
        nargs="?",
        default=os.environ.get("CRIU_VERSION", DEFAULT_VERSION),
        help=f"CRIU git tag to clone when --src-dir is not given (default: {DEFAULT_VERSION}).",
    )
    parser.add_argument(
        "--src-dir",
        default=None,
        help="Existing CRIU checkout to build (skips cloning). Supplied by the criu_runtime fixture.",
    )
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        _log(f"ERROR: CRIU is Linux-only; cannot install on {sys.platform!r}.")
        return 1

    _log(f"CRIU installer starting (version={args.version}, src_dir={args.src_dir or '<clone>'})")
    try:
        install_prereqs()
        build_and_install(args.version, src_dir=args.src_dir)
        verify()
    except subprocess.CalledProcessError as exc:
        _log(f"ERROR: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        return exc.returncode or 1
    _log("Done. Ensure passwordless sudo is configured for the CI/test user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
