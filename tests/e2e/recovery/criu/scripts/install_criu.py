#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""install_criu.py -- Reusable installer for CRIU + the AMD amdgpu CRIU plugin.

Provisions CRIU on a GPU node so the ``criu_runtime`` fixture finds it ready. Run manually or
from fleet provisioning; the fixture also invokes it on demand.

What it does:
    1. Install CRIU build prerequisites (apt / dnf / zypper auto-detected).
    2. git clone CRIU at the requested tag (default v4.1).
    3. ``make -j`` && ``sudo make install``            (installs to /usr/local/sbin).
    4. ``sudo make amdgpu_plugin``.
    5. ``sudo mkdir -p /usr/lib/criu`` && copy ``amdgpu_plugin.so`` there.
    6. ``sudo criu check``  (should print "Looks good").

Usage:
    python3 install_criu.py [CRIU_VERSION_TAG]
    CRIU_VERSION=v4.1 python3 install_criu.py
    CRIU_SRC_DIR=/opt/src python3 install_criu.py v4.1

Requires: sudo privileges, git, a C toolchain, and network access. Linux only.
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
        "pkg-config",
        "libdrm-dev",
        "asciidoc",
        "xmlto",
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
        "pkgconfig",
        "libdrm-devel",
        "asciidoc",
        "xmlto",
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
        "pkg-config",
        "libdrm-devel",
        "asciidoc",
        "xmlto",
    ],
}


def _log(message: str) -> None:
    """Print a namespaced progress line."""
    print(f"\n[install_criu] {message}", flush=True)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    """Run *cmd*, streaming output; raise CalledProcessError on non-zero exit."""
    _log("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def install_prereqs() -> None:
    """Install CRIU build prerequisites with the first available package manager."""
    if shutil.which("apt-get"):
        _log("Installing prerequisites with apt-get (Ubuntu/Debian)")
        _run(["sudo", "apt-get", "update"])
        _run(["sudo", "apt-get", "install", "-y", *_PREREQS["apt-get"]])
    elif shutil.which("dnf"):
        _log("Installing prerequisites with dnf (RHEL/CentOS/Fedora)")
        _run(["sudo", "dnf", "install", "-y", *_PREREQS["dnf"]])
    elif shutil.which("zypper"):
        _log("Installing prerequisites with zypper (SLES/openSUSE)")
        _run(["sudo", "zypper", "--non-interactive", "install", *_PREREQS["zypper"]])
    else:
        _log("WARNING: no supported package manager (apt/dnf/zypper) found.")
        _log("Install CRIU build prerequisites manually, then re-run.")


def build_and_install(version: str) -> None:
    """Clone CRIU at *version*, build it, and install CRIU + the amdgpu plugin."""
    _log(f"Cloning CRIU {version} into {CRIU_SRC_DIR}")
    if os.path.exists(CRIU_SRC_DIR):
        shutil.rmtree(CRIU_SRC_DIR)
    _run(["git", "clone", "-b", version, CRIU_REPO, CRIU_SRC_DIR])

    _log("Building CRIU (make -j)")
    _run(["make", f"-j{os.cpu_count() or 1}"], cwd=CRIU_SRC_DIR)

    _log("Installing CRIU (sudo make install -> /usr/local/sbin)")
    _run(["sudo", "make", "install"], cwd=CRIU_SRC_DIR)

    _log("Building the amdgpu CRIU plugin (sudo make amdgpu_plugin)")
    _run(["sudo", "make", "amdgpu_plugin"], cwd=CRIU_SRC_DIR)

    _log(f"Installing amdgpu_plugin.so into {CRIU_PLUGIN_DIR}")
    _run(["sudo", "mkdir", "-p", CRIU_PLUGIN_DIR])

    plugin_so = _find_plugin_so()
    if not plugin_so:
        _log("ERROR: amdgpu_plugin.so was not produced by the build.")
        sys.exit(1)
    _run(["sudo", "cp", plugin_so, os.path.join(CRIU_PLUGIN_DIR, "amdgpu_plugin.so")])


def _find_plugin_so() -> str | None:
    """Return the built amdgpu_plugin.so path under plugins/amdgpu, or None."""
    amdgpu_dir = os.path.join(CRIU_SRC_DIR, "plugins", "amdgpu")
    for root, _dirs, files in os.walk(amdgpu_dir):
        if "amdgpu_plugin.so" in files:
            return os.path.join(root, "amdgpu_plugin.so")
    return None


def verify() -> None:
    """Run ``sudo criu check`` (with /usr/local/sbin on PATH); exit 1 on failure."""
    _log("Verifying installation with 'sudo criu check'")
    result = subprocess.run(
        ["sudo", "env", f"PATH={CRIU_PATH}", "criu", "check"],
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
    parser = argparse.ArgumentParser(description="Install CRIU + the AMD amdgpu CRIU plugin.")
    parser.add_argument(
        "version",
        nargs="?",
        default=os.environ.get("CRIU_VERSION", DEFAULT_VERSION),
        help=f"CRIU git tag to build (default: {DEFAULT_VERSION}).",
    )
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        _log(f"ERROR: CRIU is Linux-only; cannot install on {sys.platform!r}.")
        return 1

    _log(f"CRIU installer starting (version={args.version})")
    try:
        install_prereqs()
        build_and_install(args.version)
        verify()
    except subprocess.CalledProcessError as exc:
        _log(f"ERROR: command failed with exit code {exc.returncode}: {' '.join(exc.cmd)}")
        return exc.returncode or 1
    _log("Done. Ensure passwordless sudo is configured for the CI/test user.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
