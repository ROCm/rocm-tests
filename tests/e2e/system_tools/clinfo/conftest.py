# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- clinfo test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def clinfo_installed(target_executor, rock_dir: str) -> None:
    """Ensure clinfo and OCL ICD are installed on the execution node.

    Detects host OS and installs prerequisites using the appropriate package manager:
    - Debian/Ubuntu: apt-get install clinfo ocl-icd-opencl-dev
    - RHEL/Fedora: dnf install clinfo ocl-icd

    Gracefully skips the test if installation fails or package manager is unavailable.
    """
    # Check if clinfo is already available
    check_cmd = f"[ -x {rock_dir}/bin/clinfo ] || which clinfo >/dev/null 2>&1"
    result = target_executor.run(check_cmd)
    if result.ok:
        return  # Already installed

    # Detect OS type from /etc/os-release
    detect_cmd = "grep -E '^ID_LIKE=|^ID=' /etc/os-release | head -1"
    os_result = target_executor.run(detect_cmd)

    install_cmd = None

    if os_result.ok:
        os_info = os_result.stdout.lower()

        # Check for Debian/Ubuntu-based systems
        if "ubuntu" in os_info:
            install_cmd = "sudo apt-get update -qq && " "sudo apt-get install -y -qq clinfo ocl-icd-opencl-dev"

        # Check for RHEL/Fedora-based systems
        elif "rhel" in os_info or "fedora" in os_info:
            install_cmd = "sudo dnf install -y -q clinfo ocl-icd"

    # If install command was determined, execute it
    if install_cmd:
        result = target_executor.run(install_cmd)
        if result.ok:
            return

    # Final check if clinfo is available
    result = target_executor.run(check_cmd)
    if result.ok:
        return

    # Cannot proceed without prerequisites
    pytest.skip("clinfo prerequisites not available or installation failed")


@pytest.fixture
def clinfo_requires(clinfo_installed: None) -> None:
    """Depend on clinfo installation, making it a prerequisite for the test."""
