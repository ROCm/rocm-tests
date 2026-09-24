# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""_constants.py -- Constants for cmake_path_verifier tests."""

from __future__ import annotations

# Packages installed before verification runs.
# On RHEL and SLES the -devel variant of each name is installed.
PACKAGES_TO_INSTALL: list[str] = [
    "rocblas",
    "rocsparse",
    "rocfft",
    "rocalution",
    "hipblas",
    "hipsparse",
    "hipfft",
    "hipcub",
    "rocthrust",
    "hiprand",
    "rocprim",
    "rccl",
    "rocrand",
]

# Packages whose cmake directories must exist regardless of installation status.
# Everything else is discovered dynamically from {rocm_dir}/lib/cmake/.
PACKAGES_ALWAYS_VERIFY: list[str] = ["hip", "amd_comgr", "amd-dbgapi"]

# Regex fragments for the hardcode check.
# HARDCODE_PATTERN:    non-comment line that references /opt/rocm — hardcoded install prefix.
# HIP_ALLOWED_PATTERN: permitted HIP exception — HINTS ${ROCM_PATH} PATHS "/opt/rocm".
HARDCODE_PATTERN: str = r"^\s*[^#\s].*/opt/rocm/*[^-|_|\w]\)?"
HIP_ALLOWED_PATTERN: str = r'^\s*[^#\s].*HINTS\s+\$\{ROCM_PATH\}\s+PATHS\s+"\/opt\/rocm"\/*[^-|_|\w]\)?'
