# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
_constants.py -- Package list for cmake path verifier tests.

Each entry in PACKAGES_TO_VERIFY is one parametrize ID that maps to a
sub-directory under ``{rock_dir}/lib/cmake/``. The ``hip`` entry is
special: the HIP-permitted fallback pattern (HINTS ${ROCM_PATH} PATHS
"/opt/rocm") is allowed and must not be flagged as a violation.
"""

from __future__ import annotations

# Packages installed by the session fixture before verification runs.
# On RHEL/SLES the -devel variant is installed; on Ubuntu the plain name is used.
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
    "hsakmt",
    "miopengemm",
    "rocm-opencl",
]

# Packages whose cmake config directories must exist and must not contain
# any line that hardcodes /opt/rocm (other than the hip-permitted fallback).
PACKAGES_TO_VERIFY: list[str] = [
    "hip",
    "amd_comgr",
    "amd-dbgapi",
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
    "hsakmt",
    "miopengemm",
    "rocm-opencl",
]

# Regex fragments used by hardcode_verifier() in the test module.
# Defined here so both the test and any future helpers share one source of truth.
#
# HARDCODE_PATTERN:     matches any non-comment line that references /opt/rocm
#                       in a cmake file — indicates a hardcoded install prefix.
# HIP_ALLOWED_PATTERN:  the one permitted exception in HIP cmake files:
#                       HINTS ${ROCM_PATH} PATHS "/opt/rocm" is the canonical
#                       "prefer ROCM_PATH, fall back to /opt/rocm" idiom and
#                       is explicitly allowed per the ROCm cmake guidelines.
HARDCODE_PATTERN: str = r'^\s*[^#\s].*/opt/rocm/*[^-|_|\w]\)?'
HIP_ALLOWED_PATTERN: str = (
    r'^\s*[^#\s].*HINTS\s+\$\{ROCM_PATH\}\s+PATHS\s+"\/opt\/rocm\/*[^-|_|\w]\)?'
)
