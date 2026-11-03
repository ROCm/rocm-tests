# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared parameters for the functional QMCPACK HIP suite.

Imported by ``conftest.py`` (clone + build) and ``test_qmcpack_functional.py``
(ctest) so the pin and the timeouts cannot drift.

Environment overrides:
    QMCPACK_URL                 git URL to clone. Default QMCPACK/qmcpack.
    QMCPACK_REF                 git ref (branch/tag/SHA). Default ``develop``.
    QMCPACK_CTEST_TIMEOUT       per-test ``ctest --timeout`` in seconds.
    QMCPACK_CTEST_RUN_TIMEOUT   outer wall-clock cap for one configuration's ctest.
"""

import os

# Upstream quantum Monte Carlo application. Cloned at runtime, not vendored.
QMCPACK_URL = os.environ.get("QMCPACK_URL", "https://github.com/QMCPACK/qmcpack.git")

# Functional suite tracks develop.
QMCPACK_REF = os.environ.get("QMCPACK_REF", "develop")

# name, QMC_COMPLEX, QMC_MIXED_PRECISION. MPI stays off. GPU is OpenMP+HIP.
CONFIGS = (
    ("omp", "OFF", "OFF"),
    ("omp_cmplx", "ON", "OFF"),
    ("omp_mixed", "OFF", "ON"),
    ("omp_mixed_cmplx", "ON", "ON"),
)

# Per-test ctest timeout. Bounds a hung deterministic target.
CTEST_TIMEOUT = os.environ.get("QMCPACK_CTEST_TIMEOUT", "1800")

# Outer cap for one configuration's ``ctest -R deterministic``. Four
# configurations run in sequence. The public e2e job limit is 230 minutes,
# so each invocation stays under the remaining budget after the CMake builds.
CTEST_RUN_TIMEOUT = float(os.environ.get("QMCPACK_CTEST_RUN_TIMEOUT", "3600"))
