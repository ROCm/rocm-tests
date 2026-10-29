# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared parameters for the functional RAJA HIP unit suite.

Imported by ``conftest.py`` (clone + build) and ``test_raja_functional.py``
(ctest) so the pin and the timeouts cannot drift.

Environment overrides:
    RAJA_URL                 git URL to clone. Default LLNL/RAJA.
    RAJA_REF                 git ref (branch/tag/SHA). Default ``develop``.
    RAJA_CTEST_TIMEOUT       per-test ``ctest --timeout`` in seconds.
    RAJA_CTEST_RUN_TIMEOUT   outer wall-clock cap for the whole ctest run.
"""

import os

# Upstream RAJA performance-portability library. Cloned at runtime, not vendored.
RAJA_URL = os.environ.get("RAJA_URL", "https://github.com/LLNL/RAJA.git")

# Functional suite tracks develop, matching the ROCmTest raja_ut pin.
RAJA_REF = os.environ.get("RAJA_REF", "develop")

# Per-test ctest timeout. Bounds a hung target without cutting a slow unit test.
CTEST_TIMEOUT = os.environ.get("RAJA_CTEST_TIMEOUT", "1800")

# Outer cap for the full ctest invocation. Above the per-test timeout so the
# executor does not pre-empt ctest's own accounting. The public e2e job limit
# is 230 minutes, so this stays under that budget after the CMake build.
CTEST_RUN_TIMEOUT = float(os.environ.get("RAJA_CTEST_RUN_TIMEOUT", "7200"))
