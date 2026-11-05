# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pins for the functional GROMACS HIP MPI suite.

GROMACS is cloned at runtime. UCX and Open MPI are the shared framework builds,
not clones started by this suite.

Environment overrides:
    GROMACS_URL            git URL. Default the upstream GitLab project.
    GROMACS_REF            git ref. Default the HIP feature-enablement branch.
    GROMACS_RUN_TIMEOUT    outer wall-clock cap for ctest.
"""

from dataclasses import dataclass
import os

GROMACS_URL = os.environ.get("GROMACS_URL", "https://gitlab.com/gromacs/gromacs.git")
GROMACS_REF = os.environ.get("GROMACS_REF", "4947-hip-feature-enablement")


@dataclass(frozen=True)
class GromacsBuild:
    """Paths the ctest step needs. The session fixture has already built."""

    build_dir: str
    mpi_home: str
    ucx_prefix: str


# Per-test ctest timeout, passed through to ``ctest --timeout``.
CTEST_TIMEOUT = os.environ.get("GROMACS_CTEST_TIMEOUT", "1800")

# Public e2e jobs stop at 230 minutes. Stay under that after checkout.
RUN_TIMEOUT = float(os.environ.get("GROMACS_RUN_TIMEOUT", "12000"))
