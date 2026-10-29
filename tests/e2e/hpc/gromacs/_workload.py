# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pins for the functional GROMACS HIP MPI suite.

The build clones these at runtime. They are not vendored in this repository.

Environment overrides:
    GROMACS_URL            git URL. Default the upstream GitLab project.
    GROMACS_REF            git ref. Default the HIP feature-enablement branch.
    GROMACS_UCX_URL        OpenUCX git URL.
    GROMACS_UCX_REF        OpenUCX git ref.
    GROMACS_OPENMPI_URL    Open MPI 4.1.4 tarball.
    GROMACS_RUN_TIMEOUT    outer wall-clock cap for the whole build and ctest.
"""

import os

GROMACS_URL = os.environ.get("GROMACS_URL", "https://gitlab.com/gromacs/gromacs.git")
GROMACS_REF = os.environ.get("GROMACS_REF", "4947-hip-feature-enablement")

UCX_URL = os.environ.get("GROMACS_UCX_URL", "https://github.com/openucx/ucx.git")
UCX_REF = os.environ.get("GROMACS_UCX_REF", "v1.21.x")

OPENMPI_URL = os.environ.get(
    "GROMACS_OPENMPI_URL",
    "https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.4.tar.gz",
)

# Per-test ctest timeout, passed through to ``ctest --timeout``.
CTEST_TIMEOUT = os.environ.get("GROMACS_CTEST_TIMEOUT", "1800")

# Public e2e jobs stop at 230 minutes. Stay under that after checkout.
RUN_TIMEOUT = float(os.environ.get("GROMACS_RUN_TIMEOUT", "12000"))
