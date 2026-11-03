# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared MPI and HPC benchmark helpers for ROCm test suites.

Mirrors the ``tests/common/criu/`` pattern. Import from here in any conftest
that needs to detect or provision an MPI runtime, construct a benchmark
environment prefix, or assert on an MPI launch result.

Public surface:
    - :func:`ensure_ssh_client` — install openssh-client when ``ssh`` is absent.
    - :func:`ensure_mpi_runtime` — detect or provision an OpenMPI runtime.
    - :func:`build_hpc_env_prefix` — construct the ``env VAR=...`` prefix for HPC runs.
    - :func:`assert_mpi_launch_ok` — assert an ``mpirun`` process exited cleanly.
"""

from __future__ import annotations

from .mpi import assert_mpi_launch_ok, build_hpc_env_prefix, ensure_mpi_runtime, ensure_ssh_client

__all__ = [
    "assert_mpi_launch_ok",
    "build_hpc_env_prefix",
    "ensure_mpi_runtime",
    "ensure_ssh_client",
]
