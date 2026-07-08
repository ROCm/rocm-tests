# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rccl_with_mgpu.py -- RCCL multi-GPU collective correctness + bandwidth.

Builds the public ``rccl-tests`` perf clients with ``MPI=1`` and runs all RCCL
TESTS operations through ``mpirun --allow-run-as-root``.

Objective: sweep all RCCL TESTS operations
(``all_gather``, ``all_reduce``, ``broadcast``, ``reduce``, ``reduce_scatter``)
across the full dtype/redop matrix and require parsed bus bandwidth plus RCCL
correctness output.

Multi-GPU (hw.multi_gpu) via the tests/e2e/rccl profile.
"""

import os

import pytest

from framework.rocm.libs.rccl import correctness_ok, run_perf

_COLLECTIVES = (
    ("all_gather", "all_gather_perf"),
    ("all_reduce", "all_reduce_perf"),
    ("broadcast", "broadcast_perf"),
    ("reduce", "reduce_perf"),
    ("reduce_scatter", "reduce_scatter_perf"),
)

_DTYPES = ("int8", "uint8", "int32", "uint32", "int64", "uint64", "half", "float", "double")
_REDOPS = ("sum", "prod", "min", "max")


def _legacy_rank_geometry(total_gpus: int) -> tuple[int, int]:
    """Return the ``mpirun -np`` and per-rank ``-g`` values."""
    ranks = 1 if total_gpus == 1 else 2
    return ranks, max(1, total_gpus // ranks)


def _run_mpi_cell(
    *,
    target_executor,
    ld_path: dict,
    mpi_runtime,
    requested_gpu_count: int,
    rccl_tests_mpi_build: str,
    collective: str,
    binary_name: str,
    dtype: str,
    redop: str,
) -> None:
    """Run one legacy RCCL_TESTS MPI cell and assert correctness + bus bandwidth."""
    if requested_gpu_count < 2:
        pytest.skip("RCCL_TESTS multi-GPU sweep requires at least 2 acquired GPUs")
    ld = ld_path["LD_LIBRARY_PATH"]
    binary = os.path.join(rccl_tests_mpi_build, binary_name)
    ranks, gpus_per_rank = _legacy_rank_geometry(requested_gpu_count)
    mpi_env = dict(mpi_runtime.env)
    mpi_env["LD_LIBRARY_PATH"] = ":".join(value for value in (mpi_env.get("LD_LIBRARY_PATH", ""), ld) if value)
    result = run_perf(
        target_executor,
        binary,
        n_gpus=gpus_per_rank,
        extra_args=f"-b 64 -e 1G -f 4 -o {redop} -d {dtype}",
        env=mpi_env,
        launcher=f"{mpi_runtime.launcher} --allow-run-as-root -np {ranks}",
        operation=f"mgpu_{collective}_{dtype}_{redop}",
        timeout=600,
    )
    assert correctness_ok(result), (
        f"RCCL multi-GPU {collective} failed validation for dtype={dtype}, redop={redop}:\n"
        f"{result.raw_output[:3000]}"
    )
    assert result.bandwidth_gbps > 0, f"expected a positive bus bandwidth from rccl-tests:\n{result.raw_output[:3000]}"


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_rccl_with_mgpu_mpi_smoke(
    target_executor,
    ld_path: dict,
    require_rccl,
    mpi_runtime,
    requested_gpu_count: int,
    rccl_tests_mpi_build: str,
):
    """Nightly MPI=1 smoke: one canonical all_reduce/float/sum cell proves e2e wiring."""
    _run_mpi_cell(
        target_executor=target_executor,
        ld_path=ld_path,
        mpi_runtime=mpi_runtime,
        requested_gpu_count=requested_gpu_count,
        rccl_tests_mpi_build=rccl_tests_mpi_build,
        collective="all_reduce",
        binary_name="all_reduce_perf",
        dtype="float",
        redop="sum",
    )


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count("ALL")
@pytest.mark.ci.weekly
@pytest.mark.runtime.soak
@pytest.mark.parametrize(
    ("collective", "binary_name"),
    _COLLECTIVES,
    ids=[name for name, _ in _COLLECTIVES],
)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("redop", _REDOPS)
def test_rccl_with_mgpu_full_matrix(
    collective: str,
    binary_name: str,
    dtype: str,
    redop: str,
    target_executor,
    ld_path: dict,
    require_rccl,
    mpi_runtime,
    requested_gpu_count: int,
    rccl_tests_mpi_build: str,
):
    """Weekly RCCL_TESTS MPI full matrix cell must validate and report bus bandwidth."""
    _run_mpi_cell(
        target_executor=target_executor,
        ld_path=ld_path,
        mpi_runtime=mpi_runtime,
        requested_gpu_count=requested_gpu_count,
        rccl_tests_mpi_build=rccl_tests_mpi_build,
        collective=collective,
        binary_name=binary_name,
        dtype=dtype,
        redop=redop,
    )
