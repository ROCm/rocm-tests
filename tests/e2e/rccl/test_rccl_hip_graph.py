# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_rccl_hip_graph.py -- RCCL HIP-graph capture-path correctness.

Build the public ``rccl-tests`` clients and run every ``*_perf`` binary in
HIP-graph mode (``-G 1``); each run must complete correctly.  The invocation
uses the rccl-tests default GPU count (single GPU per process, no ``-g``) and
iterates over all built perf binaries.

Single-GPU per process (matches the ``-G 1`` invocation without ``-g``);
hw.gpu is declared per function and overrides the hw.multi_gpu profile default.
"""

import os

import pytest

from framework.rocm.libs.rccl import correctness_ok, run_perf


@pytest.mark.hw.gpu
@pytest.mark.runtime.medium
def test_rccl_hip_graph(
    target_executor,
    ld_path: dict,
    require_rccl,
    rccl_perf_binaries: list,
):
    """Every rccl-tests perf binary must pass validation in HIP-graph mode (-G 1)."""
    ld = ld_path["LD_LIBRARY_PATH"]
    perf_binaries = rccl_perf_binaries
    assert perf_binaries, "no rccl-tests *_perf binaries found (rccl_perf_binaries fixture is empty)"

    failures: list[str] = []
    for binary in perf_binaries:
        result = run_perf(
            target_executor,
            binary,
            n_gpus=1,
            extra_args="-n 8 -b 16 -e 1G -f 2 -G 1 -c 1",
            env={"LD_LIBRARY_PATH": ld},
            operation=f"hip_graph:{os.path.basename(binary)}",
        )
        if not correctness_ok(result):
            failures.append(f"{os.path.basename(binary)}:\n{result.raw_output[:1500]}")

    assert not failures, "RCCL HIP-graph validation failed for:\n" + "\n---\n".join(failures)
