# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RCCL collectives under forced kernel serialization.

Public ``rccl-tests`` perf clients are run with ``AMD_SERIALIZE_KERNEL=1`` to
cover the hang class.  The two legacy legs are preserved: an AllReduce-only
smoke and an all-collectives sweep.
"""

import os
import shlex

import pytest

_DATA_BYTES = int(os.environ.get("DATA_BYTES", "1048576"))
_STEP_FACTOR = int(os.environ.get("STEP_FACTOR", "2"))
_STEP_TIMEOUT = float(os.environ.get("TIMEOUT_SEC", "300"))
_COLLECTIVE_BINS = (
    ("allreduce", "all_reduce_perf"),
    ("broadcast", "broadcast_perf"),
    ("reduce", "reduce_perf"),
    ("allgather", "all_gather_perf"),
    ("reducescatter", "reduce_scatter_perf"),
)


def _effective_gpu_counts(max_gpus: int) -> list[int]:
    """Return legacy all-collectives ``-g`` values capped to the acquired GPUs."""
    if max_gpus < 2:
        pytest.skip("distributed collective serialization requires at least 2 acquired GPUs")
    values = {candidate for candidate in (2, 4, 8, max_gpus) if 2 <= candidate <= max_gpus}
    return sorted(values) or [max_gpus]


def _perf_binary(rccl_tests_build: str, binary_name: str) -> str:
    """Return the expected rccl-tests binary path under the pytest-built tree."""
    return os.path.join(rccl_tests_build, binary_name)


def _binary_exists(target_executor, binary: str) -> bool:
    """Check binary availability on the execution node, not the pytest coordinator."""
    result = target_executor.run(f"test -x {shlex.quote(binary)}", timeout=15)
    return result.ok


def _run_serialized_step(
    *,
    target_executor,
    ld_path: dict,
    rock_dir: str,
    label: str,
    binary: str,
    args: str,
) -> str | None:
    """Run one serialized rccl-tests client; return a failure summary or ``None``."""
    ld = ld_path["LD_LIBRARY_PATH"]
    cmd = (
        "env "
        f"AMD_SERIALIZE_KERNEL=1 "
        f"ROCM_PATH={shlex.quote(rock_dir)} "
        f"RCCL_BIN_DIR={shlex.quote(os.path.dirname(binary))} "
        f"LD_LIBRARY_PATH={shlex.quote(ld)} "
        f"{shlex.quote(binary)} {args}"
    )
    result = target_executor.run(cmd, timeout=_STEP_TIMEOUT)
    if result.ok:
        return None
    return (
        f"{label}: exit={result.exit_code}\n"
        f"cmd: {cmd}\n"
        f"stdout:\n{result.stdout[:2000]}\n"
        f"stderr:\n{result.stderr[:1000]}"
    )


def _assert_no_failures(title: str, failures: list[str]) -> None:
    """Preserve the shell driver's aggregate pass/fail behavior."""
    if failures:
        pytest.fail(
            f"OVERALL: FAILED ({len(failures)} failing step(s))\n\n" f"{title} failed:\n\n" + "\n\n".join(failures)
        )
    print("OVERALL: PASSED")


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_distributed_collective_serialization_allreduce_only(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    require_rccl,
    requested_gpu_count: int,
    rccl_tests_build: str,
):
    """--allreduce-only: full-width AllReduce under AMD_SERIALIZE_KERNEL=1 must report OVERALL: PASSED."""
    if requested_gpu_count < 2:
        pytest.skip("distributed collective serialization requires at least 2 acquired GPUs")
    binary = _perf_binary(rccl_tests_build, "all_reduce_perf")
    failures: list[str] = []
    if not _binary_exists(target_executor, binary):
        failures.append(f"all_reduce_perf not found or not executable: {binary}")
    else:
        failure = _run_serialized_step(
            target_executor=target_executor,
            ld_path=ld_path,
            rock_dir=rock_dir,
            label="serialized_allreduce_all_gpus",
            binary=binary,
            args=(f"-b {_DATA_BYTES} -e {_DATA_BYTES} " f"-f {_STEP_FACTOR} -g {requested_gpu_count}"),
        )
        if failure:
            failures.append(failure)
    _assert_no_failures("serialization AllReduce smoke", failures)


def _run_all_collectives(
    *,
    target_executor,
    ld_path: dict,
    rock_dir: str,
    requested_gpu_count: int,
    rccl_tests_build: str,
) -> list[str]:
    """Run the legacy all-collectives serialized sweep and return failure summaries."""
    failures: list[str] = []
    for ng in _effective_gpu_counts(requested_gpu_count):
        for short_name, binary_name in _COLLECTIVE_BINS:
            binary = _perf_binary(rccl_tests_build, binary_name)
            label = f"serialized_{short_name}_g{ng}"
            if not _binary_exists(target_executor, binary):
                failures.append(f"{label}: {binary_name} not found or not executable: {binary}")
                continue
            failure = _run_serialized_step(
                target_executor=target_executor,
                ld_path=ld_path,
                rock_dir=rock_dir,
                label=label,
                binary=binary,
                args=f"-b {_DATA_BYTES} -e {_DATA_BYTES} -g {ng}",
            )
            if failure:
                failures.append(failure)

        if ng == 2:
            sendrecv = next(
                (
                    _perf_binary(rccl_tests_build, candidate)
                    for candidate in ("sendrecv_perf", "sendrecv")
                    if _binary_exists(
                        target_executor,
                        _perf_binary(rccl_tests_build, candidate),
                    )
                ),
                None,
            )
            if sendrecv:
                failure = _run_serialized_step(
                    target_executor=target_executor,
                    ld_path=ld_path,
                    rock_dir=rock_dir,
                    label="serialized_sendrecv_g2",
                    binary=sendrecv,
                    args=f"-b {_DATA_BYTES} -e {_DATA_BYTES} -g 2",
                )
                if failure:
                    failures.append(failure)
    return failures


@pytest.mark.gpu_count("ALL")
@pytest.mark.runtime.medium
def test_distributed_collective_serialization_all_collectives(
    target_executor,
    ld_path: dict,
    rock_dir: str,
    require_rccl,
    requested_gpu_count: int,
    rccl_tests_build: str,
):
    """--all-collectives-only: collective sweep under AMD_SERIALIZE_KERNEL=1 must report OVERALL: PASSED."""
    failures = _run_all_collectives(
        target_executor=target_executor,
        ld_path=ld_path,
        rock_dir=rock_dir,
        requested_gpu_count=requested_gpu_count,
        rccl_tests_build=rccl_tests_build,
    )
    _assert_no_failures("serialization all-collectives sweep", failures)
