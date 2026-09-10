# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""MGBench L1 multi-GPU data transfer benchmarks.

Runs the three vendored MGBench L1 benchmarks across every GPU on the node:

- ``fullduplex`` — bi-directional inter-GPU memory exchange.
- ``halfduplex`` — uni-directional host/GPU and inter-GPU transfers.
- ``uva``        — DMA exchange over unified addressing, in four flag combinations.

These benchmarks report bandwidth rather than a pass token, so the legacy pass
criteria are preserved: each run must reach its measurement phase and must not
emit a failure or abort line.
"""

import re

import pytest

_TIMEOUT_SECS = 1800.0

# Echoed once flags are parsed and the measurement loop is about to start, so
# its presence is what the legacy parser scored fullduplex and uva on.
_REPETITIONS_LINE = "Repetitions: 100"

# halfduplex is instead scored on having produced at least one host-to-GPU and
# one outbound-from-GPU bandwidth measurement.
_HOST_TO_GPU_RE = re.compile(r"Copying from host to GPU\s+\d+:\s+\d+\.\d+\s+\w+/s\s+\(\d+\.\d+\s+\w+\)")
_FROM_GPU_RE = re.compile(r"Copying from GPU \d+ \w+ \w+:\s+\d+\.\d+\s+\w+/s\s+\(\d+\.\d+\s+\w+\)")

# Exit codes the legacy runner classified as an abnormal termination rather
# than a test verdict; a plain non-zero exit was left to the log parser.
_CRASH_EXIT_CODES = frozenset({-11, -6, -15, -9, -2, 127, 130, 134, 137, 139, 143})

# Flag combinations the legacy suite ran against the single uva binary.
_UVA_VARIANTS = (
    pytest.param("", id="uva"),
    pytest.param("--write", id="uva_write"),
    pytest.param("--write --fullduplex", id="uva_write_fullduplex"),
    pytest.param("--fullduplex", id="uva_fullduplex"),
)


def _run_benchmark(target_executor, ld_path: dict, binary: str, args: str, label: str):
    """Run one benchmark and apply the checks common to every MGBench case."""
    cmd = f"env LD_LIBRARY_PATH={ld_path['LD_LIBRARY_PATH']} {binary}"
    if args:
        cmd = f"{cmd} {args}"
    result = target_executor.run(cmd, timeout=_TIMEOUT_SECS)

    assert result.exit_code not in _CRASH_EXIT_CODES, (
        f"{label} terminated abnormally (exit={result.exit_code}):\n"
        f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-500:]}"
    )
    offenders = [line for line in result.stdout.splitlines() if "FAIL" in line or "ABORT" in line]
    assert not offenders, f"{label} reported a failure line:\n" + "\n".join(offenders[:20])
    return result


@pytest.mark.hw.multi_gpu
@pytest.mark.runtime.medium
@pytest.mark.gpu_count("all")
def test_hip_mgbench_fullduplex(target_executor, ld_path: dict, mgbench_binary):
    """Exchange memory bi-directionally between every pair of GPUs."""
    result = _run_benchmark(target_executor, ld_path, mgbench_binary("fullduplex"), "", "fullduplex")
    assert (
        _REPETITIONS_LINE in result.stdout
    ), f"fullduplex did not reach its measurement phase — no '{_REPETITIONS_LINE}':\n{result.stdout[-2000:]}"


@pytest.mark.hw.multi_gpu
@pytest.mark.runtime.medium
@pytest.mark.gpu_count("all")
def test_hip_mgbench_halfduplex(target_executor, ld_path: dict, mgbench_binary):
    """Transfer memory uni-directionally between host and every GPU pair."""
    result = _run_benchmark(target_executor, ld_path, mgbench_binary("halfduplex"), "", "halfduplex")
    assert _HOST_TO_GPU_RE.search(
        result.stdout
    ), f"halfduplex reported no host-to-GPU bandwidth measurement:\n{result.stdout[-2000:]}"
    assert _FROM_GPU_RE.search(
        result.stdout
    ), f"halfduplex reported no outbound-from-GPU bandwidth measurement:\n{result.stdout[-2000:]}"


@pytest.mark.hw.multi_gpu
@pytest.mark.runtime.medium
@pytest.mark.gpu_count("all")
@pytest.mark.parametrize("args", _UVA_VARIANTS)
def test_hip_mgbench_uva(target_executor, ld_path: dict, mgbench_binary, args: str):
    """Exchange memory over unified addressing, sweeping DMA direction and duplex mode."""
    label = f"uva {args}".strip()
    result = _run_benchmark(target_executor, ld_path, mgbench_binary("uva"), args, label)
    assert (
        _REPETITIONS_LINE in result.stdout
    ), f"{label} did not reach its measurement phase — no '{_REPETITIONS_LINE}':\n{result.stdout[-2000:]}"
