# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""test_tf32_hipblaslt_tunableops_perf.py -- TF32 hipBLASLt TunableOps performance validation.

Validates TF32 and TunableOp correctness and performance for nn.Linear at
dimensions (M=8196, K=512, N=3456) via hipBLASLt on AMD GPUs.

Markers auto-injected by CATEGORY_PROFILES for tests/e2e/hipblaslt/:
    hw.gpu, layer.math_lib, ci.nightly, e2e.stack, os.linux

Explicit markers per test function:
    runtime.medium

Supported GPU architectures: gfx942 (MI300X, MI325X), gfx950 (MI350X, MI355X, MI375X).
Supported OS: Ubuntu 22.04, Ubuntu 24.04, RHEL 9.7, SLES 15.7.
Environment: Bare Metal, Docker Container. Single GPU / Single Node.
"""

from __future__ import annotations

import base64
import pathlib
import re
import shlex
import textwrap

import pytest

from tests.common.ml_provisioning.workload import workload_failure_detail

_SRC = pathlib.Path(__file__).parent / "src" / "tf32_matmul_workload.py"
_SRC_DIR = str(_SRC.parent)

M, K, N = 8196, 512, 3456
REGRESSION_TOLERANCE_PCT = 20

# Supported GPU architecture prefixes — skip on all others.
_SUPPORTED_ARCH_PREFIXES = ("gfx942", "gfx950")


def _check_arch(gpu_arch: str | None) -> None:
    """Skip when gpu_arch is set and not in the supported set."""
    if gpu_arch and not gpu_arch.startswith(_SUPPORTED_ARCH_PREFIXES):
        pytest.skip(f"TF32 hipBLASLt tests not supported on {gpu_arch}; requires gfx942 or gfx950")


def _parse_avg_time_ms(stdout: str) -> float:
    """Extract the average matrix multiplication time (ms) from workload stdout."""
    match = re.search(
        r"Average time taken for matrix multiplication over \d+ runs:\s*([\d.]+)\s*ms",
        stdout,
    )
    if not match:
        raise ValueError(f"Could not parse timing from stdout:\n{stdout}")
    return float(match.group(1))


def _stage_workload(target_executor) -> str:
    """Stage src/ to the remote node when executor supports upload_tree.

    Returns the path (remote or local) to tf32_matmul_workload.py.
    """
    executor = next(iter(target_executor))
    if hasattr(executor, "upload_tree"):
        remote_src_dir: str = executor.upload_tree(_SRC_DIR)
        return remote_src_dir + "/" + _SRC.name
    return str(_SRC)


def _tf32_inline_script(allow_tf32: bool) -> str:
    """Return a self-contained Python script that runs a timed TF32/FP32 forward pass."""
    return textwrap.dedent(f"""\
        import torch, time, torch.nn as nn
        torch.backends.cuda.matmul.allow_tf32 = {allow_tf32}
        assert torch.backends.cuda.matmul.allow_tf32 == {allow_tf32}, (
            f"TF32 setting failed: expected {allow_tf32}, "
            f"got {{torch.backends.cuda.matmul.allow_tf32}}"
        )
        print(f"TF32 enabled: {{torch.backends.cuda.matmul.allow_tf32}}")
        M, K, N = {M}, {K}, {N}
        inp = torch.randn((M, K), dtype=torch.float32, device="cuda")
        layer = nn.Linear(K, N, dtype=torch.float32, device="cuda")
        for _ in range(10):
            layer(inp)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            layer(inp)
            torch.cuda.synchronize()
        t1 = time.time()
        avg = (t1 - t0) / 10.0
        print(f"Average time taken for matrix multiplication over 10 runs: {{avg * 1000.0}} ms")
        """)


def _tunableop_inline_script(workload_script: str, enabled: bool) -> str:
    """Return a Python script that enables/disables TunableOp then execs the workload."""
    return textwrap.dedent(f"""\
        import torch
        import torch.cuda.tunable as tunable
        tunable.enable({enabled})
        actual = tunable.is_enabled()
        assert actual == {enabled}, (
            f"TunableOp setting failed: expected {enabled}, got {{actual}}"
        )
        print(f"TunableOp enabled: {{actual}}")
        exec(open({workload_script!r}).read())
        """)


def _run_script_in_tmpdir(
    target_executor,
    script: str,
    env_prefix: str,
    python: str,
    timeout: int = 300,
) -> object:
    """Write script via base64 into a tmpdir and run it — avoids all shell quoting conflicts.

    Encodes the script as base64, decodes it on the remote side into a temp file,
    then runs the file. The trace/output files land in the same tmpdir.
    """
    b64 = base64.b64encode(script.encode()).decode()
    cmd = (
        f"sh -c '"
        f"D=$(mktemp -d) && "
        f"echo {b64} | base64 -d > $D/_script.py && "
        f"cd $D && "
        f"{env_prefix} {python} $D/_script.py && "
        f"ls trace_forward_matmul_*.json 2>/dev/null | wc -l"
        f"'"
    )
    return target_executor.run(cmd, timeout=timeout)


# ---------------------------------------------------------------------------
# TestTF32LinearForward
# ---------------------------------------------------------------------------


class TestTF32LinearForward:
    """Verify that the full workload runs successfully and emits a profiler trace."""

    @pytest.mark.runtime.medium
    def test_tf32_linear_forward_generates_profiler_trace(
        self,
        gpu_arch: str | None,
        require_torch,
        torch_python: str,
        target_executor,
        ld_path: dict,
    ):
        """Run tf32_matmul_workload.py and verify exactly one Chrome trace file is written."""
        _check_arch(gpu_arch)
        script_path = _stage_workload(target_executor)
        workload_script = pathlib.Path(script_path).read_text()

        ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
        python = shlex.quote(torch_python)

        result = _run_script_in_tmpdir(
            target_executor,
            script=workload_script,
            env_prefix=f"env LD_LIBRARY_PATH={ld}",
            python=python,
        )
        detail = workload_failure_detail(result, "tf32_linear_forward_generates_profiler_trace")
        assert result.ok, (
            f"tf32_matmul_workload failed (exit={result.exit_code}):{detail}\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
        trace_count = result.stdout.strip().splitlines()[-1].strip()
        assert trace_count == "1", (
            f"Expected exactly 1 trace file, got count={trace_count!r}\n" f"stdout: {result.stdout[:1000]}"
        )


# ---------------------------------------------------------------------------
# TestTF32vsF32LinearPerformance
# ---------------------------------------------------------------------------


class TestTF32vsF32LinearPerformance:
    """Parametric TF32 ON/OFF correctness and regression checks."""

    @pytest.mark.runtime.medium
    @pytest.mark.parametrize(
        "allow_tf32",
        [True, False],
        ids=["tf32_ON", "tf32_OFF"],
    )
    def test_linear_matmul_runs_with_tf32_on_off(
        self,
        allow_tf32: bool,
        gpu_arch: str | None,
        require_torch,
        torch_python: str,
        target_executor,
        ld_path: dict,
    ):
        """Verify nn.Linear completes without error with TF32 enabled or disabled."""
        _check_arch(gpu_arch)
        script = _tf32_inline_script(allow_tf32)
        hipblaslt_tf32 = "1" if allow_tf32 else "0"
        ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
        python = shlex.quote(torch_python)

        result = target_executor.run(
            f"env HIPBLASLT_ALLOW_TF32={hipblaslt_tf32} LD_LIBRARY_PATH={ld}" f" {python} -c {shlex.quote(script)}",
            timeout=300,
        )
        detail = workload_failure_detail(result, f"tf32_linear_matmul tf32={allow_tf32}")
        assert result.ok, (
            f"test_linear_matmul_runs_with_tf32_on_off[tf32={allow_tf32}] "
            f"failed (exit={result.exit_code}):{detail}\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
        assert (
            f"TF32 enabled: {allow_tf32}" in result.stdout
        ), f"Expected 'TF32 enabled: {allow_tf32}' in stdout:{detail}\n{result.stdout[:1000]}"
        assert (
            "Average time taken for matrix multiplication" in result.stdout
        ), f"No timing output in stdout:{detail}\n{result.stdout[:1000]}"
        # Additional verification: parse the numeric timing to confirm a real value was
        # produced — the string check above only confirms the print ran.
        time_ms = _parse_avg_time_ms(result.stdout)
        assert time_ms is not None, f"No valid numeric timing in stdout:{detail}\n{result.stdout[:1000]}"
        assert time_ms > 0, f"Timing was zero or negative:{detail}\n{result.stdout[:1000]}"

    @pytest.mark.runtime.medium
    def test_tf32_not_slower_than_fp32(
        self,
        gpu_arch: str | None,
        require_torch,
        torch_python: str,
        target_executor,
        ld_path: dict,
    ):
        """Assert TF32 ON is not more than 20% slower than FP32 (regression guard)."""
        _check_arch(gpu_arch)
        ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
        python = shlex.quote(torch_python)

        # Run FP32 baseline
        script_off = _tf32_inline_script(allow_tf32=False)
        result_off = target_executor.run(
            f"env HIPBLASLT_ALLOW_TF32=0 LD_LIBRARY_PATH={ld}" f" {python} -c {shlex.quote(script_off)}",
            timeout=300,
        )
        detail_off = workload_failure_detail(result_off, "tf32_not_slower[fp32]")
        assert result_off.ok, (
            f"FP32 run failed (exit={result_off.exit_code}):{detail_off}\n"
            f"stdout: {result_off.stdout[:2000]}\nstderr: {result_off.stderr[:500]}"
        )
        time_off = _parse_avg_time_ms(result_off.stdout)

        # Run TF32
        script_on = _tf32_inline_script(allow_tf32=True)
        result_on = target_executor.run(
            f"env HIPBLASLT_ALLOW_TF32=1 LD_LIBRARY_PATH={ld}" f" {python} -c {shlex.quote(script_on)}",
            timeout=300,
        )
        detail_on = workload_failure_detail(result_on, "tf32_not_slower[tf32]")
        assert result_on.ok, (
            f"TF32 run failed (exit={result_on.exit_code}):{detail_on}\n"
            f"stdout: {result_on.stdout[:2000]}\nstderr: {result_on.stderr[:500]}"
        )
        time_on = _parse_avg_time_ms(result_on.stdout)

        threshold = time_off * (1.0 + REGRESSION_TOLERANCE_PCT / 100.0)
        assert time_on <= threshold, (
            f"TF32 regression detected: TF32 ON ({time_on:.2f} ms) is slower than "
            f"TF32 OFF ({time_off:.2f} ms) beyond {REGRESSION_TOLERANCE_PCT}% tolerance "
            f"(threshold={threshold:.2f} ms)."
        )


# ---------------------------------------------------------------------------
# TestTunableOpLinearMatmul
# ---------------------------------------------------------------------------


class TestTunableOpLinearMatmul:
    """TunableOp ON/OFF correctness and trace generation checks.

    ``require_torch_tunableop`` skips the entire class when torch.cuda.tunable
    is absent from the installed PyTorch build.
    """

    @pytest.mark.runtime.medium
    @pytest.mark.parametrize(
        "tunableop_enabled",
        [True, False],
        ids=["tunableop_ON", "tunableop_OFF"],
    )
    def test_linear_matmul_runs_with_tunableop_on_off(
        self,
        tunableop_enabled: bool,
        gpu_arch: str | None,
        require_torch_tunableop,
        torch_python: str,
        target_executor,
        ld_path: dict,
    ):
        """Verify nn.Linear completes without error with TunableOp enabled or disabled."""
        _check_arch(gpu_arch)
        script_path = _stage_workload(target_executor)
        script = _tunableop_inline_script(script_path, tunableop_enabled)

        tunableop_val = "1" if tunableop_enabled else "0"
        tuning_val = "1" if tunableop_enabled else "0"
        ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
        python = shlex.quote(torch_python)

        result = target_executor.run(
            f"env PYTORCH_TUNABLEOP_ENABLED={tunableop_val}"
            f" PYTORCH_TUNABLEOP_TUNING={tuning_val}"
            f" LD_LIBRARY_PATH={ld}"
            f" {python} -c {shlex.quote(script)}",
            timeout=300,
        )
        detail = workload_failure_detail(result, f"tunableop_linear_matmul enabled={tunableop_enabled}")
        assert result.ok, (
            f"test_linear_matmul_runs_with_tunableop_on_off[enabled={tunableop_enabled}] "
            f"failed (exit={result.exit_code}):{detail}\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
        assert (
            f"TunableOp enabled: {tunableop_enabled}" in result.stdout
        ), f"Expected 'TunableOp enabled: {tunableop_enabled}' in stdout:{detail}\n{result.stdout[:1000]}"
        assert (
            "Average time taken for matrix multiplication" in result.stdout
        ), f"No timing output in stdout:{detail}\n{result.stdout[:1000]}"
        # Additional verification: parse the numeric timing to confirm a real value was
        # produced — the string check above only confirms the print ran.
        time_ms = _parse_avg_time_ms(result.stdout)
        assert time_ms is not None, f"No valid numeric timing in stdout:{detail}\n{result.stdout[:1000]}"
        assert time_ms > 0, f"Timing was zero or negative:{detail}\n{result.stdout[:1000]}"

    @pytest.mark.runtime.medium
    @pytest.mark.parametrize(
        "tunableop_enabled",
        [True, False],
        ids=["tunableop_ON", "tunableop_OFF"],
    )
    def test_linear_matmul_generates_trace_with_tunableop(
        self,
        tunableop_enabled: bool,
        gpu_arch: str | None,
        require_torch_tunableop,
        torch_python: str,
        target_executor,
        ld_path: dict,
    ):
        """Verify profiler trace is generated when TunableOp is enabled or disabled."""
        _check_arch(gpu_arch)
        script_path = _stage_workload(target_executor)
        script = _tunableop_inline_script(script_path, tunableop_enabled)

        tunableop_val = "1" if tunableop_enabled else "0"
        ld = shlex.quote(ld_path["LD_LIBRARY_PATH"])
        python = shlex.quote(torch_python)

        result = _run_script_in_tmpdir(
            target_executor,
            script=script,
            env_prefix=(
                f"env PYTORCH_TUNABLEOP_ENABLED={tunableop_val}"
                f" PYTORCH_TUNABLEOP_TUNING={tunableop_val}"
                f" LD_LIBRARY_PATH={ld}"
            ),
            python=python,
        )
        detail = workload_failure_detail(result, f"tunableop_trace enabled={tunableop_enabled}")
        assert result.ok, (
            f"test_linear_matmul_generates_trace_with_tunableop[enabled={tunableop_enabled}] "
            f"failed (exit={result.exit_code}):{detail}\n"
            f"stdout: {result.stdout[:2000]}\nstderr: {result.stderr[:500]}"
        )
        trace_count = result.stdout.strip().splitlines()[-1].strip()
        assert trace_count == "1", (
            f"Expected 1 trace file, got count={trace_count!r} "
            f"(tunableop_enabled={tunableop_enabled})\n"
            f"stdout: {result.stdout[:1000]}"
        )
