# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""hipblaslt_bench: run hipblaslt-bench GEMM sweep under power/temp monitoring.

Shapes come from ROCmTestInternal's hipBLASlt_GEMM.sh.

``hipblaslt-bench`` is a ROCm test client (shipped in the
``hipblaslt-benchmarks`` / ``amdrocm-blas-test*`` package, and in the ROCm
test tarball) and is expected to be **preinstalled** at
``<rocm_root>/bin/hipblaslt-bench`` before the suite runs -- exactly how
ROCmTest consumes it (``runHipBLASlt_GEMM.py`` installs the package as a
prerequisite, then runs ``<rocm_dir>/bin/hipblaslt-bench``). This test does
NOT build hipBLASLt from source or install a package; if the binary is
absent it reports ``BUILD_FAILED`` so CI surfaces the missing prerequisite.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.executor_bridge import run_command_captured
from tests.common.gpu_monitored.workloads.base import BuildContext, BuildStatus, RunContext, RunResult, Test, TestSpec


# GEMM shapes: (M, N, K, batch_count)
NN_SHAPES: List[Tuple[int, int, int, int]] = [
    (8192, 320, 320, 1),
    (2048, 640, 640, 1),
    (512, 1280, 1280, 1),
    (8192, 320, 1280, 1),
    (512, 10240, 1280, 1),
    (2048, 5120, 640, 1),
    (8192, 2560, 320, 1),
    (512, 1280, 5120, 1),
    (2048, 640, 2560, 1),
    (154, 320, 768, 1),
    (154, 1280, 768, 1),
    (4096, 40, 4096, 16),
    (1024, 80, 1024, 16),
    (1024, 80, 77, 16),
]

NT_SHAPES: List[Tuple[int, int, int, int]] = [
    (4096, 4096, 40, 16),
    (1024, 1024, 80, 16),
    (4096, 77, 40, 16),
    (256, 77, 160, 16),
    (1024, 77, 80, 16),
]


class HipblasltBench(Test):
    spec = TestSpec(
        name="hipblaslt_bench",
        goal="hipBLASLt GEMM perf sweep with monitoring",
        # A GEMM perf sweep is not a utilisation workload. Each of the 19
        # shapes runs as its own short-lived hipblaslt-bench process whose
        # wall time is dominated by host-side setup and the CPU reference
        # computation, while the GEMM kernels themselves take microseconds
        # (shape 1 on MI325X: 9.79 us on GPU vs 58 ms on CPU). Sampled at
        # 1 Hz, a perfectly healthy run therefore peaks around 6% GFX util
        # and averages under 1%, so a compute-style min_util only
        # manufactures warnings. Health is judged by the GEMM results
        # themselves (19 shapes validated), not by GFX utilisation.
        workload_profile={"min_util": 0, "min_vram_pct": 0, "serial": True},
    )

    ITERS = 600
    COLD_ITERS = 10

    def build(self, ctx: BuildContext) -> BuildStatus:
        if self._is_installed(ctx.rocm_root):
            print(f"  [build] hipblaslt-bench: found at {ctx.rocm_root}/bin")
            return BuildStatus.OK

        print(f"  [build] hipblaslt-bench: not found at "
              f"{ctx.rocm_root}/bin/hipblaslt-bench. It ships in the ROCm "
              f"'hipblaslt-benchmarks' / amdrocm-blas-test* package (and the "
              f"ROCm test tarball) and must be preinstalled in this ROCm root "
              f"as a prerequisite; the harness does not build it from source. "
              f"Install it and re-run.")
        return BuildStatus.BUILD_FAILED

    def available(self, config: Config) -> bool:
        return self._find_bin(config) is not None

    def run(self, ctx: RunContext) -> RunResult:
        bench = self._find_bin(ctx.config)
        if bench is None:
            print("hipblaslt-bench not found")
            return RunResult(exit_code=1)

        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        rocm_lib = f"{ctx.rocm_root}/lib"
        env = {
            "LD_LIBRARY_PATH": f"{rocm_lib}:{existing_ld}" if existing_ld else rocm_lib,
        }
        common_args = [
            "--precision", "f16_r",
            "--compute_type", "f32_r",
            "--activation_type", "none",
            "--iters", str(self.ITERS),
            "--cold_iters", str(self.COLD_ITERS),
            "--alpha", "1",
            "--beta", "0",
        ]

        total_shapes = len(NN_SHAPES) + len(NT_SHAPES)
        print(f"  Running {total_shapes} GEMM shapes ({len(NN_SHAPES)} NN + {len(NT_SHAPES)} NT), "
              f"{self.ITERS} iters each")
        shape_timeout = ctx.config.per_iter_watchdog or None

        # Reproduce command for first shape
        timeout_part = f"timeout {shape_timeout} " if shape_timeout else ""
        reproduce = (f"{timeout_part}{bench} -v --transA N --transB N "
                     f"-m 8192 -n 320 -k 320 "
                     + " ".join(common_args) + " --batch_count 1")

        shape_num = 0
        passed = 0
        failed = 0
        # Pin subprocess cwd to run_dir so any temp files the bench writes
        # (profiling traces, tuning caches) land in the per-test output
        # directory instead of the orchestrator's script dir.
        shape_cwd = ctx.run_dir if ctx.run_dir.is_dir() else None

        for (M, N, K, B) in NN_SHAPES:
            shape_num += 1
            print_header = (shape_num == 1)
            ok = self._run_shape(ctx, bench, "N", "N", M, N, K, B, shape_num, total_shapes,
                                 common_args, env, print_header, cwd=shape_cwd,
                                 timeout=shape_timeout)
            if ok:
                passed += 1
            else:
                failed += 1

        for (M, N, K, B) in NT_SHAPES:
            shape_num += 1
            ok = self._run_shape(ctx, bench, "N", "T", M, N, K, B, shape_num, total_shapes,
                                 common_args, env, print_header=False, cwd=shape_cwd,
                                 timeout=shape_timeout)
            if ok:
                passed += 1
            else:
                failed += 1

        print(f"  Completed: {passed}/{total_shapes} shapes passed, {failed} failed")
        return RunResult(exit_code=1 if failed > 0 else 0, reproduce_cmd=reproduce)

    # --- Helpers ---
    @staticmethod
    def _is_installed(rocm_root: Path) -> bool:
        p = rocm_root / "bin" / "hipblaslt-bench"
        return p.is_file() and os.access(p, os.X_OK)

    @classmethod
    def _find_bin(cls, config: Config) -> Optional[Path]:
        if cls._is_installed(config.rocm_root):
            return config.rocm_root / "bin" / "hipblaslt-bench"
        return None

    @staticmethod
    def _run_shape(ctx, bench, tA, tB, M, N, K, B, n, total, common_args, env_overrides,
                   print_header: bool, cwd: Optional[Path] = None,
                   timeout: Optional[int] = None) -> bool:
        # BLAS convention: leading dimension tracks the storage layout, which
        # flips under a transpose.
        #   transA == "N" → A stored M×K → lda = M
        #   transA == "T" → A stored K×M → lda = K
        #   transB == "N" → B stored K×N → ldb = K
        #   transB == "T" → B stored N×K → ldb = N
        # Strides are invariant to the transpose (M*K == K*M, K*N == N*K).
        # Passing the wrong ldb for transB="T" relied on hipblaslt-bench
        # silently correcting the invalid value — a stricter future build
        # (or a different GEMM engine) would fail outright.
        lda = M if tA == "N" else K
        ldb = K if tB == "N" else N
        cmd = [
            str(bench), "-v",
            "--transA", tA, "--transB", tB,
            "-m", str(M), "-n", str(N), "-k", str(K),
            "--lda", str(lda), "--stride_a", str(M * K),
            "--ldb", str(ldb), "--stride_b", str(K * N),
            "--ldc", str(M), "--stride_c", str(M * N),
            "--ldd", str(M), "--stride_d", str(M * N),
            *common_args,
            "--batch_count", str(B),
        ]
        env = dict(os.environ)
        env.update(env_overrides)
        executor = None
        if ctx.target_executor is not None:
            executor = ctx.target_executor
        elif ctx.monitor_executor is not None:
            executor = ctx.monitor_executor
        res = run_command_captured(
            executor,
            cmd,
            env=env,
            cwd=cwd,
            timeout=float(timeout) if timeout is not None else None,
        )
        # When an executor is wired, framework logging already copies
        # captured stdout into console.log; printing data rows here
        # duplicates them and breaks shape-identity validation.
        if res.exit_code == 124:
            print(f"  [hipblaslt] FAIL: watchdog timeout — shape {n}/{total} "
                  f"({tA}{tB} {M}x{N}x{K}x{B}) did not complete within "
                  f"--per-iter-watchdog {timeout}s")
            if res.stdout:
                print(res.stdout[-1000:])
            if res.stderr:
                print(res.stderr[-1000:])
            return False
        if res.exit_code != 0:
            # The "[hipblaslt] WARNING: shape" prefix is load-bearing:
            # ``_validate_hipblaslt`` counts occurrences of it to derive
            # shape_fail, which fails the test. Despite reading as a warning
            # this is a hard per-shape failure. Do not reword or re-label it
            # (e.g. WARNING -> ERROR) without updating the regex in
            # validation.py in the same commit -- on its own, a rename
            # silently drops the gate to zero matches.
            print(f"  [hipblaslt] WARNING: shape {n}/{total} ({tA}{tB} {M}x{N}x{K}x{B}) "
                  f"failed (exit {res.exit_code})")
            # print last 5 lines of output
            out_lines = (res.stdout + res.stderr).splitlines()[-5:]
            for line in out_lines:
                print(line)
            return False

        header_line = None
        data_line = None
        for line in (res.stdout + res.stderr).splitlines():
            if header_line is None and re.match(r"^\[\d+\]:transA", line):
                header_line = line
            if data_line is None and re.match(r"^\s+[NT],[NT],\d", line):
                data_line = line

        if print_header and header_line and executor is None:
            print(header_line)
        if data_line:
            if executor is None:
                print(data_line)
            return True
        else:
            # Same load-bearing prefix as the exit-code branch above; see the
            # note there before changing this wording.
            print(f"  [hipblaslt] WARNING: shape {n}/{total} ({tA}{tB} {M}x{N}x{K}x{B}) "
                  f"produced no data row")
            return False
