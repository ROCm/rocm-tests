# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""cudamemtest -- GPU memory integrity with power/temp monitoring."""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import time

from tests.common.gpu_monitored.config import Config
from tests.common.gpu_monitored.executor_bridge import format_shell_command, run_command, run_command_captured
from tests.common.gpu_monitored.workloads._memory_sizing import DEFAULT_FREE_VRAM_PCT, high_intensity_blocks_mb
from tests.common.gpu_monitored.workloads.base import BuildContext, BuildStatus, RunContext, RunResult, Test, TestSpec


class CudaMemtest(Test):
    spec = TestSpec(
        name="cudamemtest",
        goal="GPU memory robustness with power/temp monitoring",
        workload_profile={"min_util": 70, "min_vram_pct": 30},
    )

    COMMIT = "0cd3a996ce82682fcf50fa6f433b6f1f2ce1353d"
    PIN_GIT_TIMEOUT_SEC = 300

    # The enforced coverage contract. Upstream's registry (tests.cu) is
    # 0-indexed and marks IDs 0..8 and 10 enabled by default (9, "Bit fade",
    # is disabled because it hardcodes a 90-minute sleep). Running all ten at
    # the automatic 90%-of-VRAM target takes hours on a large-VRAM host --
    # test6 alone has been measured at roughly 32 s per GB per GPU, i.e. ~2 h
    # at 235 GB -- so the previous code simply ran as many as fit inside a
    # timer and reported the truncated prefix as PASS. That made the covered
    # algorithms differ per host and per run, and let IDs 3-8 and 10 break
    # without ever failing CI.
    #
    # Require the full algorithm prefix exercised by the pre-contract MI300A
    # baseline (tests 1-5), plus test0 which that baseline accidentally skipped.
    # This preserves the previous stress breadth at the same automatic target
    # while making coverage deterministic on every host. Tests 3-5 add 8-bit
    # and random moving-inversion patterns plus the 64-move block test.
    REQUIRED_SUBTESTS = (0, 1, 2, 3, 4, 5)

    @classmethod
    def _source_dir(cls, config: Config) -> Path:
        """Return cuda_memtest checkout (``output/external/...`` after framework remap)."""
        override = os.environ.get("ROCM_TEST_CUDA_MEMTEST_SRC", "").strip()
        if override:
            return Path(override)
        return config.build_dir / "cuda_memtest"

    def build(self, ctx: BuildContext) -> BuildStatus:  # noqa: C901
        dir_ = self._source_dir(ctx.config)
        ex = ctx.monitor_executor
        print("  [build] cuda_memtest: building...")
        if not (dir_ / ".git").is_dir():
            print("  [build] cuda_memtest: source missing — cuda_memtest_source " "fixture did not clone")
            return BuildStatus.BUILD_FAILED
        print("  [build] cuda_memtest: checking cached pinned source")

        def _reset_to_pin() -> int:
            return run_command(
                ex,
                ["git", "reset", "--hard", self.COMMIT],
                cwd=dir_,
                timeout=self.PIN_GIT_TIMEOUT_SEC,
            )

        if _reset_to_pin() != 0:
            print(f"  [build] cuda_memtest: {self.COMMIT[:12]} not present; " f"fetching it explicitly")
            fetch_rc = run_command(
                ex,
                ["git", "fetch", "--depth", "1", "origin", self.COMMIT],
                cwd=dir_,
                timeout=self.PIN_GIT_TIMEOUT_SEC,
            )
            if fetch_rc != 0 or _reset_to_pin() != 0:
                print(
                    f"  [build] cuda_memtest: FAILED — cannot check out the "
                    f"pinned commit {self.COMMIT}. Upstream may have "
                    f"force-pushed or pruned it. Refusing to build "
                    f"origin/HEAD, which would change the algorithms and "
                    f"output syntax this test validates."
                )
                return BuildStatus.BUILD_FAILED

        clean_rc = run_command(
            ex,
            ["git", "clean", "-dfx"],
            cwd=dir_,
            timeout=self.PIN_GIT_TIMEOUT_SEC,
        )
        if clean_rc != 0:
            print("  [build] cuda_memtest: FAILED — cannot clean cached source")
            return BuildStatus.BUILD_FAILED

        head_res = run_command_captured(
            ex,
            ["git", "rev-parse", "HEAD"],
            cwd=dir_,
            timeout=self.PIN_GIT_TIMEOUT_SEC,
        )
        head = head_res.stdout.strip() if head_res.exit_code == 0 else ""
        if head != self.COMMIT:
            print(f"  [build] cuda_memtest: FAILED — HEAD is {head or 'unknown'}, " f"expected pinned {self.COMMIT}")
            return BuildStatus.BUILD_FAILED
        print(f"  [build] cuda_memtest: pinned at {self.COMMIT[:12]}")
        (dir_ / "cuda_memtest").unlink(missing_ok=True)

        hipify = ctx.rocm_root / "bin" / "hipify-perl"
        for pattern in ("cuda_memtest.*", "misc.*", "tests.cu"):
            for f in dir_.glob(pattern):
                if not f.is_file():
                    continue
                tmp = dir_ / f"hip_{f.name}"
                res = run_command_captured(ex, [str(hipify), str(f)])
                if res.exit_code == 0:
                    tmp.write_text(res.stdout)
                    f.unlink()
                    tmp.rename(f)

        header = dir_ / "cuda_memtest.h"
        if header.is_file():
            content = header.read_text()
            new = content.replace("MEMTEST_PP_CONCAT_DO(cuda, name)", "MEMTEST_PP_CONCAT_DO(hip, name)")
            if new != content:
                header.write_text(new)

        for src_name in ("cuda_memtest.cu", "cuda_memtest.cpp"):
            src = dir_ / src_name
            if src.is_file():
                content = src.read_text()
                new = content.replace("hipHostGetDevicePointer(", "hipHostGetDevicePointer((void **)")
                if new != content:
                    src.write_text(new)
                break

        hipcc = ctx.rocm_root / "bin" / "hipcc"
        stderr_path = dir_ / "build.stderr.log"
        cu_file = dir_ / "cuda_memtest.cu"
        if cu_file.is_file():
            srcs = ["cuda_memtest.cu", "misc.cpp", "tests.cu"]
        else:
            srcs = ["cuda_memtest.cpp", "misc.cpp", "tests.cpp"]

        build_cmd = format_shell_command(
            [str(hipcc), "-DENABLE_NVML=0", *srcs, "-o", "cuda_memtest"],
            cwd=dir_,
        )
        stderr_q = shlex.quote(str(stderr_path.resolve()))
        build_rc = run_command(ex, f"{build_cmd} 2> {stderr_q}", timeout=600)

        if not (dir_ / "cuda_memtest").is_file():
            print(f"  [build] cuda_memtest: FAILED — binary not produced " f"(hipcc rc={build_rc}); see {stderr_path}")
            return BuildStatus.BUILD_FAILED
        print("  [build] cuda_memtest: OK")
        return BuildStatus.OK

    def available(self, config: Config) -> bool:
        return (self._source_dir(config) / "cuda_memtest").is_file()

    def run(self, ctx: RunContext) -> RunResult:
        bin_ = self._source_dir(ctx.config) / "cuda_memtest"
        # The automatic target retains high memory pressure without consuming
        # every free byte (which can starve HSA runtime bookkeeping). An
        # explicit operator cap still wins.
        blocks = str(ctx.config.memtest_blocks).strip()
        if not blocks:
            auto_blocks = high_intensity_blocks_mb(
                ctx.rocm_root,
                max(1, getattr(ctx.config, "num_gpus", 1)),
                host_backed=False,
                monitor_executor=ctx.monitor_executor,
            )
            if auto_blocks is None:
                print(
                    "  [cudamemtest] FAIL: cannot size the automatic memory "
                    "target from amd-smi; pass --memtest-blocks explicitly"
                )
                return RunResult(exit_code=1)
            blocks = str(auto_blocks)
            print(
                f"  [cudamemtest] Auto memory target: {blocks} MB/GPU "
                f"({DEFAULT_FREE_VRAM_PCT}% of the least-free GPU)"
            )
        extra_args = ["--max_num_blocks", blocks]

        # Test9 ("Bit fade") hardcodes ``sleep(60*90)`` (90 min) in
        # upstream ``cuda_memtest/tests.cu``. Cost is independent of
        # VRAM size; including it adds 90 min to every run. Excluded
        # by default; ``--include-bit-fade`` opts in for genuine bit-
        # decay validation runs. ``getattr`` default keeps unit-test
        # fixtures (which build minimal ``SimpleNamespace`` configs)
        # working without forcing them to pass every new attribute.
        # Coverage is a contract, not whatever fit before a timer expired.
        # ``REQUIRED_SUBTESTS`` must complete in full or the test FAILs, so the
        # same algorithms run on every host and a truncated prefix can no
        # longer be reported as PASS.
        sub_tests = list(self.REQUIRED_SUBTESTS)
        if getattr(ctx.config, "include_bit_fade", False):
            sub_tests.append(9)

        extra_str = (" " + " ".join(extra_args)) if extra_args else ""
        # Reflect ``--per-iter-watchdog`` in the reproduce string so a
        # copy-paste run mirrors what the harness actually executed
        # (consistent with ``hmm_cuda_memtest`` and ``transferbench``).
        wd = ctx.config.per_iter_watchdog or 0
        timeout_part = f"timeout {wd} " if wd else ""
        repro_subtests = " ".join(str(n) for n in sub_tests)
        reproduce = (
            f"# cuda_memtest sub-tests {repro_subtests} "
            f"(sub-test start budget {ctx.config.memtest_duration}s):\n"
            f"  for n in {repro_subtests}; do\n"
            f"    {timeout_part}{bin_} --disable_all --enable_test $n --num_passes 1"
            f"{extra_str} || exit 1;\n"
            f"  done"
        )
        if ctx.config.memtest_duration < 1:
            print("  [cudamemtest] ERROR: --memtest-duration must be >= 1")
            return RunResult(exit_code=1, reproduce_cmd=reproduce)

        # Monotonic clock — immune to NTP steps during the run.
        deadline = time.monotonic() + ctx.config.memtest_duration

        first_fail_rc: int = 0
        ran = 0
        # Per-sub-test watchdog is opt-in (default 0 = no kill, sub-test
        # runs to completion). The cuda_memtest binary self-bounds via
        # ``--num_passes 1``; on a sane GPU that's the only bound we
        # need. Test6 ("Moving inversions, 32-bit pattern") in
        # particular has been observed to take tens of minutes on
        # high-VRAM HBM2e multi-GPU hosts (e.g. MI210 8x64 GB), so any
        # fixed floor short of that turned a healthy slow workload
        # into a false FAIL. CI / wedged-GPU detection now passes
        # ``--per-iter-watchdog <SEC>`` explicitly (recommended ≥ 3600
        # for high-VRAM gfx90a hosts; see the README's host-class
        # table for current guidance).
        per_iter_timeout = ctx.config.per_iter_watchdog or None
        for test_num in sub_tests:
            if time.monotonic() >= deadline:
                break
            cmd = [str(bin_), "--disable_all", "--enable_test", str(test_num), "--num_passes", "1", *extra_args]
            iter_start = time.monotonic()
            rc = ctx.exec(cmd, timeout=per_iter_timeout)
            iter_dur = time.monotonic() - iter_start
            ran += 1
            # Per-sub-test elapsed surfaced unconditionally so operators
            # triaging "why did the loop only finish N/M sub-tests?" can
            # see which sub-test is slow without grepping ``console.log``
            # for the binary's own per-GPU "Test$n finished in ..." lines.
            print(f"  [cudamemtest] enable_test {test_num} " f"finished in {iter_dur:.1f}s (rc={rc})")
            if rc == 124:
                # Watchdog timeout on a single sub-test — almost
                # always a wedged GPU. Stop trying further sub-tests
                # (they'd just hang the same way) and surface a clean
                # FAIL with an actionable message. Same convention as
                # transferbench / sln_stress / hmm_cuda_memtest.
                print(
                    f"  [cudamemtest] FAIL: watchdog timeout — "
                    f"enable_test {test_num} did not complete within "
                    f"--per-iter-watchdog {per_iter_timeout}s. GPU is "
                    f"likely wedged; stopping further sub-tests."
                )
                if first_fail_rc == 0:
                    first_fail_rc = 1
                break
            # First non-zero rc wins; continue running the remaining
            # sub-tests within the budget so monitoring accumulates samples
            # but never forget that an earlier sub-test failed.
            if rc != 0 and first_fail_rc == 0:
                first_fail_rc = rc
                print(f"  [cudamemtest] enable_test {test_num} exited " f"rc={rc} — continuing remaining sub-tests")

        print(f"  [cudamemtest] Ran {ran}/{len(sub_tests)} sub-test(s), " f"first_fail_rc={first_fail_rc}")
        if ran < len(sub_tests):
            # The required set is the coverage contract: reporting a truncated
            # prefix as PASS would let the sub-tests that never started break
            # unnoticed. Name the missing IDs and how to get them to run.
            missing = " ".join(str(n) for n in sub_tests[ran:])
            print(
                f"  [cudamemtest] FAIL: incomplete coverage — {ran} of "
                f"{len(sub_tests)} required sub-tests ran; {missing} never "
                f"started within --memtest-duration "
                f"{ctx.config.memtest_duration}s. Raise --memtest-duration "
                f"or lower the footprint with --memtest-blocks."
            )
            if first_fail_rc == 0:
                first_fail_rc = 1
        return RunResult(exit_code=first_fail_rc, reproduce_cmd=reproduce)
