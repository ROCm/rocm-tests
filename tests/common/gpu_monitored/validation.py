# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""5-layer result validation:

Layer 1: Crash / kernel indicators in test output (all tests)
Layer 2: Test-specific log parsing (per-test function)
Layer 3: dmesg delta — critical kernel events (GPU reset, panic, lockup,
         fault) FAIL the test; informational events are annotated only
Layer 4: Empty log + non-zero exit → promoted to FAIL
Layer 5: Exit-code consistency — non-zero exits are surfaced in the
         validation message even if earlier layers already flagged FAIL
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from framework.executors.cpu_executor import CpuExecutor

import contextlib

from tests.common.gpu_monitored.dmesg_capture import capture_dmesg_text
from tests.common.gpu_monitored.workloads.cudamemtest import CudaMemtest
from tests.common.gpu_monitored.workloads.hipblaslt_bench import NN_SHAPES, NT_SHAPES

ERROR_RATE_FAIL_PCT = 1.0


# Layer 1 patterns. Anchored with word boundaries (or strict prefixes) so
# incidental substrings don't trip a false FAIL:
#   - "core dumped"  must not match "score dumped"
#   - "Aborted"      must not match "aborted here" vs. "test not aborted"
#   - "BUG:"         must not match userspace "debug:" lines in dmesg
# Case-sensitive: these come from libc/kernel with a stable case. The
# earlier re.IGNORECASE on these expanded matches into many false positives.
# Action names declared by the upstream RVS ``tst_single.conf``. RVS logs one
# ``[<action>]`` per configured action per GPU, so this set is what proves a
# run covered the whole configuration rather than half of it.
RVS_TST_EXPECTED_ACTIONS = {"action_1", "action_2"}

CRASH_PATTERNS = re.compile(
    r"\b(?:core dumped|Aborted|Segmentation fault|Kernel panic|"
    r"general protection fault|corrupted memory|resetting device|"
    r"device not responding|hardware error)\b|"
    r"\bCall Trace:|"
    r"\bEXIT_CODE=134\b|"
    r"watchdog: BUG: soft lockup|"
    r"watchdog: CPU stuck"
)

# Critical dmesg patterns — these FAIL the test when found in the dmesg
# delta (new kernel messages during the test window). Aligned with
# ROCmTestInternal's ``system_health_watchdog.KERNEL_ALERT_RULES``
# critical categories. A GPU reset, kernel panic, or watchdog lockup
# during a test means the hardware was unhealthy regardless of what the
# test's own stdout reported.
# NOTE on pattern precision: the 10x-run validation surfaced two
# false-positive classes that required tightening:
#
#  1. ``clocksource: timekeeping watchdog`` is the kernel's HPET
#     sanity checker, NOT a CPU/GPU lockup. The bare ``\bwatchdog\b``
#     pattern matched it. Fixed by requiring ``watchdog:`` followed
#     by ``BUG`` or ``soft lockup`` / ``hard lockup``.
#
#  2. OOM-killer's ``Call Trace:`` fires on container memory-cgroup
#     kills that are already surfaced by the test's own exit=-9.
#     We keep it as critical because any kernel call trace during a
#     GPU test is worth flagging, but operators should know that an
#     OOM-kill call trace is expected in memory-constrained containers.
DMESG_CRITICAL = re.compile(
    r"\bKernel panic\b|"
    r"\bpanic\b|"
    r"watchdog:.*\b(?:BUG|soft lockup|hard lockup)\b|"
    r"\bsoft lockup\b|"
    r"\bhard lockup\b|"
    r"\bBUG:|"  # ``\bBUG:\b`` doesn't match "BUG: " (space after colon)
    r"\bkernel BUG\b|"
    r"Oops:\s*[0-9a-fA-F]+|"  # "Oops: 0000 [#1]" — driver/kernel oops
    r"\bCall Trace\b|"
    r"\bcall trace\b|"
    r"\bblocked for more than\b|"
    r"\bhung task\b|"
    r"\bgpu reset\b|"
    r"\bamdgpu.*reset\b|"
    r"\bring .*timeout\b|"
    r"\bGPU fault\b|"
    r"\bRAS\b.*\berror\b|"
    r"\b(?:uncorrected|uncorrectable|fatal)\b.*\b(?:hardware error|ECC|RAS)\b|"
    r"\b(?:hardware error|ECC|RAS)\b.*\b(?:uncorrected|uncorrectable|fatal)\b|"
    r"\bIOMMU\b.*\bfault\b|"
    r"\bDMAR\b.*\bfault\b|"
    r"\bsegfault\b|"
    r"\bsegmentation fault\b",
    re.IGNORECASE,
)

# Informational dmesg patterns — annotated in health_checks / summary
# but do NOT flip pass/fail. Covers general hardware events that are
# worth noting but are not necessarily test-breaking (e.g. ECC
# single-bit corrections, page faults that the driver handled, general
# protection faults in unrelated processes).
DMESG_INFO = re.compile(
    r"\b(?:general protection fault|"
    r"resetting device|device not responding|hardware error|"
    r"page fault|ECC)\b|"
    r"\bRAS\b.*\b(?:correctable|corrected)\b",
    re.IGNORECASE,
)

RAS_CORRECTED_ONLY = re.compile(
    r"\b(?:RAS|hardware error|ECC)\b.*\b(?:correctable|corrected)\b",
    re.IGNORECASE,
)

RAS_UNCORRECTABLE = re.compile(
    r"\b(?:uncorrected|uncorrectable|fatal)\b",
    re.IGNORECASE,
)

RAS_NON_ACTIONABLE_UNCORRECTABLE = re.compile(
    r"\b(?:no|zero|0)\s+(?:uncorrected|uncorrectable|fatal)"
    r"(?:\s+ECC)?\s+errors?\b|"
    r"\b(?:uncorrected|uncorrectable|fatal)(?:\s+ECC)?\s+errors?"
    r"\s*(?::|=)?\s*0\b",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    message: str  # validation message (e.g. "PASS (19 GEMM results)")
    failed: bool  # True if validation detected failure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_log(log_file: Path) -> str:
    """Read ``log_file`` as text, replacing encoding errors.

    Returns a sentinel ``_READ_FAILED:`` prefix when the file exists but
    can't be read (FS error, permissions, ...) so callers can distinguish
    "empty log" from "unreadable log". Without this, a transient read
    failure would return ``""`` and validation could silently fall through
    to a PASS path (no crash patterns, no Layer 2 message, and Layer 4
    treats an empty string + exit 0 as PASS).
    """
    try:
        return log_file.read_text(errors="replace")
    except Exception as e:
        return f"_READ_FAILED: {type(e).__name__}: {e}"


def ras_downgrade(line: str) -> str | None:
    """Normalise a dmesg line for RAS severity, or ``None`` to ignore it.

    Strips the "0 uncorrected errors" style phrasing that would otherwise
    read as an uncorrected event, then drops the line entirely when what
    remains reports only corrected/correctable errors -- those are
    informational and must not fail a run.

    Shared deliberately: the FAIL gate (``_count_lines_matching`` over
    ``DMESG_CRITICAL``) and the triage breakdown
    (``categorize_dmesg_critical``, which feeds ``pretest_health.json``)
    each carried their own copy of this. Editing one without the other
    would have let the per-category counts a triager reads disagree with
    the decision the harness actually made.
    """
    candidate = RAS_NON_ACTIONABLE_UNCORRECTABLE.sub("", line)
    if RAS_CORRECTED_ONLY.search(candidate) and not RAS_UNCORRECTABLE.search(candidate):
        return None
    return candidate


def _count_lines_matching(text: str, pattern: re.Pattern) -> int:
    count = 0
    for line in text.splitlines():
        candidate = line
        if pattern is DMESG_CRITICAL:
            downgraded = ras_downgrade(line)
            if downgraded is None:
                continue
            candidate = downgraded
        if pattern.search(candidate):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Per-test Layer 2 validators
# ---------------------------------------------------------------------------
def _validate_memtest(  # noqa: C901
    log_file: Path,
    exit_code: int,
    *,
    require_coverage: bool = False,
) -> ValidationResult:
    """cudamemtest + hmm_cuda_memtest

    The earlier pattern ``^.*ERROR:`` matched every line containing the
    substring ``ERROR:`` — including benign lines such as ``"No ERROR:"``
    or ``"Test for ERROR: false"``. Anchor the tool-emitted error
    markers more precisely:

    - ``[ERROR]`` — the bracketed prefix cuda_memtest actually uses
      (``printf("[ERROR] ...")``), including when wrapped with the
      per-GPU tag (``[0] test2: [ERROR] ...``).
    - ``ERROR:`` at line start **or immediately after the per-GPU tag**.
      The pinned upstream emits every memory error through ``PRINTF`` /
      ``FPRINTF`` (e.g. ``FPRINTF("ERROR: (%s) %d errors found in block
      %d\\n", ...)``), and those macros prepend ``[%s][%s][%d]:`` —
      ``[%s][%s][%d][%d C]:`` under ``--monitor_temp``. A real error line
      is therefore ``[time][host][gpu]:ERROR: ...``, where ``ERROR:`` is
      mid-line and unbracketed. A line-anchored ``^\\s*ERROR:`` misses it
      completely, so genuine memory corruption was reported as
      ``PASS (0 memory errors)`` — a memory test that could not fail on
      memory errors. Anchoring on ``]:`` as well as line start restores
      detection while still rejecting the false positives below.
    - ``Memory access fault`` / ``HIP error:`` — HIP runtime errors
      wherever they appear.

    This catches the real cuda_memtest error lines while ignoring the
    "No ERROR: ..." / "Test for ERROR: false" false positives.
    """
    text = _read_log(log_file)
    if not text:
        return ValidationResult("", False)
    # Watchdog timeout sentinels emitted by the per-test ``run`` paths
    # when ``ctx.exec`` returns rc=124 (subprocess.TimeoutExpired
    # bubbled up). Both hmm_cuda_memtest (single-invocation timeout)
    # and cudamemtest (per-iter timeout) flag this as FAIL — surface
    # the message in the validation column rather than letting the
    # generic "0 memory errors" path mask a wedged GPU.
    if re.search(r"\[hmm_memtest\] FAIL: watchdog timeout", text):
        return ValidationResult("watchdog timeout (HMM did not complete)", True)
    if re.search(r"\[cudamemtest\] FAIL: watchdog timeout", text):
        return ValidationResult("watchdog timeout (sub-test did not complete)", True)
    err_pat = re.compile(
        r"\[ERROR\]\s|"  # cuda_memtest's bracketed form
        # Colon form at line start OR right after the ``[time][host][gpu]:``
        # / ``[time][host][gpu][NN C]:`` tag every PRINTF/FPRINTF emits.
        r"(?:^|\]:)\s*ERROR:\s+\S|" r"\bMemory access fault\b|" r"\bHIP error:",
        re.IGNORECASE | re.MULTILINE,
    )
    err_count = len(err_pat.findall(text))
    if err_count > 0:
        return ValidationResult(f"{err_count} memory/runtime errors", True)
    # ``cudamemtest`` declares a required sub-test set and must complete all
    # of it (the denominator grows by one under ``--include-bit-fade``).
    # Accepting any ``Ran N/M`` as PASS meant a run that started only some of
    # the required algorithms still reported success, so the ones that never
    # started could break without failing CI. Require N == M.
    ran_m = re.search(r"\[cudamemtest\]\s+Ran\s+(\d+)/(\d+)\s+sub-test", text)
    if ran_m:
        ran = int(ran_m.group(1))
        total = int(ran_m.group(2))
        # ``Ran 0/0`` would satisfy ``ran >= total`` and report PASS on a run
        # that executed nothing -- the same vacuous-denominator hole this
        # validator closes elsewhere (``no RVS pass/fail results``, ``no
        # transfer results``, ``no power bands completed``). A zero
        # denominator means the coverage line itself is wrong, so treat it as
        # a failure rather than trusting it.
        if total <= 0:
            return ValidationResult(
                f"coverage line reports a zero denominator ({ran}/{total} "
                f"sub-tests); cannot confirm any sub-test ran",
                True,
            )
        if ran < total:
            return ValidationResult(
                f"incomplete coverage ({ran}/{total} required sub-tests ran)",
                True,
            )
        if ran > total:
            return ValidationResult(
                f"inconsistent coverage ({ran}/{total} required sub-tests ran; " f"expected exact equality)",
                True,
            )
        expected = list(CudaMemtest.REQUIRED_SUBTESTS)
        if total == len(expected) + 1:
            expected.append(9)
        elif total != len(expected):
            # Every accepted denominator must map to a known required set, so
            # the identity check below can verify *which* sub-tests ran. A
            # denominator we cannot map is a coverage line we cannot trust.
            return ValidationResult(
                f"unexpected cudamemtest coverage denominator {total}; "
                f"expected {len(CudaMemtest.REQUIRED_SUBTESTS)} or "
                f"{len(CudaMemtest.REQUIRED_SUBTESTS) + 1}",
                True,
            )
        completed = [
            (int(test_id), int(rc))
            for test_id, rc in re.findall(
                r"\[cudamemtest\]\s+enable_test\s+(\d+)\s+finished\s+" r"in\s+[\d.]+s\s+\(rc=(-?\d+)\)",
                text,
            )
        ]
        completed_ids = [test_id for test_id, _rc in completed]
        if completed_ids != expected or any(rc != 0 for _test_id, rc in completed):
            return ValidationResult(
                "incomplete cudamemtest identity coverage " f"(completed IDs {completed_ids}, expected {expected})",
                True,
            )
        return ValidationResult(
            f"PASS ({ran}/{total} sub-tests, 0 memory errors)",
            False,
        )
    if require_coverage:
        return ValidationResult(
            "missing cudamemtest coverage summary (expected Ran N/M sub-tests)",
            True,
        )
    return ValidationResult("PASS (0 memory errors)", False)


def _validate_cudamemtest(log_file: Path, exit_code: int) -> ValidationResult:
    return _validate_memtest(log_file, exit_code, require_coverage=True)


def _validate_hmm_memtest(
    log_file: Path,
    exit_code: int,
    expected_gpus: int | None = None,
) -> ValidationResult:
    result = _validate_memtest(log_file, exit_code, require_coverage=False)
    if result.failed:
        return result
    text = _read_log(log_file)
    completions = re.findall(
        r"\[[^\]\r\n]+\]\[[^\]\r\n]+\]\[(\d+)\]:" r"Test0 finished in [\d.]+ seconds\b",
        text,
    )
    if expected_gpus is not None:
        expected = {str(gpu) for gpu in range(expected_gpus)}
        if len(completions) != expected_gpus or set(completions) != expected:
            return ValidationResult(
                "incomplete HMM per-GPU completion coverage "
                f"(reported {sorted(set(completions))}, "
                f"expected {sorted(expected)})",
                True,
            )
    elif not completions and not re.search(
        r"\bTest0 finished in [\d.]+ seconds\b",
        text,
    ):
        return ValidationResult("missing HMM Test0 completion marker", True)
    return result


def _validate_rvs(  # noqa: C901
    log_file: Path,
    exit_code: int,
    expected_checks: int | None = None,
    expected_gpus: int | None = None,
    checks_per_gpu: int | None = None,
    expected_actions: set[str] | None = None,
) -> ValidationResult:
    """rvs_iet_stress + rvs_tst

    RVS emits one ``pass: TRUE|FALSE`` line per (GPU, action), so the
    denominator we report is the number of checks, not the number of
    GPUs. We also anchor ``ABORT`` on RVS's own line prefix so libc
    ``abort()`` text elsewhere in the log can't promote the run to
    FAIL.
    """
    text = _read_log(log_file)
    if not text:
        return ValidationResult("no RVS output", True)
    if re.search(
        r"\[rvs_(?:iet_stress|tst)\] FAIL: watchdog timeout",
        text,
    ):
        return ValidationResult("watchdog timeout (RVS did not complete)", True)
    # RVS prints the TRUE/FALSE tokens in uppercase; case-sensitive match
    # is enough and avoids matching unrelated "pass: true" narratives.
    true_count = len(re.findall(r"\bpass:\s*TRUE\b", text))
    false_count = len(re.findall(r"\bpass:\s*FALSE\b", text))
    # Case-sensitive so libc's lowercase "abort()" in a Python traceback
    # or a "double free, aborting" message doesn't falsely trigger.
    abort_count = len(
        re.findall(
            r"(?m)^\s*\[(?:RESULT|ERROR)\].*\bABORT\b",
            text,
        )
    )
    total = true_count + false_count

    if abort_count > 0:
        return ValidationResult("ABORT detected", True)
    if total > 0:
        if false_count > 0:
            return ValidationResult(f"{false_count}/{total} GPU check(s) failed", True)
        if expected_checks is not None and total != expected_checks:
            return ValidationResult(
                f"incomplete RVS coverage ({total}/{expected_checks} expected " "GPU checks reported)",
                True,
            )
        if expected_gpus is not None and checks_per_gpu is not None:
            result_rows = re.findall(
                r"(?m)^\s*\[RESULT\]\s+\[[^\]\r\n]+\]\s+"
                r"\[([^\]\r\n]+)\]\s+\[GPU::\s*([^\]\r\n]+?)\s*\]"
                r"\s+pass:\s*(?:TRUE|FALSE)\b",
                text,
            )
            if len(result_rows) != total:
                return ValidationResult(
                    f"incomplete RVS identity evidence ({len(result_rows)}/"
                    f"{total} result lines identify action and GPU)",
                    True,
                )
            identities = [(action.strip(), gpu.strip()) for action, gpu in result_rows]
            if len(set(identities)) != len(identities):
                return ValidationResult(
                    "duplicate RVS action/GPU result identity detected",
                    True,
                )
            actions = {action for action, _gpu in identities}
            gpus = {gpu for _action, gpu in identities}
            if expected_actions is not None and actions != expected_actions:
                return ValidationResult(
                    "unexpected RVS action coverage "
                    f"({sorted(actions)} reported, "
                    f"expected {sorted(expected_actions)})",
                    True,
                )
            if len(actions) != checks_per_gpu or len(gpus) != expected_gpus:
                return ValidationResult(
                    "incomplete RVS identity coverage "
                    f"({len(gpus)}/{expected_gpus} GPUs, "
                    f"{len(actions)}/{checks_per_gpu} actions)",
                    True,
                )
            if any((action, gpu) not in set(identities) for action in actions for gpu in gpus):
                return ValidationResult(
                    "incomplete RVS action/GPU result matrix",
                    True,
                )
        return ValidationResult(f"PASS ({true_count}/{total} GPU checks passed)", False)
    return ValidationResult("no RVS pass/fail results found in output", True)


def _validate_sln_stress(log_file: Path, exit_code: int) -> ValidationResult:
    # The runner always tees the test's stdout/stderr to ``console.log``
    # (see ``runner._run_workload``) so the previous "file missing" branch
    # was dead in practice and read as "pass if no file" — a footgun if
    # a future runner ever omitted the file. Fall through to the empty
    # string + promotion-to-FAIL path in ``validate_result`` (Layer 4)
    # if for some reason the file really isn't there.
    text = _read_log(log_file)
    if not text:
        return ValidationResult("no SLN output", True)
    if "[sln_stress] FAIL: watchdog timeout" in text:
        return ValidationResult(
            "watchdog timeout (iteration did not complete)",
            True,
        )
    hip_err = len(re.findall(r"HIP error:", text, re.IGNORECASE))
    if hip_err > 0:
        return ValidationResult(f"{hip_err} HIP error(s)", True)
    completed = re.search(
        r"\[sln_stress\] Completed (\d+) iteration\(s\).*exit=(\d+)",
        text,
    )
    if completed is None or int(completed.group(1)) < 1:
        return ValidationResult("missing SLN completion summary", True)
    return ValidationResult("PASS (no HIP errors)", False)


def _validate_power_band(log_file: Path, exit_code: int) -> ValidationResult:  # noqa: C901
    text = _read_log(log_file)
    if not text:
        return ValidationResult("no power-band output", True)

    # Config errors (invalid --power-bands, empty bands) — short-circuit
    # with a useful message instead of reporting "PASS (0 bands cycled)".
    if re.search(r"\[power_band\] ERROR: Invalid --power-bands", text):
        return ValidationResult("invalid --power-bands configuration", True)
    if re.search(r"\[power_band\] ERROR: --power-bands is empty", text):
        return ValidationResult("--power-bands is empty", True)

    restore_failed = re.search(
        r"\[power_band\] ERROR: Failed to restore original power caps " r"on (\d+)/(\d+) GPU\(s\)",
        text,
    )
    if restore_failed:
        return ValidationResult(
            f"power-cap restore failed on {restore_failed.group(1)}/" f"{restore_failed.group(2)} GPUs",
            True,
        )

    # Surface which band number the workload died on — the single most
    # useful diagnostic for debugging power_band_stress failures. Also
    # matches the settle-period failure (no band number).
    settle_died = re.search(r"\[power_band\] ERROR: Workload exited during settle period", text)
    if settle_died:
        return ValidationResult("workload exited during settle period", True)

    timeout_hit = re.search(r"\[power_band\] ERROR: Workload hit watchdog timeout after (\d+)s", text)
    if timeout_hit:
        return ValidationResult(f"workload hit watchdog timeout after {timeout_hit.group(1)}s", True)

    band_died = re.search(r"\[power_band\] ERROR: Workload died during\s+band (\d+)(?:/(\d+))? " r"\(([^)]+)\)", text)
    if band_died:
        n, total, pct = band_died.group(1), band_died.group(2), band_died.group(3)
        suffix = f"/{total}" if total else ""
        return ValidationResult(f"workload died on band {n}{suffix} ({pct})", True)

    completed = re.findall(
        r"\[power_band\] Band (\d+)/(\d+) complete",
        text,
    )
    bands_applied = len(completed)
    partial_apply = re.search(
        r"\[power_band\] ERROR: Power cap changes failed on (\d+)/(\d+) band\(s\)",
        text,
    )
    if partial_apply:
        failed_bands, total_bands = partial_apply.group(1), partial_apply.group(2)
        return ValidationResult(
            f"power-cap changes failed on {failed_bands}/{total_bands} bands",
            True,
        )
    if completed:
        totals = {int(total) for _, total in completed}
        band_numbers = [int(number) for number, _ in completed]
        if len(totals) != 1:
            return ValidationResult(
                "inconsistent power-band denominators in output",
                True,
            )
        expected = totals.pop()
        expected_order = list(range(1, expected + 1))
        if expected <= 0 or band_numbers != expected_order:
            return ValidationResult(
                f"invalid power-band coverage sequence " f"({band_numbers!r}; expected {expected_order!r})",
                True,
            )
    if bands_applied == 0 and exit_code == 0:
        return ValidationResult("no power bands completed", True)

    # The band schedule can complete while the *workload's* own end-of-run
    # thresholds (RCCL/rocBLAS presence, kernel counts, step variance, NaN
    # telemetry) reject the run. Left to the generic path this became
    # "exit N; workload OK (M bands cycled)" -- accurate about the cycling
    # but pointing away from the thing that actually failed, since here it
    # is the workload's validator that objected. Name it and name the
    # report to open. Checked last so the cap-restore and band-coverage
    # failures above, which are more specific, still win.
    wl_failed = re.search(
        r"\[power_band\] ERROR: Workload's own validation failed " r"\(exit (-?\d+)\)",
        text,
    )
    if wl_failed:
        return ValidationResult(
            f"workload's own validation failed (exit {wl_failed.group(1)}); "
            f"{bands_applied} bands cycled — see "
            f"validator_results/stress_validation_report.json",
            True,
        )
    return ValidationResult(f"PASS ({bands_applied} bands cycled)", False)


def _validate_inference_server(log_file: Path, exit_code: int) -> ValidationResult:  # noqa: C901
    """inference_server_stress"""
    text = _read_log(log_file)
    if not text:
        return ValidationResult("", False)
    # Server never became healthy → hard fail
    if re.search(r"\[inference_server_stress\] ERROR: server did not become healthy", text):
        return ValidationResult("server never became healthy", True)
    # Explicit error-rate / stall decisions made by the test body
    m = re.search(r"\[inference_server_stress\] ERROR: error rate " r"(\d+\.\d+)%", text)
    if m:
        return ValidationResult(f"error rate {m.group(1)}%", True)
    m = re.search(r"\[inference_server_stress\] ERROR: overlap stalls " r"detected \((\d+) >= (\d+)\)", text)
    if m:
        return ValidationResult(f"{m.group(1)} overlap stalls (threshold {m.group(2)})", True)
    # /metrics scrape-side decisions made by the test body. These
    # diagnose two distinct failure modes that the test deliberately
    # split apart so operators don't waste triage time on the wrong
    # cause; the validator must surface them too or Layer 5 falls
    # back to a bare "exit 1" and the diagnostic split is wasted.
    if re.search(r"\[inference_server_stress\] ERROR: /metrics endpoint " r"was unreachable", text):
        return ValidationResult("metrics endpoint unreachable", True)
    if re.search(r"\[inference_server_stress\] ERROR: metrics sampler " r"wrote no rows", text):
        return ValidationResult("metrics sampler produced no rows", True)
    if re.search(r"\[inference_server_stress\] ERROR: /metrics parser " r"recognised none of the metric names", text):
        return ValidationResult("metrics regex drift vs vLLM build", True)
    if re.search(r"\[inference_server_stress\] ERROR: zero overlap samples", text):
        return ValidationResult("zero overlap samples — prefill/decode overlap was not exercised", True)
    if re.search(r"\[inference_server_stress\] ERROR: load phase produced " r"zero request results", text):
        return ValidationResult("load phase produced zero requests", True)
    # Server-side failures in the tee'd log (HIP crashes, CUDA errors).
    # Case-insensitive for symmetry with _validate_memtest /
    # _validate_hipblaslt; if a torch fork or vLLM build emits
    # lowercase "hip error:", we want to catch it the same way the
    # other validators would.
    hip_err = len(re.findall(r"\bHIP error:|\bCUDA error:", text, flags=re.IGNORECASE))
    if hip_err > 0:
        return ValidationResult(f"{hip_err} HIP/CUDA error(s) in server log", True)
    # Happy-path summary line
    m = re.search(r"\[inference_server_stress\] Requests: (\d+) \((\d+) errors, " r"([\d.]+)% error rate\)", text)
    if m:
        total, errs, rate = m.group(1), m.group(2), m.group(3)
        total_count = int(total)
        error_count = int(errs)
        reported_rate = float(rate)
        if total_count == 0:
            return ValidationResult("load phase produced zero requests", True)
        calculated_rate = error_count / total_count * 100.0
        if abs(reported_rate - calculated_rate) > 0.01:
            return ValidationResult(
                "inconsistent inference error summary "
                f"({error_count}/{total_count} implies {calculated_rate:.2f}%, "
                f"reported {reported_rate:.2f}%)",
                True,
            )
        if calculated_rate >= ERROR_RATE_FAIL_PCT:
            return ValidationResult(
                f"inference error rate {calculated_rate:.2f}% >= "
                f"{ERROR_RATE_FAIL_PCT:.2f}% "
                f"({error_count}/{total_count} requests failed)",
                True,
            )
        metrics_line = re.search(
            r"\[inference_server_stress\] /metrics samples: " r"(\d+)/(\d+)/(\d+) parsed/fetched/written",
            text,
        )
        if metrics_line is None:
            return ValidationResult(
                "missing inference metrics completion summary",
                True,
            )
        parsed, fetched, written = map(int, metrics_line.groups())
        if not (0 < parsed <= fetched <= written):
            return ValidationResult(
                "invalid inference metrics coverage " f"({parsed}/{fetched}/{written} parsed/fetched/written)",
                True,
            )
        # Pull positive overlap evidence out of the orchestrator's
        # diagnostic line (``Overlap samples: N samples ...``) to
        # include in the PASS message. Falls back to the bare
        # request/error tuple if the line is absent (older log
        # format on a captured-during-rolling-deploy host).
        overlap_line = re.search(r"\[inference_server_stress\] Overlap samples: (\d+) samples", text)
        if overlap_line is None or int(overlap_line.group(1)) <= 0:
            return ValidationResult(
                "missing positive inference overlap evidence",
                True,
            )
        return ValidationResult(
            f"PASS ({total} requests, {errs} errors, {rate}% rate, " f"{overlap_line.group(1)} overlap samples)", False
        )
    return ValidationResult("missing inference request summary", True)


def _validate_hipblaslt(log_file: Path, exit_code: int) -> ValidationResult:  # noqa: C901
    text = _read_log(log_file)
    if not text:
        return ValidationResult("", False)
    # Anchoring each error keyword to a following colon (or a well-known
    # failure phrase) prevents benign prose like "hipBLASLt error handling
    # enabled" from counting as a failure. We keep IGNORECASE so
    # "hipBLASLt" and "hipblaslt" both match, but every pattern requires
    # either a colon or end-of-line immediately after the error keyword.
    hip_err = len(
        re.findall(
            r"HIP error\s*:" r"|\bhipblaslt\s+error\s*:" r"|Segmentation fault",
            text,
            re.IGNORECASE,
        )
    )
    # Counts the literal marker emitted by ``_run_shape`` in
    # tests/hipblaslt_bench.py for both per-shape failure modes (non-zero exit
    # and no data row). The two strings are a contract; both ends carry a note.
    shape_fail = len(re.findall(r"\[hipblaslt\] WARNING: shape", text))
    watchdog = "[hipblaslt] FAIL: watchdog timeout" in text
    data_rows = 0
    result_shapes = []
    # hipblaslt-bench emits a 44-column CSV whose first seven columns are:
    #
    #   0        1        2             3             4   5   6
    #   transA , transB , grouped_gemm, batch_count , m , n , k , alpha, ...
    #
    # So m/n/k are 4/5/6 and batch_count is 3 -- NOT the
    # ``transA,transB,m,n,k,batch_count`` ordering the shape tuples in
    # ``tests/hipblaslt_bench.py`` are written in. The two orderings differ by
    # the ``grouped_gemm`` and ``batch_count`` columns that sit before m, and
    # reading the tuple order onto the CSV silently yields shapes that never
    # match the canonical set. Verified against real run output:
    #   N,N,0,1,8192,320,320,1,8192,...  ->  (N, N, 8192, 320, 320, batch 1)
    HB_TRANS_A, HB_TRANS_B, HB_BATCH, HB_M, HB_N, HB_K = 0, 1, 3, 4, 5, 6  # noqa: N806 — CSV column indices
    for line in text.splitlines():
        if re.match(r"^\s+[NT],[NT],\d", line):
            data_rows += 1
            fields = next(csv.reader([line.strip()]))
            if len(fields) >= 7:
                with contextlib.suppress(TypeError, ValueError):
                    result_shapes.append(
                        (
                            fields[HB_TRANS_A],
                            fields[HB_TRANS_B],
                            int(fields[HB_M]),
                            int(fields[HB_N]),
                            int(fields[HB_K]),
                            int(fields[HB_BATCH]),
                        )
                    )

    if hip_err > 0:
        return ValidationResult(f"{hip_err} HIP/hipblaslt error(s)", True)
    if watchdog:
        return ValidationResult(
            "watchdog timeout (GEMM shape did not complete)",
            True,
        )
    if shape_fail > 0:
        return ValidationResult(f"{shape_fail} shape(s) failed", True)
    if data_rows == 0:
        return ValidationResult("no hipBLASLt GEMM results", True)
    completed = re.search(
        r"Completed:\s*(\d+)/(\d+) shapes passed,\s*(\d+) failed",
        text,
    )
    if completed is None:
        return ValidationResult("missing hipBLASLt completion summary", True)
    passed, total, failed_shapes = map(int, completed.groups())
    canonical_shapes = [("N", "N", m, n, k, batch) for m, n, k, batch in NN_SHAPES] + [
        ("N", "T", m, n, k, batch) for m, n, k, batch in NT_SHAPES
    ]
    if total != len(canonical_shapes):
        return ValidationResult(
            f"unexpected hipBLASLt shape denominator {total}; " f"expected {len(canonical_shapes)}",
            True,
        )
    if passed != total or failed_shapes != 0 or data_rows < total:
        return ValidationResult(
            f"incomplete hipBLASLt coverage ({passed}/{total} shapes, " f"{data_rows} result rows)",
            True,
        )
    if result_shapes != canonical_shapes:
        return ValidationResult(
            "incomplete hipBLASLt shape identity coverage",
            True,
        )
    return ValidationResult(f"PASS ({data_rows} GEMM results)", False)


def _validate_transferbench(  # noqa: C901
    log_file: Path,
    exit_code: int,
    expected_gpus: int | None = None,
) -> ValidationResult:
    """Validate TransferBench rsweep output.

    Aligned with ROCmTestInternal's ``Transfer_bench`` parser: look for
    ``Transfer N | X.XX GB/s | ...`` data lines and ``Aggregate (CPU)``
    summary. Flag ``[ERROR]`` lines as failures.
    """
    text = _read_log(log_file)
    if not text:
        return ValidationResult("", False)
    if text.startswith("_READ_FAILED:"):
        return ValidationResult(f"log unreadable ({text})", True)

    # Watchdog-fired sentinel emitted by ``transferbench.run()`` when
    # ``ctx.exec`` returns rc=124. Without this branch a watchdog kill
    # would still leave behind partial ``Transfer N | … GB/s`` rows
    # in the log, so Layer 2 would happily report "PASS (N transfers,
    # aggregate OK)" even though the sweep was killed mid-run.
    # Surface the watchdog cleanly, matching the convention used in
    # ``_validate_memtest`` / ``_validate_sln_stress``. The substring is
    # anchored on "[transferbench] FAIL: watchdog timeout" so an operator
    # ``grep -F "FAIL: watchdog timeout"`` across the tests' logs finds
    # every watchdog kill uniformly.
    if "[transferbench] FAIL: watchdog timeout" in text:
        return ValidationResult(
            "watchdog timeout (rsweep did not complete)",
            True,
        )

    errors = len(re.findall(r"\[ERROR\]\s*\w+", text))
    if errors > 0:
        return ValidationResult(f"{errors} TransferBench error(s)", True)

    # Count transfer data lines: "Transfer N | X.XX GB/s | ..."
    # Newer TransferBench versions use Unicode box-drawing │ (U+2502)
    # instead of ASCII | (U+007C), so accept both.
    transfers = len(re.findall(r"Transfer\s+\d+\s*[│|]\s*[\d.]+\s*GB/s", text))
    # Check for aggregate summary
    has_aggregate = bool(re.search(r"Aggregate\s*\(CPU\)\s*[│|]\s*[\d.]+\s*GB/s", text))

    if transfers == 0:
        return ValidationResult("no transfer results in output", True)
    if not has_aggregate:
        return ValidationResult(
            f"incomplete rsweep ({transfers} transfer(s), aggregate missing)",
            True,
        )
    if "[transferbench] Completed rsweep successfully (rc=0)" not in text:
        return ValidationResult("missing TransferBench completion sentinel", True)
    gpu_count_match = re.search(
        r"NUM_GPU_DEVICES\s*=\s*(\d+)\s*:\s*Using\s+(\d+)\s+GPUs",
        text,
    )
    if gpu_count_match is None:
        return ValidationResult(
            "TransferBench reported no usable GPU devices",
            True,
        )
    detected, used = map(int, gpu_count_match.groups())
    if detected <= 0 or used <= 0:
        return ValidationResult("TransferBench reported no usable GPU devices", True)
    if expected_gpus is not None and (detected != expected_gpus or used != expected_gpus):
        return ValidationResult(
            "TransferBench GPU count mismatch " f"(detected {detected}, used {used}, expected {expected_gpus})",
            True,
        )
    endpoints = [
        (int(match.group(1)), int(match.group(2)))
        for line in text.splitlines()
        if re.search(r"Transfer\s+\d+\s*[│|]", line)
        for match in [re.search(r"\bG(\d+)\s*->.*->\s*G(\d+)\b", line)]
        if match is not None
    ]
    if not any(source != destination for source, destination in endpoints):
        return ValidationResult(
            "no inter-GPU transfer route found in rsweep output",
            True,
        )
    if expected_gpus is not None:
        seen_gpus = {gpu for endpoints_pair in endpoints for gpu in endpoints_pair}
        expected = set(range(expected_gpus))
        if seen_gpus != expected:
            return ValidationResult(
                "incomplete TransferBench per-GPU route coverage "
                f"(reported {sorted(seen_gpus)}, expected {sorted(expected)})",
                True,
            )
    return ValidationResult(
        f"PASS ({transfers} transfer(s), aggregate OK)",
        False,
    )


# Validator registry: test_name -> function
_VALIDATORS: dict[str, Callable[[Path, int], ValidationResult]] = {
    "cudamemtest": _validate_cudamemtest,
    "hmm_cuda_memtest": _validate_hmm_memtest,
    "rvs_iet_stress": _validate_rvs,
    "rvs_tst": _validate_rvs,
    "sln_stress": _validate_sln_stress,
    "power_band_stress": _validate_power_band,
    "hipblaslt_bench": _validate_hipblaslt,
    "inference_server_stress": _validate_inference_server,
    "transferbench": _validate_transferbench,
}


def unsupported_reason_from_log(log_file: Path, test_name: str) -> str:
    """Return the workload's UNSUPPORTED explanation from ``console.log``.

    Workloads print a human-readable ``UNSUPPORTED: …`` line before returning
    ``RunResult(unsupported=True)``.  The orchestrator used to replace that
    with the bare word ``UNSUPPORTED``, which made pytest reports useless.
    """
    if log_file.is_file():
        text = _read_log(log_file)
        if not text.startswith("_READ_FAILED:"):
            for line in text.splitlines():
                stripped = line.strip()
                if "UNSUPPORTED:" in stripped and len(stripped) > len("UNSUPPORTED:"):
                    return stripped
    return f"{test_name}: unsupported on this device"


# ---------------------------------------------------------------------------
# Main entry point (replaces shell `validate_result`)
# ---------------------------------------------------------------------------
def validate_result(  # noqa: C901
    test_name: str,
    log_file: Path,
    exit_code: int,
    dmesg_file: Path | None = None,
    *,
    skip_test_specific: bool = False,
    num_gpus: int | None = None,
) -> ValidationResult:
    """Run the 5-layer validation and return (message, failed).

    - Layer 1 sets failed=True if crash patterns found in test output
    - Layer 2 can add to message and/or set failed=True
    - Layer 3 critical dmesg events (GPU reset, panic, lockup) set
      failed=True; informational events are annotated only
    - Layer 4: empty log + non-zero exit sets failed=True
    - Layer 5 prefixes a non-zero exit / signal onto the message so it
      agrees with the status, and rewrites a leftover Layer 2 ``PASS (...)``
      as ``workload OK (...)`` once any layer has failed

    The monitoring-evidence gates (samples, per-GPU identity, sensor
    coverage, activity, telemetry spanning the run) live in
    ``runner.run_one`` rather than here, because they read
    ``power_temp.csv`` rather than the workload's log.
    """
    msg = ""
    failed = False

    # Layer 1: Crash / kernel indicators
    crash_count = 0
    text = ""
    if log_file.is_file():
        text = _read_log(log_file)
        if text.startswith("_READ_FAILED:"):
            msg = f"log unreadable ({text})"
            failed = True
            return ValidationResult(msg, failed)
        crash_count = _count_lines_matching(text, CRASH_PATTERNS)
    if crash_count > 0:
        msg = f"crash/kernel indicator in output ({crash_count} matches)"
        failed = True

    # Layer 2: Test-specific validation
    validator = _VALIDATORS.get(test_name)
    if not skip_test_specific and validator is not None and log_file.is_file():
        if test_name in {"rvs_iet_stress", "rvs_tst"} and num_gpus is not None:
            if num_gpus <= 0:
                res = ValidationResult(
                    "RVS GPU count unavailable; cannot validate per-GPU coverage",
                    True,
                )
            else:
                checks_per_gpu = 1 if test_name == "rvs_iet_stress" else 2
                res = _validate_rvs(
                    log_file,
                    exit_code,
                    expected_checks=num_gpus * checks_per_gpu,
                    expected_gpus=num_gpus,
                    checks_per_gpu=checks_per_gpu,
                    # Both action names and ``checks_per_gpu`` are pinned to
                    # the shape of the upstream ``tst_single.conf`` that ROCm
                    # ships: it declares two actions named ``action_1`` and
                    # ``action_2``, so a full run logs two checks per GPU.
                    # Verified against the confs on disk (the generic
                    # top-level file and the MI210 variant are the only two
                    # that exist) and against real MI300A/MI325X runs on ROCm
                    # 7.14 and 10.1. If upstream ever renames or adds an
                    # action, both this set and ``checks_per_gpu`` above must
                    # change together, and ``RVS_TST_EXPECTED_ACTIONS`` is
                    # what the regression test pins.
                    expected_actions=(RVS_TST_EXPECTED_ACTIONS if test_name == "rvs_tst" else None),
                )
        elif test_name == "hmm_cuda_memtest" and num_gpus is not None:
            res = _validate_hmm_memtest(log_file, exit_code, num_gpus)
        elif test_name == "transferbench" and num_gpus is not None:
            res = _validate_transferbench(log_file, exit_code, num_gpus)
        else:
            res = validator(log_file, exit_code)
        if res.failed:
            # Guard against a validator that returns `failed=True` with an
            # empty message — the summary table would otherwise show a
            # blank Validation column for a real failure.
            layer2_msg = res.message or f"{test_name} validator flagged failure"
            msg = f"{msg}; {layer2_msg}" if msg else layer2_msg
            failed = True
        elif res.message:
            msg = msg or res.message

    # Layer 3: dmesg — critical kernel events FAIL the test; informational
    # events are annotated but don't flip pass/fail. This is the key
    # differentiator from ROCmTestInternal: a GPU that silently reset and
    # recovered mid-test will report PASS from the test's own stdout, but
    # dmesg captures the hardware-level fault. Without this, operators
    # would need to manually cross-reference dmesg after every run.
    if dmesg_file and dmesg_file.is_file() and dmesg_file.stat().st_size > 0:
        dmesg_text = _read_log(dmesg_file)
        if dmesg_text.startswith("_READ_FAILED:"):
            dmesg_note = f"dmesg unreadable ({dmesg_text})"
            msg = f"{msg}; {dmesg_note}" if msg else dmesg_note
            failed = True
        else:
            snapshot_unavailable = (
                "[dmesg-delta] pretest snapshot unavailable" in dmesg_text
                or "[dmesg-delta] posttest snapshot unavailable" in dmesg_text
            )
            critical_count = _count_lines_matching(dmesg_text, DMESG_CRITICAL)
            info_count = _count_lines_matching(dmesg_text, DMESG_INFO)
            if snapshot_unavailable:
                dmesg_note = "dmesg: snapshot unavailable; cannot validate " "kernel health for this workload"
                msg = f"{msg}; {dmesg_note}" if msg else dmesg_note
                failed = True
            elif critical_count > 0:
                dmesg_note = (
                    f"dmesg: {critical_count} CRITICAL kernel event(s) "
                    f"(GPU reset / panic / lockup / fault) — see dmesg.log"
                )
                msg = f"{msg}; {dmesg_note}" if msg else dmesg_note
                failed = True
            elif info_count > 0:
                dmesg_note = f"dmesg: {info_count} kernel message(s) — see dmesg.log"
                msg = f"{msg}; {dmesg_note}" if msg else dmesg_note

    # Layer 4: an empty/missing captured log is never positive test evidence.
    if not log_file.is_file() or log_file.stat().st_size == 0:
        if exit_code != 0:
            if not msg:
                msg = f"exit {exit_code} with no output"
            failed = True
        else:
            if not msg:
                msg = "empty output; cannot confirm workload completion"
            failed = True

    # Layer 5: Exit-code consistency. None of layers 1-4 are required
    # to consult ``exit_code`` (most don't), so a workload that gets
    # SIGKILLed mid-run with a clean truncated stdout and no critical
    # dmesg event would otherwise produce ``PASS (0 memory errors)``
    # in the validation column despite ``status=FAIL`` from the
    # non-zero exit. Always surface the signal/exit when non-zero so
    # the message agrees with the ``exit_code`` column.
    #
    # We previously only fired this when no earlier layer had failed,
    # to avoid redundant noise. That mis-served the
    # ``hmm_cuda_memtest`` MI210 OOM case (dmesg's ``Out of memory:``
    # event tripped Layer 3 *and* SIGKILL gave us ``exit_code=-9``):
    # the message read ``"workload OK (0 memory errors); dmesg: 1
    # CRITICAL kernel event(s)..."`` with no hint the process had
    # been killed mid-run, so a triager couldn't tell from the
    # validation column alone whether the workload completed or was
    # aborted. Prepending the exit/signal unconditionally makes
    # every failure mode self-describing in the message and removes
    # the need to cross-reference the separate ``exit_code`` field.
    if exit_code != 0:
        exit_note = f"killed by signal {-exit_code}" if exit_code < 0 else f"exit {exit_code}"
        msg = f"{exit_note}" + (f"; {msg}" if msg else "")
        failed = True

    # Rewrite the misleading ``PASS (...)`` segment when a later layer
    # marked the test as failed. Layer 2 validators commit a "PASS
    # (0 memory errors)" / "PASS (19 GEMM results)" message before
    # Layer 3 (dmesg) and Layer 4 (empty log) get to vote, which
    # produced summaries like:
    #   "PASS (0 memory errors); dmesg: 1 CRITICAL kernel event(s)..."
    # The reader has to infer that the leading "PASS" referred only
    # to the workload's own internal counters, not the overall result.
    # Re-prefixing as "workload OK (...)" preserves the workload's
    # self-report while making clear that something *else* failed.
    # ``\b`` matches start-of-string too, so this single regex handles
    # both the leading "PASS (...)" case (Layer 2 message alone) and
    # the mid-message case ("exit 1; PASS (...)" after Layer 5
    # prepended an exit note).
    if failed:
        msg = re.sub(r"\bPASS \(", "workload OK (", msg)

    if not msg:
        msg = "PASS"

    return ValidationResult(msg, failed)


# ---------------------------------------------------------------------------
# Pre-test health probe (Design B)
# ---------------------------------------------------------------------------
# Per-category narrowings of ``DMESG_CRITICAL`` for the pre-test probe.
# Same severity classes as ROCmTestInternal's ``KERNEL_ALERT_RULES``
# critical bucket so triage tooling that already understands category
# names (gpu_reset, kernel_panic, watchdog_lockup, ...) can consume our
# ``pretest_health.json`` directly. Layer 3 still uses the flat
# ``DMESG_CRITICAL`` regex; this categorization exists purely to make
# the pre-test probe's output actionable rather than emitting an opaque
# "N critical events" count.
DMESG_CATEGORY_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("kernel_panic", re.compile(r"\bKernel panic\b|\bpanic\b", re.IGNORECASE)),
    (
        "watchdog_lockup",
        re.compile(
            r"watchdog:.*\b(?:BUG|soft lockup|hard lockup)\b|" r"\bsoft lockup\b|\bhard lockup\b", re.IGNORECASE
        ),
    ),
    ("bug_oops", re.compile(r"\bBUG:|\bkernel BUG\b|Oops:\s*[0-9a-fA-F]+", re.IGNORECASE)),
    ("call_trace", re.compile(r"\bCall Trace\b|\bcall trace\b", re.IGNORECASE)),
    ("hung_task", re.compile(r"\bblocked for more than\b|\bhung task\b", re.IGNORECASE)),
    (
        "hardware_uncorrected",
        re.compile(
            r"\b(?:uncorrected|uncorrectable|fatal)\b.*"
            r"\b(?:hardware error|ECC|RAS|error)\b|"
            r"\b(?:hardware error|ECC|RAS)\b.*"
            r"\b(?:uncorrected|uncorrectable|fatal)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gpu_reset",
        re.compile(
            r"\bgpu reset\b|\bamdgpu.*reset\b|\bring .*timeout\b|" r"\bGPU fault\b|\bRAS\b.*\berror\b", re.IGNORECASE
        ),
    ),
    ("iommu_fault", re.compile(r"\bIOMMU\b.*\bfault\b|\bDMAR\b.*\bfault\b", re.IGNORECASE)),
    ("segfault", re.compile(r"\bsegfault\b|\bsegmentation fault\b", re.IGNORECASE)),
]


def categorize_dmesg_critical(text: str) -> dict[str, int]:
    """Per-category counts of DMESG_CRITICAL matches in ``text``.

    Returns a dict with stable keys (every category present, zero when
    not matched) so dashboards can chart consistently across runs and
    consumers can rely on every category being present in
    ``pretest_health.json``. The first matching category wins per line
    — same convention as ROCmTestInternal's ``KERNEL_ALERT_RULES`` —
    so a line containing both ``BUG:`` and ``Call Trace`` is counted
    under ``bug_oops`` only.
    """
    counts: dict[str, int] = {cat: 0 for cat, _ in DMESG_CATEGORY_RULES}
    if not text:
        return counts
    for line in text.splitlines():
        candidate = ras_downgrade(line)
        if candidate is None:
            continue
        for cat, pat in DMESG_CATEGORY_RULES:
            if pat.search(candidate):
                counts[cat] += 1
                break
    return counts


_DMESG_TS_RE = re.compile(r"^\[([^\]]+)\]")


def _parse_dmesg_ts(line: str) -> datetime | None:
    """Parse a ``dmesg -T`` timestamp from the start of ``line``.

    Returns ``None`` when there's no timestamp (raw kernel ring) or
    when the format is unparseable. ``%a %b %e`` is the
    space-padded-day variant some locales emit (e.g. ``Apr  8`` with
    two spaces); we accept both.
    """
    m = _DMESG_TS_RE.match(line)
    if not m:
        return None
    ts_text = m.group(1)
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b %e %H:%M:%S %Y"):
        try:
            return datetime.strptime(ts_text, fmt)
        except ValueError:
            continue
    return None


def _filter_dmesg_recent(text: str, cutoff: datetime) -> str:
    """Keep dmesg lines whose timestamp is >= ``cutoff``.

    Lines without a parseable timestamp are kept (conservative): we'd
    rather over-report a stale event than miss a fresh one. The
    pre-test probe's output is informational by default (operators
    decide when to opt in to ``--strict-pretest-gate``), so a small
    over-report is preferable to silent under-report.
    """
    keep: list[str] = []
    for line in text.splitlines():
        ts = _parse_dmesg_ts(line)
        if ts is None or ts >= cutoff:
            keep.append(line)
    return "\n".join(keep)


def pretest_health_probe(
    lookback_min: int = 30,
    cpu_executor: CpuExecutor | None = None,
) -> tuple[bool, dict]:
    """Probe pre-existing dmesg for critical events in the last N minutes.

    Returns ``(clean, summary)``:

    * ``clean`` — ``True`` iff no DMESG_CRITICAL matches were found
      in the lookback window. Probe failures (dmesg unavailable,
      timeout, non-zero return code) return ``True`` so the probe
      never blocks a run on its own bug; opt-in strictness via
      ``--strict-pretest-gate`` only fires on confirmed dirty state.
    * ``summary`` — dict suitable for ``json.dump`` into
      ``pretest_health.json``: probe status, lookback window, total
      critical count, per-category counts (stable keys), and up to 5
      sample matching lines for triage.
    """
    summary: dict = {
        "probe": "skipped",
        "lookback_min": lookback_min,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "critical_total": 0,
        "by_category": {cat: 0 for cat, _ in DMESG_CATEGORY_RULES},
        "samples": [],
        "reason": None,
    }
    available, stdout = capture_dmesg_text(cpu_executor)
    if not available or not stdout:
        summary["reason"] = "dmesg capture unavailable (empty stdout)"
        return True, summary

    cutoff = datetime.now() - timedelta(minutes=lookback_min)
    recent = _filter_dmesg_recent(stdout, cutoff)
    counts = categorize_dmesg_critical(recent)
    total = sum(counts.values())
    summary["probe"] = "ok"
    summary["critical_total"] = total
    summary["by_category"] = counts

    # Capture up to 5 sample matching lines so the operator can triage
    # without re-grepping dmesg manually. Each sample is the first
    # occurrence per category, in DMESG_CATEGORY_RULES order.
    samples: list[str] = []
    for cat, pat in DMESG_CATEGORY_RULES:
        if counts[cat] == 0:
            continue
        for line in recent.splitlines():
            if pat.search(line):
                samples.append(line.strip())
                break
        if len(samples) >= 5:
            break
    summary["samples"] = samples

    return total == 0, summary
