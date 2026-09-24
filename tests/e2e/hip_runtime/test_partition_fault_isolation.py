# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
test_partition_fault_isolation.py -- HIP_VISIBLE_DEVICES / partition fault isolation regression.

Validates:
    1. hipGetDeviceCount() returns >= 2 on the test node; otherwise skip.
       (On AMD Instinct systems with compute partitioning such as CPX, one
       physical GPU exposes multiple logical HIP devices. The test is equally
       valid on any multi-GPU host where N >= 2 logical devices are visible.)

    2. Each fault-injection scenario (oob-write, oob-read, null-deref) runs
       with HIP_VISIBLE_DEVICES=<buggy_hip_idx> (= hip_n - 1), produces a
       non-success sync result (the fault is observable), exits 0, and prints
       the SUITE_MARKER sentinel.

    3. One golden_workload background process per HIP index in 0..N-2 runs
       concurrently throughout all buggy scenarios, each restricted to a
       disjoint HIP device via HIP_VISIBLE_DEVICES.  All golden processes must
       survive every buggy scenario and complete their full timed run with
       exit 0, demonstrating that faults on one partition do not crash or stall
       workloads on disjoint partitions.

Architecture of the port:
    ``hw.multi_gpu`` + ``gpu_count("all")`` cause ``target_executor`` to acquire
    every GPU slot available on one node, so the framework sets
    ``ROCR_VISIBLE_DEVICES`` to the full allocated set for every process launched
    through it.  Within that namespace, HIP renumbers the devices from 0..N-1.

    The test layers ``HIP_VISIBLE_DEVICES`` on top of the framework-managed
    ``ROCR_VISIBLE_DEVICES`` to give each process exclusive single-device
    visibility — this is the mechanism under test.

    With gpu_count("all") the index layout is:
      - normal_indices = [0, 1, ..., N-2]  → N-1 concurrent golden processes
      - buggy_hip_idx  = N-1               → one buggy process

    Each C++ binary receives ``--device 0`` because HIP_VISIBLE_DEVICES already
    restricts visibility to a single device; ``hipSetDevice(0)`` addresses that
    one visible device.  The ``--device N`` parameter is preserved in both
    binaries for standalone use without HIP_VISIBLE_DEVICES.

    Both processes are launched through ``target_executor`` so they land on the
    correct node in both local and remote mode.

    ``HIP_VISIBLE_DEVICES`` is set in the ``env ...`` command prefix because it
    is the mechanism under test — this is intentional and does not violate the
    framework rule against setting ``ROCR_VISIBLE_DEVICES`` in test code.

Sentinel assertions:
    - ``hip_device_count`` stdout: integer >= 2 (skip otherwise).
    - Each ``buggy_workload`` run: exit 0 AND ``[BUGGY] SUITE_MARKER success scenario=<name>``.
    - All ``golden_workload`` background processes: exit 0 AND ``GOLDEN_OK`` in stdout.

    hw.gpu is the profile default; hw.multi_gpu overrides it at function level
    to acquire all GPU slots on one node so ROCR_VISIBLE_DEVICES reflects the
    full target topology.
"""

from __future__ import annotations

import time

import pytest

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------

# Estimated wall time of the full buggy suite (3 scenarios x ~55 s each).
# The golden workload uses 4x this as its GOLDEN_SECONDS so it outlasts buggy.
_BUGGY_EST_SECS = 55
_GOLDEN_FACTOR = 4
_GOLDEN_SECS = _GOLDEN_FACTOR * _BUGGY_EST_SECS  # 220 s

# Per-scenario timeout: 4x the per-GPU estimate + 60 s executor overhead buffer.
_SCENARIO_TIMEOUT_SECS = _BUGGY_EST_SECS * 4 + 60  # 280 s

# Overall test timeout: golden duration + scenario total + startup headroom.
_TEST_TIMEOUT_SECS = _GOLDEN_SECS + (3 * _SCENARIO_TIMEOUT_SECS) + 120  # 1,180 s

# Ordered buggy scenarios — matches the original shell script's sequence.
_BUGGY_SCENARIOS = ("oob-write", "oob-read", "null-deref")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_device_count(stdout: str) -> int | None:
    """Return the integer device count from ``hip_device_count`` stdout, or None on parse failure.

    Args:
        stdout: Captured stdout from the ``hip_device_count`` binary.

    Returns:
        Parsed integer, or ``None`` when the output is not a valid positive integer.
    """
    line = stdout.strip().splitlines()[0].strip() if stdout.strip() else ""
    try:
        n = int(line)
        return n if n > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------


@pytest.mark.hw.multi_gpu
@pytest.mark.gpu_count("all")
@pytest.mark.runtime.medium
def test_partition_fault_isolation(
    request,
    target_executor,
    ld_path: dict,
    golden_workload_binary: str,
    buggy_workload_binary: str,
    hip_device_count_binary: str,
):
    """Validate HIP partition isolation: faults on one logical device must not affect others.

    On AMD systems with compute partitioning (CPX / DPX / QPX / SPX on MI300X, or
    any multi-GPU host with N >= 2 logical HIP devices), this test checks that:

    * A fault-injection binary (oob-write, oob-read, null-deref) launched with
      ``HIP_VISIBLE_DEVICES=<hip_n-1>`` exits 0 with the expected fault-observed sentinel.
    * One golden SAXPY loop per HIP index in 0..N-2, each launched with the
      corresponding ``HIP_VISIBLE_DEVICES=<idx>``, survives the full timed run and
      exits 0 — demonstrating isolation across all partitions simultaneously.

    Requires >= 2 HIP logical devices.  Skips gracefully when fewer devices are
    available (single-GPU hosts without partitioning).

    The ``hw.multi_gpu`` + ``gpu_count("all")`` markers cause ``target_executor``
    to acquire all GPU slots on one node, setting ``ROCR_VISIBLE_DEVICES`` to the
    full allocated set.  The test then layers per-process ``HIP_VISIBLE_DEVICES``
    on top to give each process exclusive single-device visibility — this is the
    ``HIP_VISIBLE_DEVICES`` isolation mechanism under test.  All processes go
    through ``target_executor`` for remote-mode compatibility.  Each binary
    receives ``--device 0`` because HIP_VISIBLE_DEVICES already restricts
    visibility to a single device.
    """
    ld = ld_path["LD_LIBRARY_PATH"]

    # ------------------------------------------------------------------
    # Step 1: Query device count (without HIP_VISIBLE_DEVICES override so
    # all framework-allocated slots are visible).  Skip when < 2 devices.
    # ------------------------------------------------------------------
    print("[STAGE: device-count] Querying HIP logical device count")
    count_result = target_executor.run(
        f"env LD_LIBRARY_PATH={ld} {hip_device_count_binary}",
    )
    assert count_result.ok, (
        f"hip_device_count failed (exit={count_result.exit_code}):\n"
        f"stdout: {count_result.stdout[:500]}\nstderr: {count_result.stderr[:500]}"
    )
    hip_n = _parse_device_count(count_result.stdout)
    if hip_n is None or hip_n < 2:
        pytest.skip(
            f"Partition isolation requires >= 2 HIP logical devices; "
            f"hipGetDeviceCount()={hip_n!r}. "
            f"Run on a system with compute partitioning (e.g. MI300X CPX) or multiple GPUs."
        )
    print(f"[STAGE: device-count] PASS — hipGetDeviceCount()={hip_n}")

    buggy_hip_idx = hip_n - 1
    normal_indices = list(range(buggy_hip_idx))  # 0..N-2

    # ------------------------------------------------------------------
    # Step 2: Launch one golden workload background process per normal HIP
    # index (0..N-2), matching the original shell script's GOLDEN_ARR loop.
    # HIP_VISIBLE_DEVICES=<idx> restricts each process to a single HIP device;
    # --device 0 addresses that one visible device inside the binary.
    # gpu_count("all") makes this one golden process per non-buggy visible HIP
    # device.
    # ------------------------------------------------------------------
    print(
        f"[STAGE: golden-launch] Starting {len(normal_indices)} golden worker(s) "
        f"(HIP_VISIBLE_DEVICES={list(normal_indices)}, GOLDEN_SECONDS={_GOLDEN_SECS})"
    )
    bg_processes: list[tuple[int, object]] = []

    def _cleanup_background_workers() -> None:
        """Stop any golden workers left running after a failure."""
        for dev_hip_idx, bg in bg_processes:
            if bg.is_alive:
                print(f"[STAGE: cleanup] stopping golden_workload dev={dev_hip_idx}")
                bg.stop()

    request.addfinalizer(_cleanup_background_workers)

    for dev_hip_idx in normal_indices:
        cmd = (
            f"env LD_LIBRARY_PATH={ld} "
            f"HIP_VISIBLE_DEVICES={dev_hip_idx} "
            f"GOLDEN_SECONDS={_GOLDEN_SECS} "
            f"{golden_workload_binary} --device 0"
        )
        bg = target_executor.start_background(
            cmd,
            log_path=f"output/artifacts/hip_runtime/golden_workload_dev{dev_hip_idx}.log",
        )
        bg_processes.append((dev_hip_idx, bg))
        print(f"[STAGE: golden-launch] golden_workload dev={dev_hip_idx} started (pid={bg.pid})")

    # Allow golden processes to reach their SAXPY loop before proceeding.
    time.sleep(2)

    # Verify every golden process is alive after startup.
    for dev_hip_idx, bg in bg_processes:
        assert bg.is_alive, (
            f"golden_workload (HIP_VISIBLE_DEVICES={dev_hip_idx}) died during startup. "
            f"Check output/artifacts/hip_runtime/golden_workload_dev{dev_hip_idx}.log"
        )
    print(f"[STAGE: golden-launch] PASS — all {len(bg_processes)} golden worker(s) alive after startup")

    # ------------------------------------------------------------------
    # Step 3: Run each buggy scenario sequentially via target_executor.
    # HIP_VISIBLE_DEVICES=<buggy_hip_idx> restricts the process to the last
    # framework-allocated GPU slot.  --device 0 addresses that single visible
    # device inside the binary.  One process per scenario because
    # hipDeviceReset() is per-process.
    # ------------------------------------------------------------------
    print(
        f"[STAGE: fault-injection] Running {len(_BUGGY_SCENARIOS)} buggy scenarios "
        f"on HIP_VISIBLE_DEVICES={buggy_hip_idx}: {list(_BUGGY_SCENARIOS)}"
    )
    for scenario_idx, scenario in enumerate(_BUGGY_SCENARIOS, start=1):
        print(
            f"[SCENARIO {scenario_idx}/{len(_BUGGY_SCENARIOS)}: {scenario}] "
            f"Starting — verifying golden workers are alive"
        )
        # Verify all golden workers are still alive before each scenario.
        for dev_hip_idx, bg in bg_processes:
            assert bg.is_alive, (
                f"golden_workload (HIP_VISIBLE_DEVICES={dev_hip_idx}) died before "
                f"buggy scenario '{scenario}'. Isolation has failed."
            )

        buggy_result = target_executor.run(
            f"env LD_LIBRARY_PATH={ld} "
            f"HIP_VISIBLE_DEVICES={buggy_hip_idx} "
            f"{buggy_workload_binary} --only {scenario} --device 0",
            timeout=_SCENARIO_TIMEOUT_SECS,
        )

        # Exit 10 means the fault was not observed — the test itself is invalid.
        assert buggy_result.exit_code != 10, (
            f"buggy_workload scenario '{scenario}' did not observe a fault (exit 10). "
            f"The isolation assertion cannot be made when buggy does not fault as expected.\n"
            f"stdout: {buggy_result.stdout[:1000]}\nstderr: {buggy_result.stderr[:1000]}"
        )

        assert buggy_result.ok, (
            f"buggy_workload scenario '{scenario}' failed (exit={buggy_result.exit_code}):\n"
            f"stdout: {buggy_result.stdout[:1000]}\nstderr: {buggy_result.stderr[:1000]}"
        )

        # Verify the authoritative success sentinel.
        expected_marker = f"[BUGGY] SUITE_MARKER success scenario={scenario}"
        assert expected_marker in buggy_result.stdout, (
            f"buggy_workload scenario '{scenario}' did not emit the expected sentinel.\n"
            f"Expected: {expected_marker!r}\n"
            f"stdout: {buggy_result.stdout[:1000]}"
        )

        # Verify all golden workers survived the buggy scenario.
        for dev_hip_idx, bg in bg_processes:
            assert bg.is_alive, (
                f"golden_workload (HIP_VISIBLE_DEVICES={dev_hip_idx}) died during "
                f"buggy scenario '{scenario}'. "
                f"Partition isolation FAILED: fault on HIP device {buggy_hip_idx} "
                f"affected HIP device {dev_hip_idx}."
            )

        print(
            f"[SCENARIO {scenario_idx}/{len(_BUGGY_SCENARIOS)}: {scenario}] "
            f"PASS — fault observed (exit={buggy_result.exit_code}), "
            f"sentinel found, all {len(bg_processes)} golden worker(s) alive"
        )

    # ------------------------------------------------------------------
    # Step 4: Wait for each golden background process to exit naturally,
    # then verify the result.  The golden workload runs for _GOLDEN_SECS
    # (220 s by default); all buggy scenarios typically complete in seconds.
    # Calling bg.stop() immediately after the buggy phase would send SIGTERM
    # to a still-running golden process, causing exit -1 and a missing
    # GOLDEN_OK sentinel.  Instead, poll until the process exits on its own
    # before collecting results — stop() skips the kill when the process has
    # already exited.
    # ------------------------------------------------------------------
    print(
        f"[STAGE: golden-join] Waiting for {len(bg_processes)} golden worker(s) to complete "
        f"(budget: {_GOLDEN_SECS + 60:.0f}s)"
    )
    poll_interval = 5.0  # seconds between liveness checks
    wait_deadline = _GOLDEN_SECS + 60.0  # budget: golden duration + 60 s headroom

    for dev_hip_idx, bg in bg_processes:
        waited = 0.0
        while bg.is_alive and waited < wait_deadline:
            time.sleep(poll_interval)
            waited += poll_interval

        stopped = bg.stop()
        assert stopped.ok, (
            f"golden_workload (HIP_VISIBLE_DEVICES={dev_hip_idx}) exited non-zero "
            f"(exit={stopped.exit_code}). NORMAL workload did not complete cleanly.\n"
            f"stdout: {stopped.stdout[:1000]}\nstderr: {stopped.stderr[:500]}"
        )
        assert "GOLDEN_OK" in stopped.stdout, (
            f"golden_workload (HIP_VISIBLE_DEVICES={dev_hip_idx}) did not emit "
            f"'GOLDEN_OK' sentinel.\nstdout: {stopped.stdout[:1000]}"
        )
        print(
            f"[STAGE: golden-join] dev={dev_hip_idx} PASS — "
            f"exit={stopped.exit_code}, GOLDEN_OK sentinel found (waited {waited:.0f}s)"
        )
